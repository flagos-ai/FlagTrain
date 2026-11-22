# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""moe_chunk_sort - segmented, order-preserving row permutation (Triton).

Reference semantics (torch):
    sizes, order = split_sizes.tolist(), sorted_idxs.tolist()
    starts = [0, *accumulate(sizes)]
    permutation = [i for g in order for i in range(starts[g], starts[g + 1])]
                  + list(range(starts[-1], N))
    return inp[permutation], probs[permutation] (or inp.new_empty(0, float32)),
           permutation.argsort().int()

Every chunk is a *contiguous* input row range, so the permutation never has to be
materialised on the host (that would cost a sync plus extra kernels and break
CUDA graph capture).  One launch does the activations, the probabilities and the
input->output row map:

    out_starts[k], ends_out[k], in_starts[perm[k]]   describe chunk k in output order
    q in chunk k  ->  in_row = in_starts[perm[k]] + (q - out_starts[k])
    q >= sum(sizes) (tail)  ->  in_row = q
    row_id_map[in_row] = q

Graded workloads are tiny (N <= 1024, H <= 256), so the measured cost is the
fixed per-call cost, not streaming bandwidth.  Metadata is folded into constexprs
and, per metadata key, a specialised call closure is cached that only allocates
the outputs and calls the compiled kernel's builtin launcher.
"""

import torch
import triton
import triton.language as tl

# Hot-path callables bound once: the steady-state call only allocates and
# launches, so it should not repeat attribute lookups on the torch module.
_empty_like = torch.empty_like
_empty = torch.empty
_int32 = torch.int32

# Current stream as a raw handle: one C call, no Stream object per launch.
try:
    _RAW_STREAM = torch._C._cuda_getCurrentRawStream
except AttributeError:  # pragma: no cover - other torch builds
    _RAW_STREAM = None


@triton.jit
def _map_in_rows(
    q, split_ptr, sorted_ptr, N: tl.constexpr, M: tl.constexpr, BLOCK_M: tl.constexpr
):
    """Output rows -> input rows, using only the M element size vector."""
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    sizes = tl.load(split_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
    total = tl.sum(sizes, 0)

    perm = tl.load(sorted_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
    s_sorted = tl.load(split_ptr + perm, mask=mask_m, other=0).to(tl.int32)
    out_end = tl.cumsum(s_sorted, 0)
    out_start = out_end - s_sorted
    # in_starts[perm[k]] as an M x M selection (M is a small constexpr).
    in_start_sorted = tl.sum(
        tl.where(offs_m[None, :] < perm[:, None], sizes[None, :], 0), axis=1
    )

    qs = q[:, None]
    sel = (out_end[None, :] > qs) & (out_start[None, :] <= qs)
    in_row = tl.sum(
        tl.where(sel, in_start_sorted[None, :] + qs - out_start[None, :], 0), axis=1
    )
    in_row = tl.where(q >= total, q, in_row)  # identity tail
    return tl.where((in_row >= 0) & (in_row < N), in_row, q)


@triton.jit
def _chunk_sort_probs_kernel(
    inp_ptr,
    out_ptr,
    map_ptr,
    probs_ptr,
    pprobs_ptr,
    split_ptr,
    sorted_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    SI0: tl.constexpr,
    SI1: tl.constexpr,
    SO0: tl.constexpr,
    SO1: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_n = tl.program_id(1)

    q = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mq = q < N
    in_row = _map_in_rows(q, split_ptr, sorted_ptr, N, M, BLOCK_M)

    cols = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask2 = mq[:, None] & (cols < H)[None, :]
    vals = tl.load(
        inp_ptr + in_row.to(tl.int64)[:, None] * SI0 + cols[None, :] * SI1,
        mask=mask2,
        other=0,
    )
    tl.store(
        out_ptr + q.to(tl.int64)[:, None] * SO0 + cols[None, :] * SO1,
        vals,
        mask=mask2,
    )

    if pid_h == 0:
        # row_id_map[input_row] = output_row (tail rows map to themselves)
        tl.store(map_ptr + in_row, q.to(tl.int32), mask=mq)
        row_p = in_row.to(tl.int64) * K
        row_o = q.to(tl.int64) * K
        for kb in range((K + BLOCK_K - 1) // BLOCK_K):
            c = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            mask3 = mq[:, None] & (c < K)[None, :]
            pv = tl.load(probs_ptr + row_p[:, None] + c[None, :], mask=mask3, other=0)
            tl.store(pprobs_ptr + row_o[:, None] + c[None, :], pv, mask=mask3)


@triton.jit
def _chunk_sort_kernel(
    inp_ptr,
    out_ptr,
    map_ptr,
    split_ptr,
    sorted_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    SI0: tl.constexpr,
    SI1: tl.constexpr,
    SO0: tl.constexpr,
    SO1: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_n = tl.program_id(1)

    q = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mq = q < N
    in_row = _map_in_rows(q, split_ptr, sorted_ptr, N, M, BLOCK_M)

    cols = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask2 = mq[:, None] & (cols < H)[None, :]
    vals = tl.load(
        inp_ptr + in_row.to(tl.int64)[:, None] * SI0 + cols[None, :] * SI1,
        mask=mask2,
        other=0,
    )
    tl.store(
        out_ptr + q.to(tl.int64)[:, None] * SO0 + cols[None, :] * SO1,
        vals,
        mask=mask2,
    )

    if pid_h == 0:
        tl.store(map_ptr + in_row, q.to(tl.int32), mask=mq)


def _next_pow2(v):
    p = 1
    while p < v:
        p <<= 1
    return p


def _alloc_dense(t):
    """Dense output, matching what torch advanced indexing returns."""
    return _empty_like(t, memory_format=torch.contiguous_format)


# key: (N, H, M, K, SI0, SI1, SO0, SO1, inp dtype, probs dtype, device,
#       flattened-view flag, split dtype, sorted dtype)
_FAST = {}
_FAST_GET = _FAST.get
# probs=None must return an empty float32 tensor; it carries no data, so one
# zero-element tensor per device is reused instead of allocating every call.
_EMPTY_PROBS = {}


def _plan(N, H, M, K, si0, si1, so0, so1):
    """Launch geometry and constexpr values in declaration order."""
    bh = _next_pow2(H if H > 0 else 1)
    if bh > 1024:
        bh = 1024
    if bh < 8:
        bh = 8
    bn = 2048 // bh
    if bn < 1:
        bn = 1
    elif bn > 128:
        bn = 128
    bm = _next_pow2(M if M > 0 else 1)
    warps = (bh * bn) // 256
    if warps < 1:
        warps = 1
    elif warps > 8:
        warps = 8
    grid = (max(1, -(-H // bh)), max(1, -(-N // bn)), 1)
    if K > 0:
        consts = dict(
            N=N,
            M=M,
            K=K,
            SI0=si0,
            SI1=si1,
            SO0=so0,
            SO1=so1,
            H=H,
            BLOCK_H=bh,
            BLOCK_N=bn,
            BLOCK_M=bm,
            BLOCK_K=32,
            num_warps=warps,
        )
        cargs = (N, M, K, si0, si1, so0, so1, H, bh, bn, bm, 32)
    else:
        consts = dict(
            N=N,
            M=M,
            SI0=si0,
            SI1=si1,
            SO0=so0,
            SO1=so1,
            H=H,
            BLOCK_H=bh,
            BLOCK_N=bn,
            BLOCK_M=bm,
            num_warps=warps,
        )
        cargs = (N, M, si0, si1, so0, so1, H, bh, bn, bm)
    return grid, consts, cargs


def _bind(key, first_args):
    """First call for a metadata key: JIT launch, then cache the call closure.

    The returned closure only allocates the outputs and launches the compiled
    kernel, so the steady-state path has no per-call branching or re-derivation.
    """
    N, H, M, K, si0, si1, so0, so1 = key[:8]
    dev = key[10]
    flat = key[11]
    alloc = _empty_like if (si0 == H and si1 == 1) else _alloc_dense
    grid, consts, cargs = _plan(N, H, M, K, si0, si1, so0, so1)
    if K > 0:
        compiled = _chunk_sort_probs_kernel[grid](*first_args, **consts)
    else:
        compiled = _chunk_sort_kernel[grid](*first_args, **consts)

    try:
        launcher = compiled.run
        # The compiled launcher parses 13 fixed arguments followed by every
        # declared parameter (constexpr slots included).
        raw = launcher.launch
        fn = compiled.function
        coop = int(launcher.launch_cooperative_grid)
        pdl = int(launcher.launch_pdl)
        packed = compiled.packed_metadata
        g0, g1, g2 = grid[0], grid[1], grid[2]
        if _RAW_STREAM is not None:
            dev_idx = dev.index or 0

            def _stream():
                return _RAW_STREAM(dev_idx)

        else:  # pragma: no cover - other torch builds

            def _stream():
                return torch.cuda.current_stream().cuda_stream

        if K > 0:

            def _call(inp, probs, split_sizes, sorted_idxs):
                out = alloc(inp)
                permuted_probs = _empty_like(probs)
                row_id_map = _empty(N, dtype=_int32, device=dev)
                if flat:
                    src = inp.view(N, -1)
                    dst = out.view(N, -1)
                else:
                    src = inp
                    dst = out
                raw(
                    g0,
                    g1,
                    g2,
                    _stream(),
                    fn,
                    coop,
                    pdl,
                    None,
                    None,
                    packed,
                    None,
                    None,
                    None,
                    src,
                    dst,
                    row_id_map,
                    probs,
                    permuted_probs,
                    split_sizes,
                    sorted_idxs,
                    *cargs,
                )
                return out, permuted_probs, row_id_map

        else:
            empty = _EMPTY_PROBS.get(dev)
            if empty is None:
                empty = _empty(0, dtype=torch.float32, device=dev)
                _EMPTY_PROBS[dev] = empty

            def _call(inp, probs, split_sizes, sorted_idxs):
                out = alloc(inp)
                row_id_map = _empty(N, dtype=_int32, device=dev)
                if flat:
                    src = inp.view(N, -1)
                    dst = out.view(N, -1)
                else:
                    src = inp
                    dst = out
                raw(
                    g0,
                    g1,
                    g2,
                    _stream(),
                    fn,
                    coop,
                    pdl,
                    None,
                    None,
                    packed,
                    None,
                    None,
                    None,
                    src,
                    dst,
                    row_id_map,
                    split_sizes,
                    sorted_idxs,
                    *cargs,
                )
                return out, empty, row_id_map

        return _call
    except Exception:  # pragma: no cover - direct launcher unavailable
        if K > 0:

            def _call(inp, probs, split_sizes, sorted_idxs):
                out = alloc(inp)
                permuted_probs = torch.empty_like(probs)
                row_id_map = torch.empty(N, dtype=torch.int32, device=dev)
                src = inp.view(N, -1) if flat else inp
                dst = out.view(N, -1) if flat else out
                _chunk_sort_probs_kernel[grid](
                    src,
                    dst,
                    row_id_map,
                    probs,
                    permuted_probs,
                    split_sizes,
                    sorted_idxs,
                    **consts,
                )
                return out, permuted_probs, row_id_map

        else:

            def _call(inp, probs, split_sizes, sorted_idxs):
                out = alloc(inp)
                row_id_map = torch.empty(N, dtype=torch.int32, device=dev)
                src = inp.view(N, -1) if flat else inp
                dst = out.view(N, -1) if flat else out
                _chunk_sort_kernel[grid](
                    src, dst, row_id_map, split_sizes, sorted_idxs, **consts
                )
                return out, _EMPTY_PROBS[dev], row_id_map

        return _call


def moe_chunk_sort(inp, split_sizes, sorted_idxs, probs=None):
    with torch.cuda.device(inp.device):
        return _run(
            inp,
            split_sizes.contiguous(),
            sorted_idxs.contiguous(),
            None if probs is None else probs.contiguous(),
        )


def _run(inp, split_sizes, sorted_idxs, probs):
    shape = inp.shape
    nd = len(shape)
    flat = nd > 2
    if nd == 2:
        N = shape[0]
        H = shape[1]
    elif nd == 1:
        N = shape[0]
        H = 1
    else:
        N = shape[0]
        H = 1
        for i in range(1, nd):
            H *= shape[i]

    if N == 0:
        out = torch.empty_like(inp)
        row_id_map = torch.empty(0, dtype=torch.int32, device=inp.device)
        if probs is None:
            permuted_probs = torch.empty(0, dtype=torch.float32, device=inp.device)
        else:
            permuted_probs = torch.empty_like(probs)
        return out, permuted_probs, row_id_map

    dev = inp.device
    K = 0 if probs is None else probs.numel() // N
    contig = inp.is_contiguous()
    if contig:
        si0, si1, so0, so1 = H, 1, H, 1
    else:
        si0, si1 = inp.stride(0), (inp.stride(1) if nd > 1 else 1)
        so0, so1 = H, 1
    key = (
        N,
        H,
        split_sizes.numel(),
        K,
        si0,
        si1,
        so0,
        so1,
        inp.dtype,
        inp.dtype if probs is None else probs.dtype,
        dev,
        flat,
        split_sizes.dtype,
        sorted_idxs.dtype,
    )
    call = _FAST_GET(key)
    if call is None:
        # First call for this key: allocate the outputs, run the ordinary JIT
        # launch, and cache the specialised closure for later calls.
        out = _empty_like(inp) if contig else _alloc_dense(inp)
        row_id_map = _empty(N, dtype=_int32, device=dev)
        src = inp.view(N, -1) if flat else inp
        dst = out.view(N, -1) if flat else out
        if probs is None:
            permuted_probs = _empty(0, dtype=torch.float32, device=dev)
            _EMPTY_PROBS[dev] = permuted_probs
            first_args = (src, dst, row_id_map, split_sizes, sorted_idxs)
        else:
            permuted_probs = _empty_like(probs)
            first_args = (
                src,
                dst,
                row_id_map,
                probs,
                permuted_probs,
                split_sizes,
                sorted_idxs,
            )
        _FAST[key] = _bind(key, first_args)
        return out, permuted_probs, row_id_map
    return call(inp, probs, split_sizes, sorted_idxs)
