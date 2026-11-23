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

"""

import torch
import triton
import triton.language as tl

from flag_train.runtime import torch_device_fn


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


def _plan(N, H, M, K, si0, si1, so0, so1):
    """Launch geometry and constexpr values in declaration order."""
    bh = triton.next_power_of_2(max(H, 1))
    if bh > 1024:
        bh = 1024
    if bh < 8:
        bh = 8
    bn = 2048 // bh
    if bn < 1:
        bn = 1
    elif bn > 128:
        bn = 128
    bm = triton.next_power_of_2(max(M, 1))
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
    return grid, consts


def moe_chunk_sort(inp, split_sizes, sorted_idxs, probs=None):
    if not inp.numel():
        reordered = (
            inp.new_empty(0, dtype=torch.float32) if probs is None else probs.clone()
        )
        return inp.clone(), reordered, inp.new_empty(0, dtype=torch.int32)

    rows = inp.shape[0]
    src = inp.reshape(rows, -1)
    out = torch.empty_like(inp, memory_format=torch.contiguous_format)
    dst = out.reshape(rows, -1)
    mapping = inp.new_empty(rows, dtype=torch.int32)
    splits = split_sizes.contiguous()
    order = sorted_idxs.contiguous()
    probabilities = None if probs is None else probs.contiguous()
    width = 0 if probabilities is None else probabilities.numel() // rows
    grid, params = _plan(
        rows, src.shape[1], splits.numel(), width, *src.stride(), *dst.stride()
    )
    with torch_device_fn.device(inp.device):
        if width:
            reordered = torch.empty_like(probabilities)
            _chunk_sort_probs_kernel[grid](
                src, dst, mapping, probabilities, reordered, splits, order, **params
            )
        else:
            reordered = (
                inp.new_empty(0, dtype=torch.float32)
                if probs is None
                else torch.empty_like(probs)
            )
            _chunk_sort_kernel[grid](src, dst, mapping, splits, order, **params)
    return out, reordered, mapping
