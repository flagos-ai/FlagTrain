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


"""Triton implementation of TransformerEngine's moe_permute_mask_map forward.

Single-launch architecture. Grid = NB token-block programs + (E + 1) range programs:
  * token-block program pid < NB : counting-sort map for its own tokens, writes
    row_id_map (count column + descending-destination slots), permuted_probs and
    gathers the activation rows it owns directly into `permuted`.
  * range program pid >= NB : zeroes one uncovered row range of `permuted` /
    `permuted_probs` (leading pad gap, inter-expert pad gaps, trailing tail).
Both program kinds derive the same expert destination ranges in-kernel, and their
stores are disjoint, so a single launch is race free and needs no scratch buffer.

The per-call host path is the bottleneck on this workload (the op moves <= 2 MB),
so the wrapper keeps only the three mandatory output allocations and one launch,
with the launch surface kept small (no NB/ROWSTRIDE parameters, positional
constexprs, inherited-dtype allocations).
"""

import torch
import triton
import triton.language as tl

from flag_train.runtime import torch_device_fn

__all__ = ["moe_permute_mask_map"]


@triton.jit
def _moe_permute_fused(
    routing_map_ptr,  # int32 [T, E]  (active iff != 0)
    probs_ptr,  # [T, E] or dummy
    pad_ptr,  # int32 [E] or dummy
    inp_ptr,  # [T, H]
    row_id_map_ptr,  # int32 [T, 2E+1]
    perm_ptr,  # [N, H]
    pprobs_ptr,  # [N] or dummy
    T,
    N,
    H,
    E: tl.constexpr,
    EP: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_Z: tl.constexpr,
    HAS_PAD: tl.constexpr,
    HAS_PROBS: tl.constexpr,
):
    pid = tl.program_id(0)
    nb = tl.cdiv(T, BLOCK_T)
    er = tl.arange(0, EP)
    em = er < E

    # ---- global per-expert counts, plus this token block's exclusive base ----
    total = tl.zeros([EP], dtype=tl.int32)
    base = tl.zeros([EP], dtype=tl.int32)
    for b in range(0, nb):
        tb = b * BLOCK_T + tl.arange(0, BLOCK_T)
        mm = (tb[:, None] < T) & em[None, :]
        a = tl.load(routing_map_ptr + tb[:, None] * E + er[None, :], mask=mm, other=0)
        cs = tl.sum((a != 0).to(tl.int32), axis=0)
        total += cs
        base += tl.where(b < pid, cs, 0)
    off = tl.cumsum(total, axis=0) - total

    if HAS_PAD:
        padv = tl.load(pad_ptr + er, mask=em, other=0).to(tl.int32)
    else:
        padv = tl.zeros([EP], dtype=tl.int32)

    if pid < nb:
        # ================= map + gather for this token block =================
        t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
        tm = t < T
        mm = tm[:, None] & em[None, :]
        a = tl.load(routing_map_ptr + t[:, None] * E + er[None, :], mask=mm, other=0)
        act = (a != 0).to(tl.int32)
        rank = tl.cumsum(act, axis=0) - act + base[None, :]
        d = off[None, :] + rank
        valid = (act != 0) & (d < N)
        vi = valid.to(tl.int32)
        kept = tl.sum(vi, axis=1)
        slot = kept[:, None] - tl.cumsum(vi, axis=1)

        rowp = row_id_map_ptr + t[:, None] * (2 * E + 1)
        tl.store(row_id_map_ptr + t * (2 * E + 1) + (2 * E), kept, mask=tm)
        tl.store(rowp + slot, d, mask=valid)
        tl.store(rowp + E + slot, er[None, :], mask=valid)

        dd = d + padv[None, :]
        wv = valid & (dd < N)
        wvi = wv.to(tl.int32)
        if HAS_PROBS:
            pv = tl.load(probs_ptr + t[:, None] * E + er[None, :], mask=mm, other=0)

        t64 = t[:, None].to(tl.int64)
        for hb in range(0, H, BLOCK_H):
            ch = hb + tl.arange(0, BLOCK_H)
            cm = ch < H
            v = tl.load(
                inp_ptr + t64 * H + ch[None, :],
                mask=tm[:, None] & cm[None, :],
                other=0,
            )
            for e in tl.static_range(E):
                sel = er[None, :] == e
                dde = tl.sum(tl.where(sel, dd, 0), axis=1)
                we = tl.sum(tl.where(sel, wvi, 0), axis=1) > 0
                if HAS_PROBS:
                    pve = tl.sum(tl.where(sel, pv, 0), axis=1)
                    tl.store(pprobs_ptr + dde, pve, mask=we)
                    val = tl.where(pve[:, None] != 0, v, 0)
                else:
                    val = v
                tl.store(
                    perm_ptr + dde[:, None].to(tl.int64) * H + ch[None, :],
                    val,
                    mask=we[:, None] & cm[None, :],
                )
    else:
        # ================= zero one uncovered destination row range ==========
        start = off + padv
        keptv = tl.minimum(tl.maximum(tl.minimum(total, N - off), 0), total)
        end = tl.minimum(start + keptv, N)
        j = pid - nb
        prev = tl.sum(tl.where(er == (j - 1), end, 0), axis=0)
        lo = tl.where(j == 0, 0, prev)
        cur = tl.sum(tl.where(er == j, start, 0), axis=0)
        hi = tl.minimum(tl.where(j == E, N, cur), N)
        for r0 in range(lo, hi, BLOCK_Z):
            r = r0 + tl.arange(0, BLOCK_Z)
            rm = (r < hi) & (r < N)
            r64 = r[:, None].to(tl.int64)
            for hb in range(0, H, BLOCK_H):
                ch = hb + tl.arange(0, BLOCK_H)
                cm = ch < H
                tl.store(
                    perm_ptr + r64 * H + ch[None, :],
                    tl.zeros([BLOCK_Z, BLOCK_H], dtype=perm_ptr.dtype.element_ty),
                    mask=rm[:, None] & cm[None, :],
                )
            if HAS_PROBS:
                tl.store(
                    pprobs_ptr + r,
                    tl.zeros([BLOCK_Z], dtype=pprobs_ptr.dtype.element_ty),
                    mask=rm,
                )


def moe_permute_mask_map(
    inp, routing_map, num_out_tokens, probs=None, pad_offsets=None
):
    with torch_device_fn.device(inp.device):
        return _run_contiguous(
            inp.contiguous(),
            routing_map.contiguous(),
            num_out_tokens,
            None if probs is None else probs.contiguous(),
            None if pad_offsets is None else pad_offsets.contiguous(),
        )


def _run_contiguous(inp, routing_map, num_out_tokens, probs, pad_offsets):
    T = inp.shape[0]
    H = inp.shape[1]
    E = routing_map.shape[1]
    N = int(num_out_tokens)
    has_probs = probs is not None

    if inp.numel() == 0:
        return (
            inp.clone(),
            inp.new_empty((0, 2 * E + 1), dtype=torch.int32),
            inp.new_empty(0, dtype=probs.dtype if has_probs else torch.float32),
        )

    row_width = 2 * E + 1
    permuted = inp.new_empty((N, H))
    row_id_map = inp.new_empty((T, row_width), dtype=torch.int32)
    if has_probs:
        permuted_probs = probs.new_empty((N,))
    else:
        permuted_probs = inp.new_empty(0, dtype=torch.float32)

    _moe_permute_fused[((T + 31) // 32 + E + 1,)](
        routing_map,
        probs if has_probs else inp,
        pad_offsets if pad_offsets is not None else routing_map,
        inp,
        row_id_map,
        permuted,
        permuted_probs if has_probs else inp,
        T,
        N,
        H,
        E,
        1 << (E - 1).bit_length(),
        32,
        1 << min((H - 1).bit_length(), 7),
        32,
        pad_offsets is not None,
        has_probs,
    )
    return permuted, row_id_map, permuted_probs
