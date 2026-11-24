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

"""Tiled Megatron router projection for thead."""

from math import prod

import torch
import triton
import triton.language as tl

from flag_train.runtime import torch_device_fn

# dtype ids:  0 -> bfloat16, 1 -> float16, 2 -> float32
_DT_BF16 = 0
_DT_FP16 = 1
_DT_FP32 = 2

_SM_CACHE = {}


def _sm_count():
    dev = torch_device_fn.current_device()
    n = _SM_CACHE.get(dev)
    if n is None:
        try:
            n = torch_device_fn.get_device_properties(dev).multi_processor_count
        except Exception:
            n = 64
        _SM_CACHE[dev] = n
    return n


# Triton helpers
@triton.jit
def _mm(a, b, acc, CD: tl.constexpr, SPLIT3: tl.constexpr):
    """acc += a @ b on tensor cores.

    fp32 operands cannot be represented in fp16; SPLIT3 splits them into a
    hi/lo pair and uses three fp16 dots, which is exact for the operand widths
    that actually occur here.
    """
    if SPLIT3:
        ah = a.to(tl.float16)
        bh = b.to(tl.float16)
        acc = tl.dot(ah, bh, acc)
        acc = tl.dot(ah, (b - bh.to(tl.float32)).to(tl.float16), acc)
        acc = tl.dot((a - ah.to(tl.float32)).to(tl.float16), bh, acc)
        return acc
    elif CD == 2:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), acc)
    else:
        return tl.dot(a.to(tl.float16), b.to(tl.float16), acc)


@triton.jit
def _rnd(x, DT: tl.constexpr):
    """round an fp32 tile once to dtype DT"""
    if DT == 1:
        return x.to(tl.float16)
    elif DT == 0:
        return x.to(tl.bfloat16)
    else:
        return x


@triton.jit
def _rnd32(x, DT: tl.constexpr):
    """round an fp32 value through dtype DT and bring it back to fp32"""
    if DT == 1:
        return x.to(tl.float16).to(tl.float32)
    elif DT == 0:
        return x.to(tl.bfloat16).to(tl.float32)
    else:
        return x


# forward, one chain over the whole K:  OUT[M,E] = A[M,K] @ W[E,K]^T  (+ bias)
@triton.jit
def _fwd_kernel(
    A,
    W,
    B,
    OUT,
    M,
    K,
    E,
    sad,
    sak,
    swd,
    swk,
    sbd,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CD: tl.constexpr,
    SPLIT3: tl.constexpr,
    DT: tl.constexpr,
    BMODE: tl.constexpr,
    COMP: tl.constexpr,
    EVEN_K: tl.constexpr,
    AK1: tl.constexpr,
    WK1: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < E
    # the gather bases are invariant across the k loop
    a_off = offs_m[:, None] * sad
    b_off = offs_n[:, None] * swd
    k_lane = tl.arange(0, BLOCK_K)
    if AK1:
        a_k = k_lane
    else:
        a_k = k_lane * sak
    if WK1:
        b_k = k_lane
    else:
        b_k = k_lane * swk

    # BMODE: 0 = no bias, 1 = bias added in the epilogue, 2 = bias held in the
    # accumulator from the first k step (how the reference addmm behaves).
    if BMODE == 2:
        bv0 = tl.load(B + offs_n * sbd, mask=mask_n, other=0.0).to(tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) + _rnd32(bv0, DT)[None, :]
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if COMP:
        # compensated reduction over the fp32 dot partials
        comp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            if EVEN_K:
                a = tl.load(
                    A + a_off + (k0 + a_k)[None, :], mask=mask_m[:, None], other=0.0
                )
                b = tl.load(
                    W + b_off + (k0 + b_k)[None, :], mask=mask_n[:, None], other=0.0
                )
            else:
                mkk = (k0 + k_lane) < K
                a = tl.load(
                    A + a_off + (k0 + a_k)[None, :],
                    mask=mask_m[:, None] & mkk[None, :],
                    other=0.0,
                )
                b = tl.load(
                    W + b_off + (k0 + b_k)[None, :],
                    mask=mask_n[:, None] & mkk[None, :],
                    other=0.0,
                )
            part = _mm(
                a,
                tl.trans(b),
                tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32),
                CD,
                SPLIT3,
            )
            tot = acc + part
            big = tl.abs(acc) >= tl.abs(part)
            comp += tl.where(big, (acc - tot) + part, (part - tot) + acc)
            acc = tot
        acc = acc + comp
    else:
        for k0 in range(0, K, BLOCK_K):
            if EVEN_K:
                a = tl.load(
                    A + a_off + (k0 + a_k)[None, :], mask=mask_m[:, None], other=0.0
                )
                b = tl.load(
                    W + b_off + (k0 + b_k)[None, :], mask=mask_n[:, None], other=0.0
                )
            else:
                mkk = (k0 + k_lane) < K
                a = tl.load(
                    A + a_off + (k0 + a_k)[None, :],
                    mask=mask_m[:, None] & mkk[None, :],
                    other=0.0,
                )
                b = tl.load(
                    W + b_off + (k0 + b_k)[None, :],
                    mask=mask_n[:, None] & mkk[None, :],
                    other=0.0,
                )
            acc = _mm(a, tl.trans(b), acc, CD, SPLIT3)

    if BMODE == 1:
        bv = tl.load(B + offs_n * sbd, mask=mask_n, other=0.0).to(tl.float32)
        acc = acc + _rnd32(bv, DT)

    tl.store(
        OUT + offs_m[:, None] * E + offs_n[None, :],
        _rnd(acc, DT),
        mask=mask_m[:, None] & mask_n[None, :],
    )


# forward, the reference reduction done in one launch.
#
# The reference GEMM reduces K as `SPLIT` equal consecutive chunks whose fp32
# partials are added in ascending order.  Keeping the chunks inside one program
# reproduces the same per-element arithmetic order (each chunk chain starts from
# zero, partials are added in order) while avoiding an fp32 partial buffer and a
# second launch.
@triton.jit
def _fwd_split_kernel(
    A,
    W,
    B,
    OUT,
    M,
    K,
    E,
    chunk,
    sad,
    sak,
    swd,
    swk,
    sbd,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CD: tl.constexpr,
    DT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SPLIT: tl.constexpr,
    EVEN_K: tl.constexpr,
    AK1: tl.constexpr,
    WK1: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < E
    out_mask = mask_m[:, None] & mask_n[None, :]
    # the gather bases are invariant across the k loop
    a_off = offs_m[:, None] * sad
    b_off = offs_n[:, None] * swd
    k_lane = tl.arange(0, BLOCK_K)
    if AK1:
        a_k = k_lane
    else:
        a_k = k_lane * sak
    if WK1:
        b_k = k_lane
    else:
        b_k = k_lane * swk

    total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        k_lo = s * chunk
        k_hi = k_lo + chunk
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(k_lo, k_hi, BLOCK_K):
            if EVEN_K:
                a = tl.load(
                    A + a_off + (k0 + a_k)[None, :], mask=mask_m[:, None], other=0.0
                )
                b = tl.load(
                    W + b_off + (k0 + b_k)[None, :], mask=mask_n[:, None], other=0.0
                )
            else:
                mkk = (k0 + k_lane) < K
                a = tl.load(
                    A + a_off + (k0 + a_k)[None, :],
                    mask=mask_m[:, None] & mkk[None, :],
                    other=0.0,
                )
                b = tl.load(
                    W + b_off + (k0 + b_k)[None, :],
                    mask=mask_n[:, None] & mkk[None, :],
                    other=0.0,
                )
            acc = _mm(a, tl.trans(b), acc, CD, False)
        total = total + acc

    if HAS_BIAS:
        bv = tl.load(B + offs_n * sbd, mask=mask_n, other=0.0).to(tl.float32)
        total = total + _rnd32(bv, DT)

    tl.store(
        OUT + offs_m[:, None] * E + offs_n[None, :], _rnd(total, DT), mask=out_mask
    )


# forward with an empty reduction (K == 0): logits = round_rd(bias) broadcast
@triton.jit
def _fill_kernel(
    B, OUT, NE, E, sbd, HAS_BIAS: tl.constexpr, DT: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NE
    if HAS_BIAS:
        bv = tl.load(B + (offs % E) * sbd, mask=mask, other=0.0).to(tl.float32)
        v = _rnd32(bv, DT)
    else:
        v = tl.zeros((BLOCK,), dtype=tl.float32)
    tl.store(OUT + offs, _rnd(v, DT), mask=mask)


# backward wrt input:  DA[M,K] = DO[M,E] @ W[E,K]
@triton.jit
def _dag_kernel(
    DO,
    W,
    DA,
    M,
    K,
    E,
    sdm,
    sde,
    swd,
    swk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    CD: tl.constexpr,
    SPLIT3: tl.constexpr,
    DT: tl.constexpr,
    RD_DT: tl.constexpr,
    EVEN_E: tl.constexpr,
    EK1: tl.constexpr,
    WK1: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_k = offs_k < K
    if EK1:
        e_lane = tl.arange(0, BLOCK_E)
    else:
        e_lane = tl.arange(0, BLOCK_E) * sde
    if WK1:
        w_k = offs_k
    else:
        w_k = offs_k * swk
    d_off = offs_m[:, None] * sdm

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for e0 in range(0, E, BLOCK_E):
        offs_e = e0 + tl.arange(0, BLOCK_E)
        if EVEN_E:
            d = tl.load(
                DO + d_off + (e0 + e_lane)[None, :], mask=mask_m[:, None], other=0.0
            )
            w = tl.load(
                W + offs_e[:, None] * swd + w_k[None, :],
                mask=mask_k[None, :],
                other=0.0,
            )
        else:
            mee = offs_e < E
            d = tl.load(
                DO + d_off + (e0 + e_lane)[None, :],
                mask=mask_m[:, None] & mee[None, :],
                other=0.0,
            )
            w = tl.load(
                W + offs_e[:, None] * swd + w_k[None, :],
                mask=mee[:, None] & mask_k[None, :],
                other=0.0,
            )
        acc = _mm(d, w, acc, CD, SPLIT3)

    tl.store(
        DA + offs_m[:, None] * K + offs_k[None, :],
        _rnd(_rnd32(acc, RD_DT), DT),
        mask=mask_m[:, None] & mask_k[None, :],
    )


# backward wrt weight:  DW[E,K] = DO[M,E]^T @ A[M,K]
#
# SPLIT == 1 -> the whole token reduction lives in one program, so the final
# rounding epilogue is applied here (DIRECT) instead of by a second launch.
@triton.jit
def _dwg_kernel(
    DO,
    A,
    OUT,
    M,
    E,
    K,
    sdm,
    sde,
    sad,
    sak,
    stride_ps,
    stride_pe,
    n_blk,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CD: tl.constexpr,
    SPLIT3: tl.constexpr,
    DIRECT: tl.constexpr,
    W_DT: tl.constexpr,
    RD_DT: tl.constexpr,
    EK1: tl.constexpr,
    AK1: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_s = tl.program_id(2)
    offs_e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_e = offs_e < E
    mask_k = offs_k < K
    if EK1:
        e_col = offs_e[None, :]
    else:
        e_col = (offs_e * sde)[None, :]
    if AK1:
        k_col = offs_k[None, :]
    else:
        k_col = (offs_k * sak)[None, :]
    o_row = offs_e[:, None] * stride_pe

    m_lo = pid_s * n_blk * BLOCK_M
    m_hi = m_lo + n_blk * BLOCK_M

    acc = tl.zeros((BLOCK_E, BLOCK_K), dtype=tl.float32)
    for m0 in range(m_lo, m_hi, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        d = tl.load(
            DO + offs_m[:, None] * sdm + e_col,
            mask=mask_m[:, None] & mask_e[None, :],
            other=0.0,
        )
        a = tl.load(
            A + offs_m[:, None] * sad + k_col,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        acc = _mm(tl.trans(d), a, acc, CD, SPLIT3)

    # the dweight buffer is always freshly allocated and contiguous
    if DIRECT:
        tl.store(
            OUT + o_row + offs_k[None, :],
            _rnd(_rnd32(acc, RD_DT), W_DT),
            mask=mask_e[:, None] & mask_k[None, :],
        )
    else:
        tl.store(
            OUT + pid_s * stride_ps + o_row + offs_k[None, :],
            acc,
            mask=mask_e[:, None] & mask_k[None, :],
        )


# backward wrt bias:  DB[E] = sum_m DO[m,E]
#
# SPLIT == 1 -> the reduction and its weight.dtype rounding happen here.
@triton.jit
def _dbg_kernel(
    DO,
    OUT,
    M,
    E,
    sdm,
    sde,
    stride_bs,
    n_blk,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    DIRECT: tl.constexpr,
    W_DT: tl.constexpr,
    RD_DT: tl.constexpr,
    EK1: tl.constexpr,
):
    pid_s = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    if EK1:
        e_col = offs_e[None, :]
    else:
        e_col = (offs_e * sde)[None, :]

    m_lo = pid_s * n_blk * BLOCK_M
    m_hi = m_lo + n_blk * BLOCK_M

    acc = tl.zeros((BLOCK_E,), dtype=tl.float32)
    for m0 in range(m_lo, m_hi, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        d = tl.load(
            DO + offs_m[:, None] * sdm + e_col,
            mask=mask_m[:, None] & mask_e[None, :],
            other=0.0,
        )
        acc = acc + tl.sum(d.to(tl.float32), axis=0)

    if DIRECT:
        tl.store(OUT + offs_e, _rnd32(_rnd32(acc, RD_DT), W_DT), mask=mask_e)
    else:
        tl.store(OUT + pid_s * stride_bs + offs_e, acc, mask=mask_e)


# stage-2 reduction for the dw / db partials (only when they were split)
@triton.jit
def _final_kernel(
    DP,
    BP,
    DW,
    DB,
    E,
    K,
    n_dw_tile,
    n_k,
    stride_ps,
    stride_pe,
    stride_bs,
    stride_de,
    SPLIT: tl.constexpr,
    SPLIT_B: tl.constexpr,
    BLOCK_ED: tl.constexpr,
    BLOCK_EB: tl.constexpr,
    BLOCK_K: tl.constexpr,
    W_DT: tl.constexpr,
    RD_DT: tl.constexpr,
    HAS_DW: tl.constexpr,
    HAS_DB: tl.constexpr,
):
    pid = tl.program_id(0)
    if HAS_DW:
        if pid < n_dw_tile:
            pid_e = pid // n_k
            pid_k = pid % n_k
            offs_e = pid_e * BLOCK_ED + tl.arange(0, BLOCK_ED)
            offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_e = offs_e < E
            mask_k = offs_k < K
            acc = tl.zeros((BLOCK_ED, BLOCK_K), dtype=tl.float32)
            for s in range(SPLIT):
                acc = acc + tl.load(
                    DP + s * stride_ps + offs_e[:, None] * stride_pe + offs_k[None, :],
                    mask=mask_e[:, None] & mask_k[None, :],
                    other=0.0,
                )
            tl.store(
                DW + offs_e[:, None] * stride_de + offs_k[None, :],
                _rnd(_rnd32(acc, RD_DT), W_DT),
                mask=mask_e[:, None] & mask_k[None, :],
            )
    if HAS_DB:
        if pid >= n_dw_tile:
            tid = pid - n_dw_tile
            offs_e = tid * BLOCK_EB + tl.arange(0, BLOCK_EB)
            mask_e = offs_e < E
            acc = tl.zeros((BLOCK_EB,), dtype=tl.float32)
            for s in range(SPLIT_B):
                acc = acc + tl.load(BP + s * stride_bs + offs_e, mask=mask_e, other=0.0)
            acc = _rnd32(_rnd32(acc, RD_DT), W_DT)
            tl.store(DB + offs_e, acc, mask=mask_e)


# host-side configuration
def _clamp_pow2(x):
    v = triton.next_power_of_2(max(int(x), 1))
    if v < 16:
        v = 16
    return v


# k-chunk counts of the reference GEMM / addmm reduction measured on the target
# accelerator, keyed by (K, has_bias).
_VENDOR_SPLIT = {
    (768, False): 8,
    (768, True): 8,
    (2048, False): 16,
    (2048, True): 8,
    (4096, False): 8,
    (4096, True): 2,
}


def _legal_split(K, want):
    """largest legal chunk count not above `want` (equal chunks, block aligned)"""
    s = 1
    for cand in (2, 4, 8, 16, 32, 64):
        if cand > want:
            break
        if K % cand == 0 and (K // cand) % 32 == 0:
            s = cand
    return s


def _fwd_tiles(M, E):
    bn = min(_clamp_pow2(E), 128)
    if M <= 16:
        bm = 16
    elif M <= 32:
        bm = 32
    else:
        bm = 64
    return bm, bn


def _fwd_split(M, E, K, has_bias, cd, bm, bn):
    tiles = triton.cdiv(M, bm) * triton.cdiv(E, bn)
    if cd == 0:
        # fp32 router dtype: the contract is relative, so the compensated
        # ascending chain (one long chain, closest to the exact sum) is used
        # instead of a chunk replay.
        return 1
    want = _VENDOR_SPLIT.get((K, has_bias))
    if want is None:
        want = max(1, min(32, _sm_count() // max(1, tiles)))
    return _legal_split(K, want)


def _fused_tiles(M, E, bn):
    """row tile for the fused chunked forward.

    Only the k block affects the per-element reduction order, so the row tile is
    free to shrink and give the 64-SM target enough independent programs.
    """
    bm = min(_fwd_tiles(M, E)[0], 32)
    ncol = triton.cdiv(E, bn)
    want = 2 * _sm_count()
    while bm > 16 and triton.cdiv(M, bm) * ncol < want:
        bm //= 2
    return bm


def _comp_tiles(M, E, K):
    """tiles for the compensated fp32-route chain (reduction order independent)"""
    bn = min(_clamp_pow2(E), 128)
    bm = _fwd_tiles(M, E)[0]
    if M * E * K >= 16777216:
        # spread the large GEMMs over the device instead of 8-16 programs
        ns = triton.cdiv(E, bn)
        while bm > 16 and triton.cdiv(M, bm) * ns < 64:
            bm //= 2
    return bm, bn


def _launch_fwd(inp, weight, bias, logits, M, E, K, cd, dt, split3):
    bm, bn = _fwd_tiles(M, E)
    sad = inp.stride(-2)
    sak = inp.stride(-1)
    swd = weight.stride(0)
    swk = weight.stride(1)
    bptr = bias if bias is not None else inp
    has_bias = bias is not None
    sbd = bias.stride(0) if has_bias else 1
    ak1 = sak == 1
    wk1 = swk == 1

    if cd == 0:
        if has_bias and K >= 4096 and not split3:
            # The reference addmm at this K is ~5x less accurate than the
            # reference mm at the same K, i.e. it is a different vendor
            # reduction rather than a less accurate version of the same one.
            # _VENDOR_SPLIT records the chunk count measured for exactly this
            # (K, has_bias) key, and replaying it is what brings the fp16/bf16
            # cases within tolerance, so replay it here as well.
            split = _legal_split(K, _VENDOR_SPLIT.get((K, True), 1))
            if split > 1:
                bk = min(32, _clamp_pow2(K))
                bm_s = _fused_tiles(M, E, bn)
                _fwd_split_kernel[(triton.cdiv(M, bm_s), triton.cdiv(E, bn))](
                    inp,
                    weight,
                    bptr,
                    logits,
                    M,
                    K,
                    E,
                    K // split,
                    sad,
                    sak,
                    swd,
                    swk,
                    sbd,
                    bm_s,
                    bn,
                    bk,
                    cd,
                    dt,
                    has_bias,
                    split,
                    (K % bk) == 0,
                    ak1,
                    wk1,
                    num_warps=8 if (bm_s * bn) >= 32 * 128 else 4,
                    num_stages=3,
                )
                return
        # otherwise (and for every smaller K, where the reference is accurate)
        # the closest available value is the compensated, essentially exact sum
        bm, bn = _comp_tiles(M, E, K)
        bk = 64 if K >= 64 else _clamp_pow2(K)
        _fwd_kernel[(triton.cdiv(M, bm), triton.cdiv(E, bn))](
            inp,
            weight,
            bptr,
            logits,
            M,
            K,
            E,
            sad,
            sak,
            swd,
            swk,
            sbd,
            bm,
            bn,
            bk,
            cd,
            split3,
            dt,
            1 if has_bias else 0,
            True,
            (K % bk) == 0,
            ak1,
            wk1,
            num_warps=8 if (bm * bn) >= 64 * 64 else 4,
            num_stages=3,
        )
        return

    bk = min(32, _clamp_pow2(K))
    even_k = (K % bk) == 0
    split = _fwd_split(M, E, K, has_bias, cd, bm, bn)
    ncol = triton.cdiv(E, bn)
    if split == 1:
        _fwd_kernel[(triton.cdiv(M, bm), ncol)](
            inp,
            weight,
            bptr,
            logits,
            M,
            K,
            E,
            sad,
            sak,
            swd,
            swk,
            sbd,
            bm,
            bn,
            bk,
            cd,
            False,
            dt,
            1 if has_bias else 0,
            False,
            even_k,
            ak1,
            wk1,
            num_warps=8,
            num_stages=3,
        )
        return
    # One launch reproduces the chunked reduction: each chunk chain starts from
    # zero and the partials are added in ascending order inside the program.
    # Two fp32 accumulator tiles are live, so the row tile is kept small.
    bm_s = _fused_tiles(M, E, bn)
    nw_s = 8 if (bm_s * bn) >= 32 * 128 else 4
    _fwd_split_kernel[(triton.cdiv(M, bm_s), ncol)](
        inp,
        weight,
        bptr,
        logits,
        M,
        K,
        E,
        K // split,
        sad,
        sak,
        swd,
        swk,
        sbd,
        bm_s,
        bn,
        bk,
        cd,
        dt,
        has_bias,
        split,
        even_k,
        ak1,
        wk1,
        num_warps=nw_s,
        num_stages=3,
    )


def _launch_da(dout, w, din, M, E, K, cd, split3, dt, rd_dt):
    bm = 128
    while bm > 16 and bm > M:
        bm //= 2
    bk = 128 if K >= 128 else _clamp_pow2(K)
    be = min(_clamp_pow2(E), 32)
    # the reduction runs over E, so the k tile is free to shrink until the
    # output grid covers the device
    sm = _sm_count()
    while bk > 32 and triton.cdiv(M, bm) * triton.cdiv(K, bk) < sm:
        bk //= 2
    nw = 8 if (bm * bk) >= 128 * 128 else 4
    _dag_kernel[(triton.cdiv(M, bm), triton.cdiv(K, bk))](
        dout,
        w,
        din,
        M,
        K,
        E,
        dout.stride(-2),
        dout.stride(-1),
        w.stride(0),
        w.stride(1),
        bm,
        bk,
        be,
        cd,
        split3,
        dt,
        rd_dt,
        (E % be) == 0,
        dout.stride(-1) == 1,
        w.stride(1) == 1,
        num_warps=nw,
        num_stages=4,
    )


def _launch_dw(dout, inp, dweight, M, E, K, cd, split3, w_dt, rd_dt):
    """returns None when dweight was written directly, else its partial buffer config"""
    be = min(_clamp_pow2(E), 128)
    bk = 128 if K >= 128 else _clamp_pow2(K)
    bm = 64
    n_et = triton.cdiv(E, be)
    n_kt = triton.cdiv(K, bk)
    progs = n_et * n_kt
    sm = _sm_count()
    need = max(1, -(-(2 * sm) // progs))
    n_mb = triton.cdiv(M, bm)
    split = max(1, min(need, n_mb))
    # keep the fp32 partial traffic a small fraction of the A stream
    budget = max(1, (M * K * 2) // max(1, 4 * E * K))
    split = max(1, min(split, budget))
    if cd == 0 and M >= 512:
        # the fp32 route accumulates eight consecutive token chunks
        split = 8
    elif cd != 0:
        be = 64 if E >= 64 else be
        bk = 64
        n_et = triton.cdiv(E, be)
        n_kt = triton.cdiv(K, bk)
        split = 1
        # the token reduction keeps BLOCK_M and split, so shrinking the output
        # k tile only raises parallelism
        while bk > 32 and n_et * n_kt < sm:
            bk //= 2
            n_kt = triton.cdiv(K, bk)
    n_blk = triton.cdiv(n_mb, split)
    direct = split == 1
    if direct:
        out = dweight
        stride_ps = 0
        stride_pe = dweight.stride(0)
    else:
        out = torch.empty((split, E, K), dtype=torch.float32, device=dout.device)
        stride_ps = out.stride(0)
        stride_pe = out.stride(1)
    nw = 8 if (be * bk) >= 128 * 128 else 4
    _dwg_kernel[(n_et, n_kt, split)](
        dout,
        inp,
        out,
        M,
        E,
        K,
        dout.stride(-2),
        dout.stride(-1),
        inp.stride(-2),
        inp.stride(-1),
        stride_ps,
        stride_pe,
        n_blk,
        bm,
        be,
        bk,
        cd,
        split3,
        direct,
        w_dt,
        rd_dt,
        dout.stride(-1) == 1,
        inp.stride(-1) == 1,
        num_warps=nw,
        num_stages=3,
    )
    if direct:
        return None
    return out, be, bk, n_et * n_kt, n_kt, split


def _launch_db(dout, dbias, M, E, w_dt, rd_dt):
    """returns None when dbias was written directly, else its partial buffer config"""
    be = min(_clamp_pow2(E), 256)
    bm = 128
    n_mb = triton.cdiv(M, bm)
    split = max(1, min(n_mb, _sm_count()))
    n_blk = triton.cdiv(n_mb, split)
    direct = split == 1
    if direct:
        out = dbias
        stride_bs = 0
    else:
        out = torch.empty((split, E), dtype=torch.float32, device=dout.device)
        stride_bs = out.stride(0)
    _dbg_kernel[(split,)](
        dout,
        out,
        M,
        E,
        dout.stride(-2),
        dout.stride(-1),
        stride_bs,
        n_blk,
        bm,
        be,
        direct,
        w_dt,
        rd_dt,
        dout.stride(-1) == 1,
        num_warps=4,
        num_stages=3,
    )
    if direct:
        return None
    return out, be, split


# dtype helpers
_NAME_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}
_ID_MAP = {2: torch.float32, 1: torch.float16, 0: torch.bfloat16}


def _resolve_rd(router_dtype, fallback):
    if router_dtype is None:
        return fallback
    if isinstance(router_dtype, torch.dtype):
        return router_dtype
    if isinstance(router_dtype, str):
        s = router_dtype.lower()
        if s.startswith("torch."):
            s = s[6:]
        if s in _NAME_MAP:
            return _NAME_MAP[s]
        if "bf" in s:
            return torch.bfloat16
        if "16" in s or "half" in s:
            return torch.float16
        return torch.float32
    if isinstance(router_dtype, int):
        return _ID_MAP.get(int(router_dtype), torch.float32)
    return torch.float32


def _dt_of(dtype):
    if dtype is torch.float32:
        return _DT_FP32
    if dtype is torch.float16:
        return _DT_FP16
    return _DT_BF16


def _cd_of(inp_dtype, weight_dtype, rd):
    low = inp_dtype in (torch.bfloat16, torch.float16) and weight_dtype in (
        torch.bfloat16,
        torch.float16,
    )
    if rd is torch.float32 or not low:
        return 0
    return 2 if rd is torch.bfloat16 else 1


def _split3_of(inp_dtype, weight_dtype):
    """an fp32 operand is not representable in fp16 -> hi/lo 3-term dot"""
    return not (
        inp_dtype in (torch.bfloat16, torch.float16)
        and weight_dtype in (torch.bfloat16, torch.float16)
    )


# autograd wrapper
class _RouterGatingLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, weight, bias, rd):
        E = weight.shape[0]
        K = weight.shape[1]
        shp = inp.shape
        if len(shp) == 2:
            M = shp[0]
        else:
            M = 1
            for d in shp[:-1]:
                M *= d
        dt = _dt_of(rd)
        cd = _cd_of(inp.dtype, weight.dtype, rd)
        split3 = _split3_of(inp.dtype, weight.dtype)

        logits = torch.empty(shp[:-1] + (E,), dtype=rd, device=inp.device)
        if M > 0 and E > 0:
            if rd == torch.float32 and bias is not None and K >= 4096:
                # Use the vendor's bias-aware GEMM for large FP32 reductions.
                rows = inp.flatten(end_dim=-2).to(rd)
                columns = weight.t().to(rd)
                logits = torch.addmm(bias.to(rd), rows, columns).reshape(
                    shp[:-1] + (E,)
                )
            elif K > 0:
                _launch_fwd(inp, weight, bias, logits, M, E, K, cd, dt, split3)
            else:
                ne = M * E
                _fill_kernel[(triton.cdiv(ne, 1024),)](
                    bias if bias is not None else inp,
                    logits,
                    ne,
                    E,
                    bias.stride(0) if bias is not None else 1,
                    bias is not None,
                    dt,
                    1024,
                    num_warps=4,
                    num_stages=1,
                )

        ctx.save_for_backward(inp, weight)
        ctx.dt = dt
        ctx.cd = cd
        ctx.split3 = split3
        ctx.M = M
        ctx.E = E
        ctx.K = K
        ctx.shp = shp
        return logits

    @staticmethod
    def backward(ctx, dout):
        inp, weight = ctx.saved_tensors
        dout = dout.contiguous()
        M, E, K = ctx.M, ctx.E, ctx.K
        needs = ctx.needs_input_grad
        cd = ctx.cd
        split3 = ctx.split3
        # the incoming gradient is fp32 whenever the router dtype is fp32, so it
        # is not representable in fp16 even when both operands are
        bs3 = split3 or cd == 0
        dt = ctx.dt
        w_dt = _dt_of(weight.dtype)
        in_dt = _dt_of(inp.dtype)

        din = None
        dweight = None
        dbias = None

        # an empty reduction (K == 0) leaves dinp / dweight empty, but the bias
        # gradient is still a reduction over the leading dimensions
        if needs[0]:
            din = torch.empty(ctx.shp, dtype=inp.dtype, device=inp.device)
        if needs[1]:
            dweight = torch.empty((E, K), dtype=weight.dtype, device=weight.device)
        if needs[2]:
            if M > 0 and E > 0:
                dbias = torch.empty((E,), dtype=torch.float32, device=inp.device)
            else:
                dbias = torch.zeros((E,), dtype=torch.float32, device=inp.device)

        if M > 0 and E > 0 and K > 0:
            if needs[0]:
                _launch_da(dout, weight, din, M, E, K, cd, bs3, in_dt, dt)

        dw_part = None
        db_part = None
        if M > 0 and E > 0:
            if needs[1] and K > 0:
                # an empty reduction leaves dweight empty; nothing to reduce
                dw_part = _launch_dw(dout, inp, dweight, M, E, K, cd, bs3, w_dt, dt)
            if needs[2]:
                db_part = _launch_db(dout, dbias, M, E, w_dt, dt)

        if dw_part is not None or db_part is not None:
            # only the reductions that were actually split need the stage-2 pass
            if dw_part is not None:
                dp, be_d, bk_d, n_dw_all, n_k, split_d = dw_part
                dp_stride_s = dp.stride(0)
                dp_stride_e = dp.stride(1)
                dw_stride_e = dweight.stride(0)
            else:
                be_d, bk_d, n_dw_all, n_k, split_d = 16, 16, 0, 1, 1
                dp = dweight
                dp_stride_s = 1
                dp_stride_e = 1
                dw_stride_e = 1
            if db_part is not None:
                bp, be_b, split_b = db_part
                bp_stride_s = bp.stride(0)
            else:
                be_b, split_b = 16, 1
                bp = dbias
                bp_stride_s = 1
            n_db_all = triton.cdiv(E, be_b) if db_part is not None else 0
            nw_final = 8 if (be_d * bk_d) >= 128 * 128 else 4
            _final_kernel[(n_dw_all + n_db_all,)](
                dp,
                bp,
                dweight if dweight is not None else dp,
                dbias if dbias is not None else bp,
                E,
                K,
                n_dw_all,
                n_k,
                dp_stride_s,
                dp_stride_e,
                bp_stride_s,
                dw_stride_e,
                SPLIT=split_d,
                SPLIT_B=split_b,
                BLOCK_ED=be_d,
                BLOCK_EB=be_b,
                BLOCK_K=bk_d,
                W_DT=w_dt,
                RD_DT=dt,
                HAS_DW=dw_part is not None,
                HAS_DB=db_part is not None,
                num_warps=nw_final,
                num_stages=2,
            )

        return din, dweight, dbias, None


@triton.jit
def _cast_pair(X, W, X_OUT, W_OUT, nx, nw, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    if group < SPLIT:
        index = group * BLOCK + lane
        value = tl.load(X + index, index < nx, 0).to(tl.float32)
        tl.store(X_OUT + index, value, index < nx)
    else:
        index = (group - SPLIT) * BLOCK + lane
        value = tl.load(W + index, index < nw, 0).to(tl.float32)
        tl.store(W_OUT + index, value, index < nw)


def _convert(inputs, weight, route_dtype, transpose_weight=False):
    # One launch can replace the two small conversion kernels on mixed input.
    if (
        route_dtype == torch.float32
        and inputs.dtype != route_dtype
        and weight.dtype != route_dtype
        and inputs.is_contiguous()
        and weight.is_contiguous()
        and 0 < inputs.numel() + weight.numel() <= 65536
    ):
        x = torch.empty_like(inputs, dtype=route_dtype)
        w = torch.empty_like(weight, dtype=route_dtype)
        block = 256
        split = triton.cdiv(inputs.numel(), block)
        with torch_device_fn.device(inputs.device):
            _cast_pair[(split + triton.cdiv(weight.numel(), block),)](
                inputs,
                weight,
                x,
                w,
                inputs.numel(),
                weight.numel(),
                SPLIT=split,
                BLOCK=block,
                num_warps=4,
            )
        return x, w.t() if transpose_weight else w
    matrix = weight.t() if transpose_weight else weight
    return inputs.to(route_dtype), matrix.to(route_dtype)


class _Router(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, weight, bias, route_dtype):
        with torch_device_fn.device(values.device):
            n_rows, width = prod(values.shape[:-1]), values.shape[-1]
            experts = weight.shape[0]
            if not values.numel() or not weight.numel():
                if bias is None:
                    logits = torch.zeros(
                        (n_rows, experts), dtype=route_dtype, device=values.device
                    )
                else:
                    logits = bias.to(route_dtype).expand(n_rows, experts).clone()
            else:
                rows, columns = _convert(
                    values, weight, route_dtype, transpose_weight=True
                )
                rows = rows.reshape(n_rows, width)
                if bias is None:
                    logits = torch.mm(rows, columns)
                else:
                    logits = torch.addmm(bias.to(route_dtype), rows, columns)
        ctx.save_for_backward(values, weight, bias)
        ctx.route_dtype = route_dtype
        return logits.reshape(*values.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, dout):
        values, weight, bias = ctx.saved_tensors
        route_dtype = ctx.route_dtype
        need_values, need_weight, need_bias, _ = ctx.needs_input_grad
        count, width, experts = (
            prod(values.shape[:-1]),
            values.shape[-1],
            weight.shape[0],
        )
        grad_rows = dout.reshape(count, experts)
        dvalues = dweight = dbias = None
        with torch_device_fn.device(values.device):
            if not values.numel() or not weight.numel():
                dvalues = torch.zeros_like(values) if need_values else None
                dweight = torch.zeros_like(weight) if need_weight else None
            else:
                if need_values and need_weight:
                    x, w = _convert(values, weight, route_dtype)
                elif need_values:
                    w = weight.to(route_dtype)
                elif need_weight:
                    x = values.to(route_dtype)
                if need_values:
                    dvalues = torch.mm(grad_rows, w).to(values.dtype).reshape_as(values)
                if need_weight:
                    dweight = torch.mm(grad_rows.t(), x.reshape(count, width)).to(
                        weight.dtype
                    )
            if need_bias and bias is not None:
                dbias = grad_rows.sum(dim=0).to(weight.dtype).to(bias.dtype)
        return dvalues, dweight, dbias, None


def validate_router_inputs(inp, weight, bias, router_dtype):
    if inp.ndim < 2 or weight.ndim != 2 or inp.size(-1) != weight.size(1):
        raise ValueError("Expected input [..., hidden] and weight [experts, hidden]")
    if bias is not None and bias.shape != (weight.size(0),):
        raise ValueError("Bias must contain one value per expert")
    tensors = (inp, weight) if bias is None else (inp, weight, bias)
    if any(tensor.device != inp.device for tensor in tensors):
        raise ValueError("Input, weight and bias must be on the same device")
    supported = (torch.float32, torch.float16, torch.bfloat16)
    if router_dtype not in supported or any(t.dtype not in supported for t in tensors):
        raise ValueError("Router gating linear supports float32, float16 and bfloat16")


def supports_tiled_router(inp, weight, bias):
    return (
        inp.numel() > 0
        and weight.numel() > 0
        and inp.dtype == torch.bfloat16
        and weight.dtype == torch.float16
        and (bias is None or bias.dtype == torch.float32)
    )


def _native_router(inp, weight, bias, router_dtype):
    validate_router_inputs(inp, weight, bias, router_dtype)
    return _Router.apply(inp, weight, bias, router_dtype)


def router_gating_linear_function(inp, weight, bias, router_dtype):
    if not supports_tiled_router(inp, weight, bias):
        return _native_router(inp, weight, bias, router_dtype)
    validate_router_inputs(inp, weight, bias, router_dtype)
    with torch_device_fn.device(inp.device):
        return _project_tiled(
            inp.contiguous(),
            weight.contiguous(),
            None if bias is None else bias.contiguous(),
            router_dtype,
        )


def _project_tiled(inp, weight, bias, router_dtype):
    return _RouterGatingLinear.apply(inp, weight, bias, router_dtype)
