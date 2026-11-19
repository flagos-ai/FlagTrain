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

"""Triton implementation of Megatron router gating and its input gradients."""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn

# ---------------------------------------------------------------------------
# dtype ids:  0 -> bfloat16, 1 -> float16, 2 -> float32
# ---------------------------------------------------------------------------
_DT_BF16 = 0
_DT_FP16 = 1
_DT_FP32 = 2


@lru_cache(maxsize=None)
def _sm_count(device):
    return torch_device_fn.get_device_properties(device).multi_processor_count


# ---------------------------------------------------------------------------
# Triton helpers
# ---------------------------------------------------------------------------
@triton.jit
def _cvt(x, CD: tl.constexpr):
    """cast an operand tile to the MMA dtype"""
    if CD == 2:
        return x.to(tl.bfloat16)
    elif CD == 1:
        return x.to(tl.float16)
    else:
        return x.to(tl.float32)


@triton.jit
def _mm(a, b, acc, CD: tl.constexpr):
    if CD == 0:
        return tl.dot(a, b, acc, input_precision="ieee")
    else:
        return tl.dot(a, b, acc)


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


# ---------------------------------------------------------------------------
# forward:  O[M,E] = A[M,K] @ W[E,K]^T  (+ bias)
# ---------------------------------------------------------------------------
@triton.jit
def _fwd_kernel(
    A,
    W,
    B,
    O,
    M,
    K,
    E,
    stride_am,
    stride_we,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CD: tl.constexpr,
    DT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SPLIT_REDUCTION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < E

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if SPLIT_REDUCTION:
        for k0 in range(0, K // 2, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            a = tl.load(
                A + offs_m[:, None] * stride_am + offs_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            b = tl.load(
                W + offs_n[:, None] * stride_we + offs_k[None, :],
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
            )
            acc = _mm(_cvt(a, CD), tl.trans(_cvt(b, CD)), acc, CD)

        first = acc
        acc = tl.full((BLOCK_M, BLOCK_N), 0.0, tl.float32)
        for k0 in range(K // 2, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            a = tl.load(
                A + offs_m[:, None] * stride_am + offs_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            b = tl.load(
                W + offs_n[:, None] * stride_we + offs_k[None, :],
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
            )
            acc = _mm(_cvt(a, CD), tl.trans(_cvt(b, CD)), acc, CD)

        acc = first + acc
    else:
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            a = tl.load(
                A + offs_m[:, None] * stride_am + offs_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            b = tl.load(
                W + offs_n[:, None] * stride_we + offs_k[None, :],
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
            )
            acc = _mm(_cvt(a, CD), tl.trans(_cvt(b, CD)), acc, CD)

    if HAS_BIAS:
        bv = tl.load(B + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        bv = _rnd32(bv, DT)
        acc = acc + bv[None, :]

    tl.store(
        O + offs_m[:, None] * stride_om + offs_n[None, :],
        _rnd(acc, DT),
        mask=mask_m[:, None] & mask_n[None, :],
    )


# ---------------------------------------------------------------------------
# backward wrt input:  DA[M,K] = DO[M,E] @ W[E,K]
# ---------------------------------------------------------------------------
@triton.jit
def _dag_kernel(
    DO,
    W,
    DA,
    M,
    K,
    E,
    stride_dm,
    stride_we,
    stride_am,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    CD: tl.constexpr,
    DT: tl.constexpr,
    RD_DT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for e0 in range(0, E, BLOCK_E):
        offs_e = e0 + tl.arange(0, BLOCK_E)
        mask_e = offs_e < E
        d = tl.load(
            DO + offs_m[:, None] * stride_dm + offs_e[None, :],
            mask=mask_m[:, None] & mask_e[None, :],
            other=0.0,
        )
        w = tl.load(
            W + offs_e[:, None] * stride_we + offs_k[None, :],
            mask=mask_e[:, None] & mask_k[None, :],
            other=0.0,
        )
        acc = _mm(_cvt(d, CD), _cvt(w, CD), acc, CD)

    tl.store(
        DA + offs_m[:, None] * stride_am + offs_k[None, :],
        _rnd(_rnd32(acc, RD_DT), DT),
        mask=mask_m[:, None] & mask_k[None, :],
    )


# ---------------------------------------------------------------------------
# backward wrt weight:  DW[E,K] = DO[M,E]^T @ A[M,K]   (fp32 partials)
# ---------------------------------------------------------------------------
@triton.jit
def _dwg_kernel(
    DO,
    A,
    DP,
    M,
    E,
    K,
    stride_dm,
    stride_am,
    stride_ps,
    stride_pe,
    n_blk,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CD: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_s = tl.program_id(2)
    offs_e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_e = offs_e < E
    mask_k = offs_k < K

    m_lo = pid_s * n_blk * BLOCK_M
    m_hi = m_lo + n_blk * BLOCK_M

    acc = tl.zeros((BLOCK_E, BLOCK_K), dtype=tl.float32)
    for m0 in range(m_lo, m_hi, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        d = tl.load(
            DO + offs_m[:, None] * stride_dm + offs_e[None, :],
            mask=mask_m[:, None] & mask_e[None, :],
            other=0.0,
        )
        a = tl.load(
            A + offs_m[:, None] * stride_am + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        acc = _mm(tl.trans(_cvt(d, CD)), _cvt(a, CD), acc, CD)

    tl.store(
        DP + pid_s * stride_ps + offs_e[:, None] * stride_pe + offs_k[None, :],
        acc,
        mask=mask_e[:, None] & mask_k[None, :],
    )


# ---------------------------------------------------------------------------
# backward wrt bias:  DB[E] = sum_m DO[m,E]   (fp32 partials)
# ---------------------------------------------------------------------------
@triton.jit
def _dbg_kernel(
    DO,
    BP,
    M,
    E,
    stride_dm,
    stride_bs,
    n_blk,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid_s = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E

    m_lo = pid_s * n_blk * BLOCK_M
    m_hi = m_lo + n_blk * BLOCK_M

    acc = tl.zeros((BLOCK_E,), dtype=tl.float32)
    for m0 in range(m_lo, m_hi, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        d = tl.load(
            DO + offs_m[:, None] * stride_dm + offs_e[None, :],
            mask=mask_m[:, None] & mask_e[None, :],
            other=0.0,
        )
        acc = acc + tl.sum(d.to(tl.float32), axis=0)

    tl.store(BP + pid_s * stride_bs + offs_e, acc, mask=mask_e)


# ---------------------------------------------------------------------------
# stage-2 reduction for the dw / db partials
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# host-side launch helpers
# ---------------------------------------------------------------------------
def _clamp_pow2(x):
    v = triton.next_power_of_2(max(int(x), 1))
    if v < 16:
        v = 16
    return v


def _matmul_warps(cd, tile_area):
    # Two warps keep FP16/BF16 on warp-level MMA. Multi-warp-group
    # tiles have produced incorrect results on Hopper.
    if cd:
        return 2
    return 8 if tile_area >= 128 * 128 else 4


def _fwd_cfg(M, E, K, cd):
    # Keep two-warp accumulator tiles small enough to avoid register spilling.
    bn = min(_clamp_pow2(E), 64 if cd else 256)
    bm = min(64 if cd else 128, 16384 // bn)
    while bm > 16 and bm > M:
        bm //= 2
    bk = 64
    if K < 64:
        bk = _clamp_pow2(K)
    nw = _matmul_warps(cd, bm * bn)
    return bm, bn, bk, nw


@triton.jit
def _split4(tile, ROWS: tl.constexpr):
    even, odd = tl.split(tl.reshape(tile, (ROWS, 2, 2)))
    zero, two = tl.split(even)
    one, three = tl.split(odd)
    return zero, one, two, three


@triton.jit
def _split8(tile, ROWS: tl.constexpr):
    lo, hi = tl.split(tl.reshape(tile, (ROWS, 4, 2)))
    a, c, e, g = _split4(lo, ROWS)
    b, d, f, h = _split4(hi, ROWS)
    return a, b, c, d, e, f, g, h


@triton.jit
def _serial_split_k(
    A,
    W,
    P,
    E: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    CHUNK: tl.constexpr,
    SPLITS: tl.constexpr,
    STRIPED: tl.constexpr,
):
    split = tl.program_id(2)
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.program_id(1) * BLOCK_E + tl.arange(0, BLOCK_E)
    lane = tl.arange(0, 8)
    acc = tl.full((BLOCK_M, BLOCK_E), 0, tl.float32)
    for group in range(CHUNK // 8):
        if STRIPED:
            k = (group * SPLITS + split) * 8 + lane
        else:
            k = split * CHUNK + group * 8 + lane
        a = tl.load(
            A + row[:, None] * K + k[None, :], (row[:, None] < M) & (k[None, :] < K), 0
        ).to(tl.float32)
        b = tl.load(
            W + col[:, None] * K + k[None, :], (col[:, None] < E) & (k[None, :] < K), 0
        ).to(tl.float32)
        left = _split8(a, BLOCK_M)
        right = _split8(b, BLOCK_E)
        for idx in tl.static_range(8):
            acc = tl.fma(left[idx][:, None], right[idx][None, :], acc)
    offsets = row[:, None] * E + col[None, :]
    tl.store(P + split * M * E + offsets, acc, (row[:, None] < M) & (col[None, :] < E))


@triton.jit
def _merge_serial_splits(
    P,
    B,
    Y,
    M: tl.constexpr,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BIAS_FIRST: tl.constexpr,
):
    output = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = output < M * E
    bias = tl.full((BLOCK,), 0, tl.float32)
    if HAS_BIAS:
        bias = tl.load(B + output % E, valid, 0)
    acc = bias if BIAS_FIRST else tl.full((BLOCK,), 0, tl.float32)
    for split in tl.static_range(SPLITS):
        acc += tl.load(P + split * M * E + output, valid, 0)
    if not BIAS_FIRST:
        acc += bias
    tl.store(Y + output, acc, valid)


def _fp32_router_forward(x, w, bias, out, rows, width, experts):
    # Match the H800 SGEMM reduction order. CUTLASS uses consecutive K
    # chunks with bias. The no-bias xmma kernel stripes 8 values over splits.
    striped = bias is None and width > 1024
    add_bias_first = bias is not None and width > 2048
    if striped:
        parts = 16 if width <= 2048 else 8
        chunk = triton.cdiv(width, parts * 8) * 8
    else:
        if width <= 1024:
            chunk = 128 if bias is not None else 64
        else:
            chunk = 320 if width <= 2048 else 512
        parts = triton.cdiv(width, chunk)
    partials = torch.empty((parts, rows, experts), dtype=torch.float32, device=x.device)
    tiles = triton.cdiv(rows * experts, 64)
    block_m, block_e = 16, 16
    grid = (triton.cdiv(rows, block_m), triton.cdiv(experts, block_e), parts)
    _serial_split_k[grid](
        x,
        w,
        partials,
        experts,
        width,
        rows,
        block_m,
        block_e,
        chunk,
        parts,
        striped,
    )
    _merge_serial_splits[(tiles,)](
        partials,
        bias if bias is not None else out,
        out,
        rows,
        experts,
        64,
        parts,
        bias is not None,
        add_bias_first,
    )


def _launch_fwd(a2d, w, bias, o2d, M, E, K, cd, dt):
    bm, bn, bk, nw = _fwd_cfg(M, E, K, cd)
    grid = (triton.cdiv(M, bm), triton.cdiv(E, bn))
    bptr = bias if bias is not None else a2d
    _fwd_kernel[grid](
        a2d,
        w,
        bptr,
        o2d,
        M,
        K,
        E,
        a2d.stride(0),
        w.stride(0),
        o2d.stride(0),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        CD=cd,
        DT=dt,
        HAS_BIAS=bias is not None,
        SPLIT_REDUCTION=bias is None and cd != 0 and K >= 4096 and K % (2 * bk) == 0,
        num_warps=nw,
        num_stages=3,
    )


def _launch_da(do2d, w, da2d, M, E, K, cd, dt, rd_dt):
    bm = 64 if cd else 128
    while bm > 16 and bm > M:
        bm //= 2
    bk = min(_clamp_pow2(K), 64 if cd else 128)
    be = min(_clamp_pow2(E), 32)
    nw = _matmul_warps(cd, bm * bk)
    grid = (triton.cdiv(M, bm), triton.cdiv(K, bk))
    _dag_kernel[grid](
        do2d,
        w,
        da2d,
        M,
        K,
        E,
        do2d.stride(0),
        w.stride(0),
        da2d.stride(0),
        BLOCK_M=bm,
        BLOCK_K=bk,
        BLOCK_E=be,
        CD=cd,
        DT=dt,
        RD_DT=rd_dt,
        num_warps=nw,
        num_stages=3,
    )


def _launch_dw(do2d, a2d, M, E, K, cd):
    be = min(_clamp_pow2(E), 128)
    bk = 128 if K >= 128 else _clamp_pow2(K)
    bm = 64
    n_et = triton.cdiv(E, be)
    n_kt = triton.cdiv(K, bk)
    progs = n_et * n_kt
    sm = _sm_count(do2d.device)
    need = max(1, -(-(2 * sm) // progs))
    n_mb = triton.cdiv(M, bm)
    split = max(1, min(need, n_mb))
    # keep the fp32 partial traffic a small fraction of the A stream
    budget = max(1, (M * K * 2) // max(1, 4 * E * K))
    split = max(1, min(split, budget))
    if cd == 0 and M >= 512:
        # The FP32 SIMT path accumulates eight consecutive token chunks.
        split = 8
    elif cd != 0:
        # Hopper nvjet does not split the token reduction for these tiles.
        be = 64 if E >= 64 else be
        bk = 64
        n_et = triton.cdiv(E, be)
        n_kt = triton.cdiv(K, bk)
        split = 1
    n_blk = triton.cdiv(n_mb, split)
    dpart = torch.empty((split, E, K), dtype=torch.float32, device=do2d.device)
    nw = _matmul_warps(cd, be * bk)
    grid = (n_et, n_kt, split)
    _dwg_kernel[grid](
        do2d,
        a2d,
        dpart,
        M,
        E,
        K,
        do2d.stride(0),
        a2d.stride(0),
        dpart.stride(0),
        dpart.stride(1),
        n_blk,
        BLOCK_M=bm,
        BLOCK_E=be,
        BLOCK_K=bk,
        CD=cd,
        num_warps=nw,
        num_stages=3,
    )
    return dpart, be, bk, n_et * n_kt, n_kt, split


def _launch_db(do2d, M, E):
    be = min(_clamp_pow2(E), 256)
    bm = 128
    n_mb = triton.cdiv(M, bm)
    split = max(1, min(n_mb, _sm_count(do2d.device)))
    n_blk = triton.cdiv(n_mb, split)
    bpart = torch.empty((split, E), dtype=torch.float32, device=do2d.device)
    _dbg_kernel[(split,)](
        do2d,
        bpart,
        M,
        E,
        do2d.stride(0),
        bpart.stride(0),
        n_blk,
        BLOCK_M=bm,
        BLOCK_E=be,
        num_warps=4,
        num_stages=3,
    )
    return bpart, be, split


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


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


# ---------------------------------------------------------------------------
# autograd wrapper
# ---------------------------------------------------------------------------
class _RouterGatingLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, weight, bias, rd):
        E, K = int(weight.shape[0]), int(weight.shape[1])
        in_shape = tuple(inp.shape)
        M = 1
        for d in in_shape[:-1]:
            M *= int(d)
        dt = _dt_of(rd)
        cd = _cd_of(inp.dtype, weight.dtype, rd)

        rows = inp.reshape(M, K)
        logits = torch.empty(in_shape[:-1] + (E,), dtype=rd, device=inp.device)
        if K == 0:
            if bias is None:
                logits.zero_()
            else:
                logits.copy_(bias.to(rd))
        elif M > 0 and E > 0:
            if rd == torch.float32 and K > 128:
                _fp32_router_forward(rows, weight, bias, logits, M, K, E)
            else:
                _launch_fwd(rows, weight, bias, logits.view(M, E), M, E, K, cd, dt)

        ctx.save_for_backward(inp, weight)
        ctx.dt = dt
        ctx.cd = cd
        ctx.M = M
        ctx.E = E
        ctx.K = K
        ctx.in_shape = in_shape
        return logits

    @staticmethod
    def backward(ctx, dout):
        inp, weight = ctx.saved_tensors
        M, E, K = ctx.M, ctx.E, ctx.K
        needs = ctx.needs_input_grad
        cd = ctx.cd
        w_dt = _dt_of(weight.dtype)
        in_dt = _dt_of(inp.dtype)

        din = None
        dweight = None
        dbias = None

        if M * E * K == 0:
            if needs[0]:
                din = torch.zeros(ctx.in_shape, dtype=inp.dtype, device=inp.device)
            if needs[1]:
                dweight = torch.zeros((E, K), dtype=weight.dtype, device=weight.device)
            if needs[2]:
                dbias = dout.reshape(M, E).sum(dim=0).to(weight.dtype)
            return din, dweight, dbias, None

        with torch_device_fn.device(inp.device):
            do2d = dout.reshape(M, E).contiguous()
            if needs[0]:
                din = torch.empty(ctx.in_shape, dtype=inp.dtype, device=inp.device)
                _launch_da(do2d, weight, din.view(M, K), M, E, K, cd, in_dt, ctx.dt)

            dw_cfg = None
            db_cfg = None
            dpart = None
            bpart = None
            if needs[1]:
                dweight = torch.empty((E, K), dtype=weight.dtype, device=weight.device)
                dpart, be, bk, n_tile, n_k, split = _launch_dw(
                    do2d, inp.reshape(M, K), M, E, K, cd
                )
                dw_cfg = (be, bk, n_tile, n_k, split)
            if needs[2]:
                dbias = torch.empty((E,), dtype=torch.float32, device=inp.device)
                bpart, be_b, split_b = _launch_db(do2d, M, E)
                db_cfg = (be_b, split_b)

            if dw_cfg is not None or db_cfg is not None:
                if dw_cfg is not None:
                    be_d, bk_d, n_dw_tile, n_k, split_d = dw_cfg
                    dp = dpart
                    dp_stride_s = dpart.stride(0)
                    dp_stride_e = dpart.stride(1)
                else:
                    be_d, bk_d, n_dw_tile, n_k, split_d = 16, 16, 0, 1, 1
                    dp = dpart if dpart is not None else bpart
                    dp_stride_s = 1
                    dp_stride_e = 1
                if db_cfg is not None:
                    be_b, split_b = db_cfg
                    bp = bpart
                    bp_stride_s = bpart.stride(0)
                else:
                    be_b, split_b = 16, 1
                    bp = bpart if bpart is not None else dpart
                    bp_stride_s = 1

                n_dw_all = n_dw_tile
                n_db_all = triton.cdiv(E, be_b) if db_cfg is not None else 0
                dw_ptr = dweight if dweight is not None else dp
                db_ptr = dbias if dbias is not None else bp
                dw_stride_e = dweight.stride(0) if dweight is not None else 1

                _final_kernel[(n_dw_all + n_db_all,)](
                    dp,
                    bp,
                    dw_ptr,
                    db_ptr,
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
                    RD_DT=ctx.dt,
                    HAS_DW=dw_cfg is not None,
                    HAS_DB=db_cfg is not None,
                    num_warps=4,
                    num_stages=2,
                )

            return din, dweight, dbias, None


def router_gating_linear_function(inp, weight, bias, router_dtype):
    """Project router logits and support first-order gradients on NVIDIA CUDA.

    Weight has shape (experts, hidden). Bias may be None or a vector of length
    experts. The output replaces the input's last dimension with experts.
    """
    if runtime_device.vendor_name != "nvidia" or inp.device.type != "cuda":
        raise NotImplementedError("Router gating linear requires an NVIDIA CUDA tensor")
    if inp.ndim < 2 or weight.ndim != 2 or inp.size(-1) != weight.size(1):
        raise ValueError("Expected input [..., hidden] and weight [experts, hidden]")
    if bias is not None and bias.shape != (weight.size(0),):
        raise ValueError("Bias must contain one value per expert")
    tensors = (inp, weight) if bias is None else (inp, weight, bias)
    if any(tensor.device != inp.device for tensor in tensors):
        raise ValueError("Input, weight and bias must be on the same device")
    if router_dtype not in _SUPPORTED_DTYPES or any(
        tensor.dtype not in _SUPPORTED_DTYPES for tensor in tensors
    ):
        raise ValueError("Router gating linear supports float32, float16 and bfloat16")

    with torch_device_fn.device(inp.device):
        return _RouterGatingLinear.apply(
            inp.contiguous(),
            weight.contiguous(),
            bias.contiguous() if bias is not None else None,
            router_dtype,
        )
