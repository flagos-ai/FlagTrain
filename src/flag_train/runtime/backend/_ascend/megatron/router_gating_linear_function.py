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

"""Triton Router gating for Ascend mixed-precision inputs."""

import torch
import triton
import triton.language as tl

from flag_train.megatron.router_gating_linear_function import (
    router_gating_linear_function as _generic_router,
)
from flag_train.runtime import torch_device_fn

__all__ = ["router_gating_linear_function"]

_FP32 = 0
_FP16 = 1
_BF16 = 2

_RD_ID = {torch.float32: _FP32, torch.float16: _FP16, torch.bfloat16: _BF16}
_RD_TORCH = {_FP32: torch.float32, _FP16: torch.float16, _BF16: torch.bfloat16}


def _np2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@triton.jit
def _rowoff(offs_m, M, D0, S0, D1, S1, D2, S2, D3, S3, ND: tl.constexpr):
    """Linear element offset of logical row offs_m for a strided (M, K) view."""
    m = tl.where(offs_m < M, offs_m, 0)
    if ND == 1:
        return m * S0
    rem = m
    if ND == 2:
        i1 = rem % D1
        rem = rem // D1
        return i1 * S1 + rem * S0
    elif ND == 3:
        i2 = rem % D2
        rem = rem // D2
        i1 = rem % D1
        rem = rem // D1
        return i2 * S2 + i1 * S1 + rem * S0
    else:
        i3 = rem % D3
        rem = rem // D3
        i2 = rem % D2
        rem = rem // D2
        i1 = rem % D1
        rem = rem // D1
        return i3 * S3 + i2 * S2 + i1 * S1 + rem * S0


@triton.jit
def _fseg(
    A,
    W,
    rowoff,
    offs_n,
    k_lo,
    k_hi,
    K,
    sa_k,
    sw_e,
    sw_k,
    rowmask,
    nmask,
    RD: tl.constexpr,
    COMP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Accumulate the K reduction, optionally with a compensated sum."""
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if COMP:
        # Compensate tile accumulation for cancellation in the K reduction.
        comp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        kk = k0 + offs_k
        kmask = kk < K
        a = tl.load(
            A + rowoff[:, None] + kk[None, :] * sa_k,
            mask=rowmask[:, None] & kmask[None, :],
            other=0.0,
        )
        w = tl.load(
            W + offs_n[:, None] * sw_e + kk[None, :] * sw_k,
            mask=nmask[:, None] & kmask[None, :],
            other=0.0,
        )
        # Cast operands for the cube dot before the FP32 accumulation.
        if RD == 2:
            part = tl.dot(a.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)))
        else:
            part = tl.dot(a.to(tl.float16), tl.trans(w.to(tl.float16)))
        if COMP:
            tot = acc + part
            big = tl.abs(acc) >= tl.abs(part)
            comp += tl.where(big, (acc - tot) + part, (part - tot) + acc)
            acc = tot
        else:
            acc = acc + part
    if COMP:
        acc = acc + comp
    return acc


@triton.jit(do_not_specialize=["M", "E"])
def _fwd_kernel(
    A,
    W,
    B,
    Out,
    M,
    K,
    E,
    NUM_M,
    sa_k,
    sw_e,
    sw_k,
    sb,
    D0,
    S0,
    D1,
    S1,
    D2,
    S2,
    D3,
    S3,
    ND: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RD: tl.constexpr,
    COMP: tl.constexpr,
    SPLIT2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid % NUM_M
    pid_n = pid // NUM_M
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    rowoff = _rowoff(offs_m, M, D0, S0, D1, S1, D2, S2, D3, S3, ND)
    rowmask = offs_m < M
    nmask = offs_n < E

    if SPLIT2:
        # Accumulate each half of the reduction before merging in FP32.
        half = (K // 2 // BLOCK_K) * BLOCK_K
        acc0 = _fseg(
            A,
            W,
            rowoff,
            offs_n,
            0,
            half,
            K,
            sa_k,
            sw_e,
            sw_k,
            rowmask,
            nmask,
            RD=RD,
            COMP=COMP,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
        acc1 = _fseg(
            A,
            W,
            rowoff,
            offs_n,
            half,
            K,
            K,
            sa_k,
            sw_e,
            sw_k,
            rowmask,
            nmask,
            RD=RD,
            COMP=COMP,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
        acc = acc0 + acc1
    else:
        acc = _fseg(
            A,
            W,
            rowoff,
            offs_n,
            0,
            K,
            K,
            sa_k,
            sw_e,
            sw_k,
            rowmask,
            nmask,
            RD=RD,
            COMP=COMP,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

    if HAS_BIAS:
        # torch.addmm receives bias.to(router_dtype) and adds it inside the
        # fp32 epilogue, so the result is rounded to router_dtype only once.
        bv = tl.load(B + offs_n * sb, mask=nmask, other=0.0).to(tl.float32)
        if RD == 1:
            bv = bv.to(tl.float16).to(tl.float32)
        elif RD == 2:
            bv = bv.to(tl.bfloat16).to(tl.float32)
        acc = acc + bv[None, :]

    if RD == 0:
        y = acc
    elif RD == 1:
        y = acc.to(tl.float16)
    else:
        y = acc.to(tl.bfloat16)

    tl.store(
        Out + offs_m[:, None] * E + offs_n[None, :],
        y,
        mask=rowmask[:, None] & nmask[None, :],
    )


@triton.jit(do_not_specialize=["M", "E"])
def _bcast_kernel(
    B,
    Out,
    M,
    E,
    sb,
    HAS_BIAS: tl.constexpr,
    RD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Empty-reduction case (K == 0): logits are just the broadcast bias."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    nmask = offs_n < E
    if HAS_BIAS:
        bv = tl.load(B + offs_n * sb, mask=nmask, other=0.0).to(tl.float32)
    else:
        bv = tl.zeros((BLOCK_N,), dtype=tl.float32)
    tile = bv[None, :] + tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if RD == 1:
        y = tile.to(tl.float16)
    elif RD == 2:
        y = tile.to(tl.bfloat16)
    else:
        y = tile
    tl.store(
        Out + offs_m[:, None] * E + offs_n[None, :],
        y,
        mask=(offs_m[:, None] < M) & nmask[None, :],
    )


@triton.jit(do_not_specialize=["M", "E"])
def _dgrad_kernel(
    G,
    W,
    GO,
    M,
    K,
    E,
    NUM_M,
    sg_e,
    sw_e,
    sw_k,
    D0,
    S0,
    D1,
    S1,
    D2,
    S2,
    D3,
    S3,
    ND: tl.constexpr,
    RD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid % NUM_M
    pid_k = pid // NUM_M
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_e = tl.arange(0, BLOCK_E)

    rowoff = _rowoff(offs_m, M, D0, S0, D1, S1, D2, S2, D3, S3, ND)
    rowmask = offs_m < M
    kmask = offs_k < K
    emask = offs_e < E

    gt = tl.load(
        G + rowoff[:, None] + offs_e[None, :] * sg_e,
        mask=rowmask[:, None] & emask[None, :],
        other=0.0,
    )
    wt = tl.load(
        W + offs_e[:, None] * sw_e + offs_k[None, :] * sw_k,
        mask=emask[:, None] & kmask[None, :],
        other=0.0,
    )

    if RD == 1:
        acc = tl.dot(gt.to(tl.float16), wt.to(tl.float16))
    elif RD == 2:
        acc = tl.dot(gt.to(tl.bfloat16), wt.to(tl.bfloat16))
    else:
        g32 = gt.to(tl.float32)
        g0 = g32.to(tl.bfloat16)
        g1 = (g32 - g0.to(tl.float32)).to(tl.bfloat16)
        w32 = wt.to(tl.float32)
        w0 = w32.to(tl.bfloat16)
        w1 = (w32 - w0.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(g0, w0)
        acc = tl.dot(g0, w1, acc)
        acc = tl.dot(g1, w0, acc)

    if RD == 1:
        y = acc.to(tl.float16).to(tl.bfloat16)
    else:
        y = acc.to(tl.bfloat16)

    tl.store(
        GO + offs_m[:, None] * K + offs_k[None, :],
        y,
        mask=rowmask[:, None] & kmask[None, :],
    )


@triton.jit(do_not_specialize=["M", "E"])
def _wgrad_kernel(
    G,
    X,
    GW,
    M,
    K,
    E,
    sg_e,
    sx_k,
    GD0,
    GS0,
    GD1,
    GS1,
    GD2,
    GS2,
    GD3,
    GS3,
    XD0,
    XS0,
    XD1,
    XS1,
    XD2,
    XS2,
    XD3,
    XS3,
    NDG: tl.constexpr,
    NDX: tl.constexpr,
    RD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid_k = tl.program_id(0)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_e = tl.arange(0, BLOCK_E)
    kmask = offs_k < K
    emask = offs_e < E

    acc = tl.zeros((BLOCK_E, BLOCK_K), dtype=tl.float32)
    for m0 in range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        rowmask = offs_m < M
        goff = _rowoff(offs_m, M, GD0, GS0, GD1, GS1, GD2, GS2, GD3, GS3, NDG)
        xoff = _rowoff(offs_m, M, XD0, XS0, XD1, XS1, XD2, XS2, XD3, XS3, NDX)
        gt = tl.load(
            G + goff[:, None] + offs_e[None, :] * sg_e,
            mask=rowmask[:, None] & emask[None, :],
            other=0.0,
        )
        xt = tl.load(
            X + xoff[:, None] + offs_k[None, :] * sx_k,
            mask=rowmask[:, None] & kmask[None, :],
            other=0.0,
        )
        if RD == 1:
            acc = tl.dot(tl.trans(gt.to(tl.float16)), xt.to(tl.float16), acc)
        elif RD == 2:
            acc = tl.dot(tl.trans(gt.to(tl.bfloat16)), xt.to(tl.bfloat16), acc)
        else:
            g32 = gt.to(tl.float32)
            g0 = g32.to(tl.bfloat16)
            g1 = (g32 - g0.to(tl.float32)).to(tl.bfloat16)
            xb = xt.to(tl.bfloat16)
            acc = tl.dot(tl.trans(g0), xb, acc)
            acc = tl.dot(tl.trans(g1), xb, acc)

    if RD == 2:
        y = acc.to(tl.bfloat16).to(tl.float16)
    else:
        y = acc.to(tl.float16)

    tl.store(
        GW + offs_e[:, None] * K + offs_k[None, :],
        y,
        mask=emask[:, None] & kmask[None, :],
    )


@triton.jit(do_not_specialize=["M", "E"])
def _dbias_kernel(
    G,
    GB,
    M,
    E,
    sg_e,
    D0,
    S0,
    D1,
    S1,
    D2,
    S2,
    D3,
    S3,
    ND: tl.constexpr,
    RD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    offs_e = tl.arange(0, BLOCK_E)
    emask = offs_e < E
    acc = tl.zeros((BLOCK_E,), dtype=tl.float32)
    for m0 in range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        rowmask = offs_m < M
        rowoff = _rowoff(offs_m, M, D0, S0, D1, S1, D2, S2, D3, S3, ND)
        g = tl.load(
            G + rowoff[:, None] + offs_e[None, :] * sg_e,
            mask=rowmask[:, None] & emask[None, :],
            other=0.0,
        )
        acc = acc + tl.sum(g.to(tl.float32), axis=0)
    # Megatron rounds the reduction to router_dtype (the matmul output dtype)
    # and autograd then converts it to weight.dtype.
    if RD == 2:
        acc = acc.to(tl.bfloat16).to(tl.float32)
    acc = acc.to(tl.float16).to(tl.float32)
    tl.store(GB + offs_e, acc, mask=emask)


def _geom(t):
    """Describe t as a logical (M, K) matrix; return (M, k_stride, nd, dims, strides)."""
    shape = t.shape
    strides = t.stride()
    sk = int(strides[-1])
    lead = shape[:-1]
    lstr = strides[:-1]
    nl = len(lead)
    M = 1
    for d in lead:
        M *= int(d)
    if nl == 0:
        return M, sk, 1, (1, 0, 1, 0, 1, 0, 1, 0)
    dims = []
    strs = []
    cur_d = int(lead[-1])
    cur_s = int(lstr[-1])
    for i in range(nl - 2, -1, -1):
        if int(lstr[i]) == cur_d * cur_s:
            cur_d = int(lead[i]) * cur_d
        else:
            dims.append(cur_d)
            strs.append(cur_s)
            cur_d = int(lead[i])
            cur_s = int(lstr[i])
    dims.append(cur_d)
    strs.append(cur_s)
    dims.reverse()
    strs.reverse()
    nd = len(dims)
    if nd > 4:
        raise RuntimeError("unsupported input rank for router gating linear")
    flat = []
    for i in range(4):
        if i < nd:
            flat.append(dims[i])
            flat.append(strs[i])
        else:
            flat.append(1)
            flat.append(0)
    return M, sk, nd, tuple(flat)


def _bk_of(K):
    p = _np2(max(int(K), 1))
    if p >= 2048:
        return 32
    bk = min(128, p)
    if bk == 64:
        bk = 32
    return max(16, bk)


class _RouterGatingLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, weight, bias, rd):
        E = int(weight.shape[0])
        K = int(weight.shape[1])
        M, sa_k, nd, geo = _geom(inp)
        out = torch.empty(
            tuple(inp.shape[:-1]) + (E,),
            dtype=_RD_TORCH[rd],
            device=inp.device,
        )
        BN = max(16, min(256, _np2(max(E, 1))))
        sb = int(bias.stride(-1)) if bias is not None else 1
        if M > 0 and E > 0:
            BM = max(16, min(64, _np2(max(1, (M + 15) // 16))))
            BK = _bk_of(K)
            # Use compensated accumulation for long low-precision reductions.
            comp = K >= 4096 and rd != _FP32
            split2 = False
            if comp:
                BK = 64
                BM = max(16, min(32, _np2(max(1, (M + 15) // 16))))
            if K == 0:
                _bcast_kernel[((M + BM - 1) // BM,)](
                    bias if bias is not None else out,
                    out,
                    M,
                    E,
                    sb,
                    HAS_BIAS=bias is not None,
                    RD=rd,
                    BLOCK_M=BM,
                    BLOCK_N=BN,
                )
            else:
                num_m = (M + BM - 1) // BM
                num_n = (E + BN - 1) // BN
                d0, s0, d1, s1, d2, s2, d3, s3 = geo
                _fwd_kernel[(num_m * num_n,)](
                    inp,
                    weight,
                    bias if bias is not None else inp,
                    out,
                    M,
                    K,
                    E,
                    num_m,
                    sa_k,
                    weight.stride(0),
                    weight.stride(1),
                    sb,
                    d0,
                    s0,
                    d1,
                    s1,
                    d2,
                    s2,
                    d3,
                    s3,
                    ND=nd,
                    HAS_BIAS=bias is not None,
                    RD=rd,
                    COMP=comp,
                    SPLIT2=split2,
                    BLOCK_M=BM,
                    BLOCK_N=BN,
                    BLOCK_K=BK,
                )
        if bias is None:
            ctx.save_for_backward(inp, weight)
            ctx.has_bias = False
        else:
            ctx.save_for_backward(inp, weight, bias)
            ctx.has_bias = True
        ctx.rd = rd
        return out

    @staticmethod
    def backward(ctx, grad_out):
        rd = ctx.rd
        if ctx.has_bias:
            inp, weight, bias = ctx.saved_tensors
        else:
            inp, weight = ctx.saved_tensors
            bias = None
        with torch_device_fn.device(inp.device):
            E = int(weight.shape[0])
            K = int(weight.shape[1])
            M, sx_k, ndx, xgeo = _geom(inp)
            gM, sg_e, ndg, ggeo = _geom(grad_out)

            grad_inp = torch.empty(tuple(inp.shape), dtype=inp.dtype, device=inp.device)
            grad_weight = torch.empty((E, K), dtype=weight.dtype, device=weight.device)

            gd0, gs0, gd1, gs1, gd2, gs2, gd3, gs3 = ggeo
            xd0, xs0, xd1, xs1, xd2, xs2, xd3, xs3 = xgeo
            BE = max(16, min(256, _np2(max(E, 1))))

            if M > 0 and K > 0:
                BM = 32
                BK = _bk_of(K)
                num_m = (M + BM - 1) // BM
                num_k = (K + BK - 1) // BK
                _dgrad_kernel[(num_m * num_k,)](
                    grad_out,
                    weight,
                    grad_inp,
                    M,
                    K,
                    E,
                    num_m,
                    sg_e,
                    weight.stride(0),
                    weight.stride(1),
                    gd0,
                    gs0,
                    gd1,
                    gs1,
                    gd2,
                    gs2,
                    gd3,
                    gs3,
                    ND=ndg,
                    RD=rd,
                    BLOCK_M=BM,
                    BLOCK_K=BK,
                    BLOCK_E=BE,
                )

            if K > 0:
                # weight gradient is required even when M == 0 (it is all zeros).
                BKW = min(64, _bk_of(K))
                BMW = 64
                _wgrad_kernel[((K + BKW - 1) // BKW,)](
                    grad_out,
                    inp,
                    grad_weight,
                    M,
                    K,
                    E,
                    sg_e,
                    sx_k,
                    gd0,
                    gs0,
                    gd1,
                    gs1,
                    gd2,
                    gs2,
                    gd3,
                    gs3,
                    xd0,
                    xs0,
                    xd1,
                    xs1,
                    xd2,
                    xs2,
                    xd3,
                    xs3,
                    NDG=ndg,
                    NDX=ndx,
                    RD=rd,
                    BLOCK_M=BMW,
                    BLOCK_K=BKW,
                    BLOCK_E=BE,
                )

            if bias is not None:
                grad_bias = torch.empty((E,), dtype=bias.dtype, device=bias.device)
                # A non-unit element stride on grad_output turns the load into a
                # gather that needs far more local buffer, so keep that tile small.
                if sg_e == 1:
                    BMD = max(16, min(256, 8192 // BE))
                else:
                    BMD = 16
                _dbias_kernel[(1,)](
                    grad_out,
                    grad_bias,
                    gM,
                    E,
                    sg_e,
                    gd0,
                    gs0,
                    gd1,
                    gs1,
                    gd2,
                    gs2,
                    gd3,
                    gs3,
                    ND=ndg,
                    RD=rd,
                    BLOCK_M=BMD,
                    BLOCK_E=BE,
                )
            else:
                grad_bias = None

            return grad_inp, grad_weight, grad_bias, None


def router_gating_linear_function(inp, weight, bias, router_dtype):
    supports_mixed_inputs = (
        2 <= inp.ndim <= 5
        and weight.ndim == 2
        and inp.shape[-1] == weight.shape[-1]
        and inp.dtype == torch.bfloat16
        and weight.dtype == torch.float16
        and inp.device == weight.device
        and router_dtype in _RD_ID
        and (
            bias is None
            or (
                bias.dtype == torch.float32
                and bias.shape == (weight.shape[0],)
                and bias.device == inp.device
            )
        )
    )
    if not supports_mixed_inputs:
        return _generic_router(inp, weight, bias, router_dtype)
    with torch_device_fn.device(inp.device):
        return _RouterGatingLinear.apply(inp, weight, bias, _RD_ID[router_dtype])
