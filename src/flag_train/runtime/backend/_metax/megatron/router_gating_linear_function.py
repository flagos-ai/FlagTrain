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

"""Tiled Megatron router projection for metax."""

from math import prod

import torch
import triton
import triton.language as tl

from flag_train.runtime import torch_device_fn

_RD_FP32 = 0
_RD_FP16 = 1
_RD_BF16 = 2


def _cdiv(a, b):
    return (a + b - 1) // b


def _next_pow2(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


_SM_CACHE = {}


def _num_sm(device):
    try:
        idx = device.index
    except AttributeError:  # pragma: no cover - non tensor input
        idx = None
    if idx is None:
        try:
            idx = torch_device_fn.current_device()
        except Exception:
            idx = 0
    val = _SM_CACHE.get(idx)
    if val is None:
        try:
            val = int(torch_device_fn.get_device_properties(idx).multi_processor_count)
        except Exception:
            val = 0
        if val <= 0:
            val = 64
        _SM_CACHE[idx] = val
    return val


@triton.jit
def _load_tile(ptr, rmask, cmask, MASK_R: tl.constexpr, MASK_C: tl.constexpr):
    """Masked 2D tile load, with `other=0.0` so padded lanes add exact zeros."""
    if MASK_R:
        if MASK_C:
            v = tl.load(ptr, mask=rmask[:, None] & cmask[None, :], other=0.0)
        else:
            v = tl.load(ptr, mask=rmask[:, None], other=0.0)
    else:
        if MASK_C:
            v = tl.load(ptr, mask=cmask[None, :], other=0.0)
        else:
            v = tl.load(ptr)
    return v


@triton.jit
def _round_rd(x, RD: tl.constexpr):
    """Round an FP32 accumulator to the router dtype and widen back to FP32."""
    if RD == 0:
        y = x
    elif RD == 1:
        y = x.to(tl.float16).to(tl.float32)
    else:
        y = x.to(tl.bfloat16).to(tl.float32)
    return y


@triton.jit
def _fwd_kernel(
    A,
    W,
    Bias,
    O,
    M,
    K,
    E,
    sa_m,
    sa_k,
    sw_e,
    sw_k,
    sb_e,
    so_m,
    so_e,
    RD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MASK_M: tl.constexpr,
    MASK_N: tl.constexpr,
    MASK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < E

    a_ptrs = A + offs_m[:, None] * sa_m + offs_k[None, :] * sa_k
    w_ptrs = W + offs_n[:, None] * sw_e + offs_k[None, :] * sw_k

    # Compensated (Kahan) accumulation: each BLOCK_K tile is reduced by an
    # independent MMA chain and folded into the running total with its rounding
    # error captured in `cmp`.  This holds the K-long FP32 sum at the level of
    # the correctly rounded value instead of drifting over K/BLOCK_K chain steps,
    # which matters because the result is rounded only once, to router_dtype.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    cmp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = k0 + offs_k < K
        a = _load_tile(a_ptrs, m_mask, k_mask, MASK_M, MASK_K)
        w = _load_tile(w_ptrs, n_mask, k_mask, MASK_N, MASK_K)
        if RD == 2:
            a = a.to(tl.bfloat16)
            w = w.to(tl.bfloat16)
        else:
            a = a.to(tl.float16)
            w = w.to(tl.float16)
        d = tl.dot(a, tl.trans(w))
        t = acc + d
        cmp += (acc - t) + d
        acc = t
        a_ptrs += BLOCK_K * sa_k
        w_ptrs += BLOCK_K * sw_k
    acc = acc + cmp

    if HAS_BIAS:
        b = tl.load(Bias + offs_n * sb_e, mask=n_mask, other=0.0)
        if RD == 1:
            b = b.to(tl.float16).to(tl.float32)
        elif RD == 2:
            b = b.to(tl.bfloat16).to(tl.float32)
        acc = acc + b[None, :]

    o_ptrs = O + offs_m[:, None] * so_m + offs_n[None, :] * so_e
    tl.store(o_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _bwd_da_kernel(
    G,
    W,
    O,
    M,
    K,
    E,
    sg_m,
    sg_e,
    sw_e,
    sw_k,
    so_m,
    so_k,
    RD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    MASK_M: tl.constexpr,
    MASK_K: tl.constexpr,
    MASK_E: tl.constexpr,
    MASK_GM: tl.constexpr,
    MASK_WR: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_e = tl.arange(0, BLOCK_E)

    m_mask = offs_m < M
    k_mask = offs_k < K

    g_ptrs = G + offs_m[:, None] * sg_m + offs_e[None, :] * sg_e
    w_ptrs = W + offs_e[:, None] * sw_e + offs_k[None, :] * sw_k

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    cmp = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for e0 in range(0, E, BLOCK_E):
        e_mask = e0 + offs_e < E
        g = _load_tile(g_ptrs, m_mask, e_mask, MASK_GM, MASK_E)
        w = _load_tile(w_ptrs, e_mask, k_mask, MASK_WR, MASK_K)
        if RD == 0:
            # dL is full precision FP32: split into two FP16 halves (hi/lo) so the
            # tensor-core products stay faithful to the FP32 reference GEMM.
            gh = g.to(tl.float16)
            gl = (g - gh.to(tl.float32)).to(tl.float16)
            wh = w.to(tl.float16)
            d = tl.dot(gh, wh) + tl.dot(gl, wh)
        elif RD == 2:
            d = tl.dot(g.to(tl.bfloat16), w.to(tl.bfloat16))
        else:
            d = tl.dot(g.to(tl.float16), w.to(tl.float16))
        t = acc + d
        cmp += (acc - t) + d
        acc = t
        g_ptrs += BLOCK_E * sg_e
        w_ptrs += BLOCK_E * sw_e
    acc = acc + cmp

    y = _round_rd(acc, RD)
    o_ptrs = O + offs_m[:, None] * so_m + offs_k[None, :] * so_k
    tl.store(o_ptrs, y, mask=m_mask[:, None] & k_mask[None, :])


@triton.jit
def _bwd_dw_kernel(
    G,
    A,
    O,
    M,
    K,
    E,
    sg_m,
    sg_e,
    sa_m,
    sa_k,
    so_e,
    so_k,
    RD: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MASK_E: tl.constexpr,
    MASK_K: tl.constexpr,
    MASK_M: tl.constexpr,
    MASK_GE: tl.constexpr,
    MASK_AR: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_e = tl.program_id(1)

    offs_e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, BLOCK_M)

    e_mask = offs_e < E
    k_mask = offs_k < K

    g_ptrs = G + offs_e[:, None] * sg_e + offs_m[None, :] * sg_m
    a_ptrs = A + offs_m[:, None] * sa_m + offs_k[None, :] * sa_k

    acc = tl.zeros((BLOCK_E, BLOCK_K), dtype=tl.float32)
    cmp = tl.zeros((BLOCK_E, BLOCK_K), dtype=tl.float32)
    for m0 in range(0, M, BLOCK_M):
        m_mask = m0 + offs_m < M
        g = _load_tile(g_ptrs, e_mask, m_mask, MASK_GE, MASK_M)
        a = _load_tile(a_ptrs, m_mask, k_mask, MASK_AR, MASK_K)
        if RD == 0:
            gh = g.to(tl.float16)
            gl = (g - gh.to(tl.float32)).to(tl.float16)
            ah = a.to(tl.float16)
            d = tl.dot(gh, ah) + tl.dot(gl, ah)
        elif RD == 2:
            d = tl.dot(g.to(tl.bfloat16), a.to(tl.bfloat16))
        else:
            d = tl.dot(g.to(tl.float16), a.to(tl.float16))
        t = acc + d
        cmp += (acc - t) + d
        acc = t
        g_ptrs += BLOCK_M * sg_m
        a_ptrs += BLOCK_M * sa_m
    acc = acc + cmp

    y = _round_rd(acc, RD)
    o_ptrs = O + offs_e[:, None] * so_e + offs_k[None, :] * so_k
    tl.store(o_ptrs, y, mask=e_mask[:, None] & k_mask[None, :])


@triton.jit
def _bwd_db_kernel(
    G,
    O,
    M,
    E,
    sg_m,
    sg_e,
    so_e,
    RD: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid_e = tl.program_id(0)
    offs_e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = offs_e < E

    acc = tl.zeros((BLOCK_E,), dtype=tl.float32)
    g_ptrs = G + offs_e * sg_e
    for _m in range(0, M):
        v = tl.load(g_ptrs, mask=e_mask, other=0.0)
        acc = acc + v.to(tl.float32)
        g_ptrs += sg_m

    y = _round_rd(acc, RD)
    y = y.to(tl.float16).to(tl.float32)
    tl.store(O + offs_e * so_e, y, mask=e_mask)


_FWD_PLAN = {}


def _fwd_plan(M, E, K, rd, has_bias, device):
    key = (M, E, K, rd, has_bias, device.index if device.index is not None else -1)
    plan = _FWD_PLAN.get(key)
    if plan is not None:
        return plan

    bk = _next_pow2(min(K, 128) if K > 0 else 1)
    bk = max(16, min(128, bk))
    bn = max(16, min(128, _next_pow2(max(E, 1))))

    if M * E * K <= (1 << 22):
        bm = max(16, min(64, _next_pow2(max(M, 1))))
        warps, stages = 4, 2
    else:
        bm = 16
        bk = min(bk, 64)
        warps, stages = 4, 2
        nsm = _num_sm(device)
        # Keep the expert tile as wide as the shape allows (floor 64) so each
        # input row tile is fetched once; the grid's fastest axis is the expert
        # axis, so the weight tile stays resident in L2 across row blocks.
        while bn > 64 and _cdiv(M, bm) * _cdiv(E, bn) < nsm:
            bn //= 2

    mask_k = (K % bk) != 0
    plan = (
        bm,
        bn,
        bk,
        warps,
        stages,
        ((M % bm) != 0) or mask_k,
        ((E % bn) != 0) or mask_k,
        mask_k,
        (_cdiv(E, bn), _cdiv(M, bm)),
    )
    _FWD_PLAN[key] = plan
    return plan


def _forward(inp, weight, bias, rd, rd_dtype):
    K = inp.shape[-1]
    E = weight.shape[0]
    M = 1
    for s in inp.shape[:-1]:
        M *= s

    out = torch.empty(inp.shape[:-1] + (E,), dtype=rd_dtype, device=inp.device)
    if out.numel() == 0:
        return out
    if rd_dtype == torch.float16 and K >= 2048:
        rows = inp.reshape(M, K).to(rd_dtype)
        columns = weight.t().to(rd_dtype)
        logits = (
            torch.mm(rows, columns)
            if bias is None
            else torch.addmm(bias.to(rd_dtype), rows, columns)
        )
        return logits.reshape_as(out)

    bm, bn, bk, warps, stages, m_mask, n_mask, k_mask, grid = _fwd_plan(
        M, E, K, rd, bias is not None, inp.device
    )

    _fwd_kernel[grid](
        inp,
        weight,
        bias if bias is not None else out,
        out,
        M,
        K,
        E,
        inp.stride(-2),
        inp.stride(-1),
        weight.stride(0),
        weight.stride(1),
        bias.stride(0) if bias is not None else 0,
        E,
        1,
        RD=rd,
        HAS_BIAS=bias is not None,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        MASK_M=m_mask,
        MASK_N=n_mask,
        MASK_K=k_mask,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def _backward(inp, weight, bias, rd, grad_out):
    K = inp.shape[-1]
    E = weight.shape[0]
    M = 1
    for s in inp.shape[:-1]:
        M *= s

    dA = torch.empty(inp.shape, dtype=inp.dtype, device=inp.device)
    if rd == _RD_BF16 and grad_out is not None:
        rows = grad_out.reshape(M, E)
        dW = torch.mm(rows.t(), inp.reshape(M, K).to(rows.dtype)).to(weight.dtype)
    else:
        dW = torch.empty_like(weight, memory_format=torch.contiguous_format)
    db = (
        None
        if bias is None
        else torch.empty(bias.shape, dtype=torch.float32, device=bias.device)
    )

    if grad_out is None:
        return None, None, None

    sg_m = grad_out.stride(-2)
    sg_e = grad_out.stride(-1)
    sa_m = inp.stride(-2)
    sa_k = inp.stride(-1)
    sw_e = weight.stride(0)
    sw_k = weight.stride(1)

    bk = max(16, min(128, _next_pow2(max(min(K, 64), 1))))
    be_w = max(16, min(64, _next_pow2(max(E, 1))))
    bk_w = max(16, min(32, _next_pow2(max(min(K, 32), 1))))

    if dA.numel() > 0:
        bm = 32
        be_da = max(16, min(32, _next_pow2(max(E, 1))))
        mask_m = (M % bm) != 0
        mask_k = (K % bk) != 0
        mask_e = (E % be_da) != 0
        _bwd_da_kernel[(_cdiv(K, bk), _cdiv(M, bm))](
            grad_out,
            weight,
            dA,
            M,
            K,
            E,
            sg_m,
            sg_e,
            sw_e,
            sw_k,
            K,
            1,
            RD=rd,
            BLOCK_M=bm,
            BLOCK_K=bk,
            BLOCK_E=be_da,
            MASK_M=mask_m,
            MASK_K=mask_k,
            MASK_E=mask_e,
            MASK_GM=mask_m or mask_e,
            MASK_WR=mask_e or mask_k,
            num_warps=4,
            num_stages=2,
        )

    if rd != _RD_BF16 and dW.numel() > 0:
        bmm = 16
        mask_m = (M % bmm) != 0
        mask_k = (K % bk_w) != 0
        mask_e = (E % be_w) != 0
        _bwd_dw_kernel[(_cdiv(K, bk_w), _cdiv(E, be_w))](
            grad_out,
            inp,
            dW,
            M,
            K,
            E,
            sg_m,
            sg_e,
            sa_m,
            sa_k,
            K,
            1,
            RD=rd,
            BLOCK_E=be_w,
            BLOCK_K=bk_w,
            BLOCK_M=bmm,
            MASK_E=mask_e,
            MASK_K=mask_k,
            MASK_M=mask_m,
            MASK_GE=mask_e or mask_m,
            MASK_AR=mask_m or mask_k,
            num_warps=4,
            num_stages=2,
        )

    if db is not None and db.numel() > 0:
        bdb = max(16, min(128, _next_pow2(max(E, 1))))
        _bwd_db_kernel[(_cdiv(E, bdb),)](
            grad_out,
            db,
            M,
            E,
            sg_m,
            sg_e,
            1,
            RD=rd,
            BLOCK_E=bdb,
            num_warps=1,
            num_stages=1,
        )

    return dA, dW, db


class _RouterGatingFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight, bias, rd, rd_dtype):
        out = _forward(inp, weight, bias, rd, rd_dtype)
        ctx.rd = rd
        ctx.save_for_backward(inp, weight, bias)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        inp, weight, bias = ctx.saved_tensors
        dA, dW, db = _backward(inp, weight, bias, ctx.rd, grad_out.contiguous())
        return dA, dW, db, None, None


def _rd_of(router_dtype):
    if router_dtype is torch.float16:
        return _RD_FP16, torch.float16
    if router_dtype is torch.bfloat16:
        return _RD_BF16, torch.bfloat16
    return _RD_FP32, torch.float32


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
    rd, rd_dtype = _rd_of(router_dtype)
    needs_grad = (
        inp.requires_grad
        or weight.requires_grad
        or (bias is not None and bias.requires_grad)
    )
    if needs_grad and torch.is_grad_enabled():
        return _RouterGatingFn.apply(inp, weight, bias, rd, rd_dtype)
    return _forward(inp, weight, bias, rd, rd_dtype)
