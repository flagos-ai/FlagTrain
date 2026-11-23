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


"""Fused weighted SwiGLU forward + backward implemented in Triton.

out[..., i] = round_y( round_y(silu(gate_i)) * linear_i ) * w[row]   (staged path)
grad_y       = concat(d_gate, d_linear)
grad_w[row]  = sum_i silu(gate_i) * linear_i * grad_out_i   (fp32)
"""

import torch
import triton
import triton.language as tl

# CUDA Triton ships libdevice under extra.cuda; FlagTree (Hygon/BW1000)
# only provides the backend-neutral extra.libdevice.
try:
    from triton.language.extra.cuda import libdevice
except ImportError:
    from triton.language.extra import libdevice

from flag_train.runtime import torch_device_fn

# ---------------------------------------------------------------------------
# tunables
# ---------------------------------------------------------------------------
_STAGED_ROUND = True  # emulate the reference's per-op rounding to y.dtype
_INTERLEAVED = False  # layout of the (gate, linear) pair inside the last dim
_TARGET_ROWS = 4096  # preferred number of elements handled per program
_MIN_PROGRAMS = 1056  # try to keep at least this many CTAs resident


def _cdiv(a, b):
    return (a + b - 1) // b


@triton.jit
def _exp(value, USE_LIBDEVICE: tl.constexpr):
    if USE_LIBDEVICE:
        return libdevice.exp(value)
    return tl.exp(value)


@triton.jit
def _fwd_kernel(
    y_ptr,
    w_ptr,
    o_ptr,
    D,
    M,
    NSPLIT: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    STAGED: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    USE_LIBDEVICE: tl.constexpr,
):
    pid_c = tl.program_id(1)
    pid_m = tl.program_id(0)
    nchunks = tl.cdiv(D, BLOCK)
    for r in tl.static_range(ROWS):
        row = pid_m * ROWS + r
        if row < M:
            w = tl.load(w_ptr + row).to(tl.float32)
            row64 = row.to(tl.int64)
            y_base = y_ptr + row64 * (2 * D)
            o_base = o_ptr + row64 * D
            for k in range(0, tl.cdiv(nchunks - pid_c, NSPLIT)):
                offs = (pid_c + k * NSPLIT) * BLOCK + tl.arange(0, BLOCK)
                if INTERLEAVED:
                    goff = 2 * offs
                    loff = 2 * offs + 1
                else:
                    goff = offs
                    loff = offs + D
                if EVEN:
                    g = tl.load(y_base + goff).to(tl.float32)
                    lin = tl.load(y_base + loff).to(tl.float32)
                else:
                    msk = offs < D
                    g = tl.load(y_base + goff, mask=msk, other=0.0).to(tl.float32)
                    lin = tl.load(y_base + loff, mask=msk, other=0.0).to(tl.float32)
                t = tl.div_rn(g, 1.0 + _exp(-g, USE_LIBDEVICE))
                if STAGED:
                    t = t.to(o_ptr.dtype.element_ty).to(tl.float32)
                    t = t * lin
                    t = t.to(o_ptr.dtype.element_ty).to(tl.float32)
                else:
                    t = t * lin
                o = t * w
                if EVEN:
                    tl.store(o_base + offs, o.to(o_ptr.dtype.element_ty))
                else:
                    tl.store(o_base + offs, o.to(o_ptr.dtype.element_ty), mask=offs < D)


@triton.jit
def _bwd_kernel(
    y_ptr,
    w_ptr,
    go_ptr,
    gy_ptr,
    gw_ptr,
    D,
    M,
    NSPLIT: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    STAGED: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    USE_LIBDEVICE: tl.constexpr,
    DO_DX: tl.constexpr,
    DO_DW: tl.constexpr,
    ATOMIC_DW: tl.constexpr,
):
    pid_c = tl.program_id(1)
    pid_m = tl.program_id(0)
    nchunks = tl.cdiv(D, BLOCK)
    for r in tl.static_range(ROWS):
        row = pid_m * ROWS + r
        if row < M:
            w = tl.load(w_ptr + row).to(tl.float32)
            row64 = row.to(tl.int64)
            y_base = y_ptr + row64 * (2 * D)
            go_base = go_ptr + row64 * D
            gy_base = gy_ptr + row64 * (2 * D)
            acc = tl.zeros([BLOCK], tl.float32)
            for k in range(0, tl.cdiv(nchunks - pid_c, NSPLIT)):
                offs = (pid_c + k * NSPLIT) * BLOCK + tl.arange(0, BLOCK)
                if INTERLEAVED:
                    goff = 2 * offs
                    loff = 2 * offs + 1
                else:
                    goff = offs
                    loff = offs + D
                if EVEN:
                    g = tl.load(y_base + goff).to(tl.float32)
                    lin = tl.load(y_base + loff).to(tl.float32)
                    go = tl.load(go_base + offs).to(tl.float32)
                else:
                    msk = offs < D
                    g = tl.load(y_base + goff, mask=msk, other=0.0).to(tl.float32)
                    lin = tl.load(y_base + loff, mask=msk, other=0.0).to(tl.float32)
                    go = tl.load(go_base + offs, mask=msk, other=0.0).to(tl.float32)
                den = 1.0 + _exp(-g, USE_LIBDEVICE)
                s = tl.div_rn(1.0, den)
                silu = tl.div_rn(g, den)
                if STAGED:
                    silu = silu.to(gy_ptr.dtype.element_ty).to(tl.float32)
                    t = (silu * lin).to(gy_ptr.dtype.element_ty).to(tl.float32)
                else:
                    t = silu * lin
                gd = go * w
                dtype = y_ptr.dtype.element_ty
                sigmoid = s.to(dtype).to(tl.float32)
                one_minus = (1.0 - sigmoid).to(dtype).to(tl.float32)
                gain = (g * one_minus).to(dtype).to(tl.float32)
                term = (1.0 + gain).to(dtype).to(tl.float32)
                d_gate = ((gd * sigmoid) * term) * lin
                d_lin = gd * silu
                if DO_DX:
                    if EVEN:
                        tl.store(gy_base + goff, d_gate.to(gy_ptr.dtype.element_ty))
                        tl.store(gy_base + loff, d_lin.to(gy_ptr.dtype.element_ty))
                    else:
                        tl.store(
                            gy_base + goff, d_gate.to(gy_ptr.dtype.element_ty), mask=msk
                        )
                        tl.store(
                            gy_base + loff, d_lin.to(gy_ptr.dtype.element_ty), mask=msk
                        )
                if DO_DW:
                    acc += t * go
            if DO_DW:
                dw = tl.sum(acc, axis=0)
                if ATOMIC_DW:
                    tl.atomic_add(gw_ptr + row, dw)
                else:
                    tl.store(gw_ptr + row, dw)


def _plan(D, M, use_libdevice):
    if D <= 256:
        BLOCK = 256
    elif D <= 512:
        BLOCK = 512
    elif D <= 1024:
        BLOCK = 1024
    elif D <= 2048 or not use_libdevice:
        BLOCK = 2048
    else:
        BLOCK = 4096
    want_rows = max(1, _TARGET_ROWS // max(D, 1))
    limit_rows = max(1, M // _MIN_PROGRAMS)
    ROWS = max(1, min(want_rows, limit_rows))
    nchunks = _cdiv(D, BLOCK)
    rows_grid = _cdiv(M, ROWS)
    NSPLIT = 1
    if nchunks > 1 and rows_grid < _MIN_PROGRAMS:
        NSPLIT = min(nchunks, max(1, _cdiv(_MIN_PROGRAMS, rows_grid)))
    warps = 8 if BLOCK >= 2048 else 4
    return BLOCK, ROWS, NSPLIT, (D % BLOCK == 0), warps


def _staged(dtype):
    return _STAGED_ROUND and dtype in (torch.float16, torch.bfloat16)


def _weighted_swiglu(y, weights, use_libdevice):
    if y.ndim == 0 or y.shape[-1] % 2:
        raise ValueError("y must have an even final dimension")
    if weights.shape != (*y.shape[:-1], 1):
        raise ValueError("weights must contain one value per token")
    if weights.device != y.device:
        raise ValueError("weights and y must be on the same device")
    if not y.is_contiguous():
        y = y.contiguous()
    if not weights.is_contiguous():
        weights = weights.contiguous()
    return _WeightedSwiGLU.apply(y, weights, use_libdevice)


class _WeightedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, weights, use_libdevice):
        D = y.shape[-1] // 2
        M = y.numel() // (2 * D) if D > 0 else 0
        out = torch.empty(y.shape[:-1] + (D,), dtype=y.dtype, device=y.device)
        ctx.use_libdevice = use_libdevice
        ctx.D = D
        ctx.M = M
        if M > 0:
            BLOCK, ROWS, NSPLIT, EVEN, warps = _plan(D, M, ctx.use_libdevice)
            grid = (_cdiv(M, ROWS), NSPLIT)
            with torch_device_fn.device(y.device):
                _fwd_kernel[grid](
                    y,
                    weights.view(-1),
                    out,
                    D,
                    M,
                    NSPLIT=NSPLIT,
                    ROWS=ROWS,
                    BLOCK=BLOCK,
                    EVEN=EVEN,
                    STAGED=_staged(y.dtype),
                    INTERLEAVED=_INTERLEAVED,
                    USE_LIBDEVICE=ctx.use_libdevice,
                    num_warps=warps,
                )
        ctx.save_for_backward(y, weights)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        y, weights = ctx.saved_tensors
        D = ctx.D
        M = ctx.M
        need_y = ctx.needs_input_grad[0]
        need_w = ctx.needs_input_grad[1]
        if not need_y and not need_w:
            return None, None, None
        if M == 0:
            return (
                torch.zeros_like(y) if need_y else None,
                torch.zeros_like(weights) if need_w else None,
                None,
            )
        go = grad_out
        if not go.is_contiguous():
            go = go.contiguous()
        BLOCK, ROWS, NSPLIT, EVEN, warps = _plan(D, M, ctx.use_libdevice)
        atomic = NSPLIT > 1
        grad_y = torch.empty_like(y) if need_y else y
        if need_w:
            if atomic:
                grad_w = torch.zeros(M, dtype=torch.float32, device=y.device)
            else:
                grad_w = torch.empty(M, dtype=torch.float32, device=y.device)
        else:
            grad_w = torch.empty(1, dtype=torch.float32, device=y.device)
        if M > 0:
            grid = (_cdiv(M, ROWS), NSPLIT)
            with torch_device_fn.device(y.device):
                _bwd_kernel[grid](
                    y,
                    weights.view(-1),
                    go,
                    grad_y,
                    grad_w,
                    D,
                    M,
                    NSPLIT=NSPLIT,
                    ROWS=ROWS,
                    BLOCK=BLOCK,
                    EVEN=EVEN,
                    STAGED=_staged(y.dtype),
                    INTERLEAVED=_INTERLEAVED,
                    USE_LIBDEVICE=ctx.use_libdevice,
                    DO_DX=need_y,
                    DO_DW=need_w,
                    ATOMIC_DW=atomic,
                    num_warps=warps,
                    enable_fp_fusion=False,
                )
        if need_w:
            grad_w = grad_w.view(weights.shape)
        else:
            grad_w = None
        return (grad_y if need_y else None), grad_w, None


def weighted_swiglu_funciton(y, weights):
    return _weighted_swiglu(y, weights, use_libdevice=True)
