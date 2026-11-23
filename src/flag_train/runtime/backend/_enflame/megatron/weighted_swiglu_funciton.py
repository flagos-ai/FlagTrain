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

"""Weighted SwiGLU using 32-bit indexing and staged rounding on GCU300."""

import torch
import triton
import triton.language as tl

from flag_train.megatron.weighted_swiglu_funciton import (
    weighted_swiglu_funciton as _generic,
)
from flag_train.runtime import torch_device_fn

from .. import arch_version


# Forward: out = round_y( round_y(silu(gate)) * linear ) * w
# silu(g) = g * (1 / (1 + exp(-g))) evaluated in fp32 with a single rounding to
# the y dtype, mirroring the reference's fp32-opmath pointwise SiLU.
@triton.jit(do_not_specialize=["d"])
def _wswiglu_fwd_kernel(y_ptr, w_ptr, o_ptr, d, BD: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1)
    col = c * BD + tl.arange(0, BD)
    mask = col < d
    base = y_ptr + r * (2 * d)
    g = tl.load(base + col, mask=mask, other=0.0)
    lin = tl.load(base + d + col, mask=mask, other=0.0)
    w = tl.load(w_ptr + r)

    et = y_ptr.dtype.element_ty
    g32 = g.to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-g32))
    s = (g32 * sig).to(et)
    p = (s.to(tl.float32) * lin.to(tl.float32)).to(et)
    out = (p.to(tl.float32) * w.to(tl.float32)).to(et)
    tl.store(o_ptr + r * d + col, out, mask=mask)


# Backward stage 1: grad_y (both halves) and, when the row fits in one tile,
# the fp32 grad_weights row sum; otherwise per-tile fp32 partials.
@triton.jit(do_not_specialize=["d"])
def _wswiglu_bwd_kernel(
    y_ptr, w_ptr, go_ptr, gy_ptr, gw_ptr, d, PARTIAL: tl.constexpr, BD: tl.constexpr
):
    r = tl.program_id(0)
    c = tl.program_id(1)
    col = c * BD + tl.arange(0, BD)
    mask = col < d
    base = y_ptr + r * (2 * d)
    g = tl.load(base + col, mask=mask, other=0.0)
    lin = tl.load(base + d + col, mask=mask, other=0.0)
    go = tl.load(go_ptr + r * d + col, mask=mask, other=0.0)
    w = tl.load(w_ptr + r)

    et = y_ptr.dtype.element_ty
    g32 = g.to(tl.float32)
    sig32f = 1.0 / (1.0 + tl.exp(-g32))
    sig = sig32f.to(et)
    s = (g32 * sig32f).to(et)

    sig32 = sig.to(tl.float32)
    s32 = s.to(tl.float32)
    lin32 = lin.to(tl.float32)
    go32 = go.to(tl.float32)
    scaled = go32 * w.to(tl.float32)

    one_minus = (1.0 - sig32).to(et)
    q = (1.0 + (g32 * one_minus.to(tl.float32)).to(et).to(tl.float32)).to(et)
    q32 = q.to(tl.float32)

    gy_gate = (scaled * sig32 * q32 * lin32).to(et)
    gy_lin = (scaled * s32).to(et)
    gybase = gy_ptr + r * (2 * d)
    tl.store(gybase + col, gy_gate, mask=mask)
    tl.store(gybase + d + col, gy_lin, mask=mask)

    p32 = (s32 * lin32).to(et).to(tl.float32)
    contrib = tl.where(mask, p32 * go32, 0.0)
    row_sum = tl.sum(contrib, axis=0)
    if PARTIAL:
        cb = (d + BD - 1) // BD
        tl.store(gw_ptr + r * cb + c, row_sum)
    else:
        tl.store(gw_ptr + r, row_sum)


# Backward stage 2: reduce the fp32 column-block partials into grad_weights.
@triton.jit(do_not_specialize=["cb"])
def _wswiglu_gw_kernel(part_ptr, gw_ptr, cb, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < cb
    vals = tl.load(part_ptr + r * cb + cols, mask=mask, other=0.0)
    tl.store(gw_ptr + r, tl.sum(vals, axis=0))


def _tile(d):
    bd = min(1024, max(64, triton.next_power_of_2(d)))
    return bd, 1 if bd <= 128 else 4


def _forward(y, weights):
    d = y.shape[-1] // 2
    out = torch.empty(y.shape[:-1] + (d,), dtype=y.dtype, device=y.device)
    if d == 0:
        return out
    n_rows = y.numel() // (2 * d)
    if n_rows == 0:
        return out
    bd, nw = _tile(d)
    cb = (d + bd - 1) // bd
    _wswiglu_fwd_kernel[(n_rows, cb)](y, weights, out, d, bd, num_warps=nw)
    return out


def _backward(y, weights, grad_out):
    d = y.shape[-1] // 2
    grad_y = torch.empty_like(y)
    grad_w = torch.empty_like(weights)
    if d == 0:
        return grad_y, torch.zeros_like(weights)
    n_rows = y.numel() // (2 * d)
    if n_rows == 0:
        return grad_y, grad_w
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()
    bd, nw = _tile(d)
    cb = (d + bd - 1) // bd
    if cb == 1:
        _wswiglu_bwd_kernel[(n_rows, 1)](
            y, weights, grad_out, grad_y, grad_w, d, False, bd, num_warps=nw
        )
    else:
        part = torch.empty((n_rows, cb), dtype=torch.float32, device=y.device)
        _wswiglu_bwd_kernel[(n_rows, cb)](
            y, weights, grad_out, grad_y, part, d, True, bd, num_warps=nw
        )
        _wswiglu_gw_kernel[(n_rows,)](part, grad_w, cb, triton.next_power_of_2(cb))
    return grad_y, grad_w


class _WeightedSwiGLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, y, weights):
        out = _forward(y, weights)
        ctx.save_for_backward(y, weights)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        y, weights = ctx.saved_tensors
        with torch_device_fn.device(y.device):
            grad_y, grad_w = _backward(y, weights, grad_out)
        return grad_y, grad_w


def weighted_swiglu_funciton(y, weights):
    """Compute weighted SwiGLU on GCU300."""
    if (
        arch_version != 300
        or y.ndim == 0
        or y.shape[-1] % 2
        or weights.shape != y.shape[:-1] + (1,)
        or weights.device != y.device
        or weights.dtype != torch.float32
        or y.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or y.numel() >= 2**31
    ):
        return _generic(y, weights)
    y, weights = y.contiguous(), weights.contiguous()
    with torch_device_fn.device(y.device):
        if torch.is_grad_enabled() and (y.requires_grad or weights.requires_grad):
            return _WeightedSwiGLU.apply(y, weights)
        return _forward(y, weights)
