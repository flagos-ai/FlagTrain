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


"""Triton backward for Transformer Engine L2 normalization."""

import torch
import triton
import triton.language as tl

from flag_train.runtime import torch_device_fn


@triton.jit
def _l2_bwd_small(G, X, RN, DX, M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Register-resident single pass: M fits in BLOCK_N, one row per program."""
    row = tl.program_id(0).to(tl.int64)
    r = tl.load(RN + row).to(tl.float32)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < M
    base = row * M
    g = tl.load(G + base + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + base + offs, mask=mask, other=0.0).to(tl.float32)
    dot = tl.sum(g * x, axis=0)
    out = r * (g - (r * r * dot) * x)
    tl.store(DX + base + offs, out.to(DX.dtype.element_ty), mask=mask)


@triton.jit
def _l2_bwd_gen(
    G,
    X,
    RN,
    DX,
    M,
    stride_g_row,
    stride_x_row,
    stride_dx_row,
    stride_rn,
    BLOCK_N: tl.constexpr,
):
    """Generic fallback: explicit strides, streaming two-pass over the last dim."""
    row = tl.program_id(0).to(tl.int64)
    r = tl.load(RN + row * stride_rn).to(tl.float32)
    g_row = G + row * stride_g_row
    x_row = X + row * stride_x_row

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for n0 in range(0, M, BLOCK_N):
        offs = n0 + tl.arange(0, BLOCK_N)
        mask = offs < M
        g = tl.load(g_row + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        acc += g * x
    dot = tl.sum(acc, axis=0)
    scale = r * r * dot

    dx_row = DX + row * stride_dx_row
    for n0 in range(0, M, BLOCK_N):
        offs = n0 + tl.arange(0, BLOCK_N)
        mask = offs < M
        g = tl.load(g_row + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            dx_row + offs, (r * (g - scale * x)).to(DX.dtype.element_ty), mask=mask
        )


def l2_normalization_backward(grad_output, x, rsqrt_norm, eps):
    """Compute the input gradient from the inverse norm saved in forward."""
    if x.ndim == 0 or grad_output.shape != x.shape or grad_output.dtype != x.dtype:
        raise ValueError("grad_output and x must have the same tensor shape and dtype")
    if grad_output.device != x.device or rsqrt_norm.device != x.device:
        raise ValueError("grad_output, x and rsqrt_norm must be on the same device")
    if not x.numel():
        return torch.empty_like(x)
    width = x.shape[-1]
    rows = x.numel() // width
    if rsqrt_norm.numel() != rows or rsqrt_norm.dtype != torch.float32:
        raise ValueError("rsqrt_norm must contain one FP32 inverse norm per row")
    x = x.contiguous()
    grad_output = grad_output.contiguous()
    rsqrt_norm = rsqrt_norm.contiguous()
    grad_x = torch.empty_like(x)
    with torch_device_fn.device(x.device):
        if width <= 2048:
            _l2_bwd_small[(rows,)](
                grad_output, x, rsqrt_norm, grad_x, width, triton.next_power_of_2(width)
            )
        else:
            _l2_bwd_gen[(rows,)](
                grad_output,
                x,
                rsqrt_norm,
                grad_x,
                width,
                width,
                width,
                width,
                1,
                BLOCK_N=1024,
            )
    return grad_x
