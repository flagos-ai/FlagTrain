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

"""P800 weighted SwiGLU with native low-precision activations and Triton products."""

import torch
import triton
import triton.language as tl
from torch.nn import functional as F

from flag_train.megatron.weighted_swiglu_funciton import (
    weighted_swiglu_funciton as _generic,
)
from flag_train.runtime import torch_device_fn


@triton.jit
def _round_product(value, dtype: tl.constexpr):
    if dtype == tl.bfloat16:
        # Materialize the BF16 rounding at this point even when XPU lowers
        # a BF16-to-FP32 round trip into a no-op.
        bits = value.to(tl.uint32, bitcast=True)
        finite = (bits & 0x7F800000) != 0x7F800000
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        value = tl.where(finite, rounded, bits).to(tl.float32, bitcast=True)
    elif dtype == tl.float16:
        value = value.to(tl.float16).to(tl.float32)
    return value


@triton.jit
def _forward_kernel(Y, SILU, W, OUT, N, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = index // WIDTH
    column = index % WIDTH
    mask = index < N
    silu = tl.load(SILU + index, mask, 0).to(tl.float32)
    linear = tl.load(Y + row * (2 * WIDTH) + WIDTH + column, mask, 0).to(tl.float32)
    weight = tl.load(W + row, mask, 0).to(tl.float32)
    product = _round_product(silu * linear, Y.dtype.element_ty)
    tl.store(OUT + index, product * weight, mask)


@triton.jit
def _backward_kernel(
    Y,
    SILU,
    SIG,
    DERIV_FACTOR,
    W,
    G,
    DY_GATE,
    DY_LINEAR,
    CONTRIBUTION,
    N,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = index // WIDTH
    column = index % WIDTH
    mask = index < N
    lin_idx = row * (2 * WIDTH) + WIDTH + column
    linear = tl.load(Y + lin_idx, mask, 0).to(tl.float32)
    silu = tl.load(SILU + index, mask, 0).to(tl.float32)
    sig = tl.load(SIG + index, mask, 0).to(tl.float32)
    factor = tl.load(DERIV_FACTOR + index, mask, 0).to(tl.float32)
    weight = tl.load(W + row, mask, 0).to(tl.float32)
    grad = tl.load(G + index, mask, 0).to(tl.float32)

    scaled_grad = grad * weight
    tl.store(DY_GATE + index, scaled_grad * sig * factor * linear, mask)
    tl.store(DY_LINEAR + index, scaled_grad * silu, mask)
    product = _round_product(silu * linear, Y.dtype.element_ty)
    tl.store(CONTRIBUTION + index, product * grad, mask)


def _forward(y, weight):
    shape = y.shape[:-1] + (y.shape[-1] // 2,)
    if y.numel() // 2 >= 32768:
        # The XPU Torch kernels have higher throughput for large tensors.
        gate, linear = y.chunk(2, dim=-1)
        return (F.silu(gate) * linear * weight).to(y.dtype)
    out = torch.empty(shape, dtype=y.dtype, device=y.device)
    if not out.numel():
        return out
    y, weight = y.contiguous(), weight.contiguous()
    gate = y[..., : shape[-1]]
    silu = F.silu(gate).contiguous()
    block = 256
    _forward_kernel[(triton.cdiv(out.numel(), block),)](
        y,
        silu,
        weight,
        out,
        out.numel(),
        shape[-1],
        block,
        enable_fp_fusion=False,
    )
    return out


class _WeightedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, weight):
        ctx.save_for_backward(y, weight)
        return _forward(y, weight)

    @staticmethod
    def backward(ctx, grad):
        y, weight = ctx.saved_tensors
        if not grad.numel():
            return torch.zeros_like(y), torch.zeros_like(weight)
        y, weight, grad = y.contiguous(), weight.contiguous(), grad.contiguous()
        width = grad.shape[-1]
        gate = y[..., :width]
        silu = F.silu(gate).contiguous()
        sigmoid = gate.sigmoid().contiguous()
        factor = (1 + gate * (1 - sigmoid)).contiguous()
        dy_gate = torch.empty_like(grad)
        dy_linear = torch.empty_like(grad)
        contribution = torch.empty_like(grad, dtype=torch.float32)
        block = 256
        with torch_device_fn.device(y.device):
            _backward_kernel[(triton.cdiv(grad.numel(), block),)](
                y,
                silu,
                sigmoid,
                factor,
                weight,
                grad,
                dy_gate,
                dy_linear,
                contribution,
                grad.numel(),
                width,
                block,
                enable_fp_fusion=False,
            )
        dy = torch.cat((dy_gate, dy_linear), dim=-1)
        return dy, contribution.sum(dim=-1, keepdim=True).to(weight.dtype)


def weighted_swiglu_funciton(y, weights):
    """Compute weighted SwiGLU using XPU3-native low-precision rounding."""
    if (
        y.ndim == 0
        or y.shape[-1] % 2
        or weights.shape != y.shape[:-1] + (1,)
        or weights.device != y.device
        or weights.dtype != torch.float32
        or y.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or y.numel() >= 2**31
    ):
        return _generic(y, weights)
    with torch_device_fn.device(y.device):
        if triton.runtime.driver.active.get_current_target().arch != 3:
            return _generic(y, weights)
        if torch.is_grad_enabled() and (y.requires_grad or weights.requires_grad):
            return _WeightedSwiGLU.apply(y, weights)
        return _forward(y, weights)
