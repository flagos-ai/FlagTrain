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


"""Megatron default WeightedSwiGLUFunction at d2b0e35cc4f05bc6b70734d558617050fbbb51ed."""

import torch
from torch.nn import functional as F

OP_NAME = "weighted_swiglu_funciton"
OVERRIDE_KEY = f"megatron.{OP_NAME}"
SHAPES = (
    (16, 32),
    (2, 5, 64),
    (1, 2),
    (33, 258),
    (128, 2048),
    (2, 128, 4096),
    (256, 8192),
)
DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def make_inputs(shape, dtype, device):
    generator = torch.Generator().manual_seed(6466)
    y = torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
    weights = torch.rand((*shape[:-1], 1), generator=generator).to(device=device)
    return y, weights


class _WeightedSwiGLUReference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, weights):
        ctx.save_for_backward(y, weights)
        gate, linear = y.chunk(2, dim=-1)
        return (F.silu(gate) * linear * weights).to(y.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        y, weights = ctx.saved_tensors
        gate, linear = y.chunk(2, dim=-1)
        sigmoid_gate = gate.sigmoid()
        scaled_grad = grad_output * weights
        grad_gate = (
            scaled_grad * sigmoid_gate * (1 + gate * (1 - sigmoid_gate)) * linear
        )
        silu_gate = F.silu(gate)
        grad_y = torch.cat((grad_gate, scaled_grad * silu_gate), dim=-1).to(y.dtype)
        grad_weights = (silu_gate * linear * grad_output.to(weights.dtype)).sum(
            dim=-1, keepdim=True
        )
        return grad_y, grad_weights.to(weights.dtype)


def reference(y, weights):
    return _WeightedSwiGLUReference.apply(y, weights)
