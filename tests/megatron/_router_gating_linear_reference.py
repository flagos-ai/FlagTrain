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

"""Megatron router Torch fallback at d2b0e35cc4f05bc6b70734d558617050fbbb51ed."""

import torch

OP_NAME = "router_gating_linear_function"
OVERRIDE_KEY = f"megatron.{OP_NAME}"
WORKLOADS = (
    ((16, 32), 13),
    ((2, 5, 32), 13),
    ((1, 64), 1),
    ((33, 127), 17),
    ((128, 768), 32),
    ((4, 128, 2048), 64),
    ((1024, 4096), 128),
)
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
INPUT_DTYPE = torch.bfloat16
WEIGHT_DTYPE = torch.float16
BIAS_DTYPE = torch.float32


def make_inputs(shape, router_dtype, device, with_bias, num_experts):
    generator = torch.Generator().manual_seed(421)
    inp = torch.randn(shape, generator=generator)
    weight = torch.randn((num_experts, shape[-1]), generator=generator)
    bias = None
    if with_bias:
        bias = torch.randn(num_experts, generator=generator)
        # This first logit loses the 2**-14 residual when GEMM rounds before adding bias.
        first_row = inp.reshape(-1, shape[-1])[0]
        first_row.zero_()
        first_row[0] = 1 + 2**-7
        weight[0].zero_()
        weight[0, 0] = 1 + 2**-7
        bias[0] = -(1 + 2**-6)
        bias = bias.to(device=device, dtype=BIAS_DTYPE)
    return (
        inp.to(device=device, dtype=INPUT_DTYPE),
        weight.to(device=device, dtype=WEIGHT_DTYPE),
        bias,
        router_dtype,
    )


class _RouterGatingLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight, bias, router_dtype):
        ctx.save_for_backward(inp, weight, bias)
        ctx.router_dtype = router_dtype
        rows = inp.flatten(end_dim=-2).to(router_dtype)
        columns = weight.t().to(router_dtype)
        if bias is None:
            logits = torch.mm(rows, columns)
        else:
            logits = torch.addmm(bias.to(router_dtype), rows, columns)
        return logits.reshape(*inp.shape[:-1], weight.size(0))

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight, bias = ctx.saved_tensors
        router_dtype = ctx.router_dtype
        grad_rows = grad_output.reshape(-1, weight.size(0))
        input_rows = inp.reshape(-1, inp.size(-1)).to(router_dtype)
        grad_input = torch.mm(grad_rows, weight.to(router_dtype)).to(inp.dtype)
        grad_weight = torch.mm(grad_rows.t(), input_rows).to(weight.dtype)
        # Megatron rounds the bias gradient to weight.dtype before autograd
        # converts it to the original bias tensor's dtype.
        grad_bias = grad_rows.sum(dim=0).to(weight.dtype) if bias is not None else None
        return grad_input.reshape_as(inp), grad_weight, grad_bias, None


def reference(inp, weight, bias, router_dtype):
    return _RouterGatingLinear.apply(inp, weight, bias, router_dtype)
