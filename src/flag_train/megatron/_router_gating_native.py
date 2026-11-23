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

"""Fuse conversion of mixed-precision router operands before vendor GEMM."""

from math import prod

import torch
import triton
import triton.language as tl

from flag_train.runtime import torch_device_fn


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


def router_gating_linear_function(inp, weight, bias, router_dtype):
    validate_router_inputs(inp, weight, bias, router_dtype)
    return _Router.apply(inp, weight, bias, router_dtype)
