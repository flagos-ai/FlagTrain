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

"""Check Megatron's Torch router branch, including the mixed-dtype bias epilogue."""

import pytest
import torch

import flag_train
from flag_train.reference_candidate import get_candidate
from tests.megatron._router_gating_linear_reference import (
    DTYPES,
    OVERRIDE_KEY,
    WORKLOADS,
    make_inputs,
    reference,
)

pytestmark = pytest.mark.skipif(
    flag_train.vendor_name != "nvidia", reason="NVIDIA Router kernel"
)


@pytest.mark.router_gating_linear_function
@pytest.mark.parametrize(
    "shape,num_experts",
    WORKLOADS,
    ids=[f"shape{index}" for index in range(len(WORKLOADS))],
)
@pytest.mark.parametrize("router_dtype", DTYPES)
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "bias"])
def test_router_gating_linear_function(shape, num_experts, router_dtype, with_bias):
    candidate = get_candidate(OVERRIDE_KEY, reference=reference)

    inp, weight, bias, _ = make_inputs(
        shape, router_dtype, flag_train.device, with_bias, num_experts
    )
    inputs = (inp, weight, bias) if with_bias else (inp, weight)
    candidate_inputs = tuple(tensor.clone().requires_grad_() for tensor in inputs)
    reference_inputs = tuple(tensor.clone().requires_grad_() for tensor in inputs)
    candidate_bias = candidate_inputs[2] if with_bias else None
    reference_bias = reference_inputs[2] if with_bias else None
    actual = candidate(*candidate_inputs[:2], candidate_bias, router_dtype)
    expected = reference(*reference_inputs[:2], reference_bias, router_dtype)
    torch.testing.assert_close(actual, expected)
    if with_bias:
        torch.testing.assert_close(
            actual.flatten()[0], expected.flatten()[0], rtol=0, atol=0
        )
    for actual_input, original in zip(candidate_inputs, inputs):
        torch.testing.assert_close(actual_input, original, rtol=0, atol=0)
        assert (
            actual.untyped_storage().data_ptr()
            != actual_input.untyped_storage().data_ptr()
        )

    grad_output = (
        torch.linspace(-0.8, 0.9, expected.numel()).reshape(expected.shape).to(expected)
    )
    actual_grads = torch.autograd.grad(actual, candidate_inputs, grad_output.clone())
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad_output)
    torch.testing.assert_close(actual_grads, expected_grads)
    if with_bias and router_dtype == torch.float32:
        torch.testing.assert_close(actual_grads[-1], expected_grads[-1], rtol=0, atol=0)
    for actual_input, original in zip(candidate_inputs, inputs):
        torch.testing.assert_close(actual_input, original, rtol=0, atol=0)


def test_router_gating_linear_noncontiguous():
    def strided(tensor):
        return tensor.repeat_interleave(2, dim=-1)[..., ::2]

    inp, weight, bias, dtype = make_inputs(
        (2, 5, 32), torch.float32, flag_train.device, True, 13
    )
    inputs = tuple(strided(tensor).requires_grad_() for tensor in (inp, weight, bias))
    actual = flag_train.megatron.router_gating_linear_function(*inputs, dtype)
    expected = reference(*inputs, dtype)
    torch.testing.assert_close(actual, expected)
    grad = strided(torch.randn_like(expected))
    torch.testing.assert_close(
        torch.autograd.grad(actual, inputs, grad),
        torch.autograd.grad(expected, inputs, grad),
    )


def test_router_gating_linear_empty_reduction_with_bias():
    device = flag_train.device
    inp = torch.empty((2, 0), device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.empty((3, 0), device=device, dtype=torch.float16, requires_grad=True)
    bias = torch.tensor([0.1, 0.2, 0.4], device=device, requires_grad=True)
    actual = flag_train.megatron.router_gating_linear_function(
        inp, weight, bias, torch.float32
    )
    torch.testing.assert_close(actual, bias.expand(2, 3))
    grad = torch.tensor([[0.2, 0.4, 0.7], [0.15, 0.25, 0.35]], device=device)
    torch.testing.assert_close(
        torch.autograd.grad(actual, (inp, weight, bias), grad),
        (torch.zeros_like(inp), torch.zeros_like(weight), grad.sum(0).half().float()),
    )
