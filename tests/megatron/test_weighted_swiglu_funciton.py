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


"""Check default WeightedSwiGLU forward and both first-order gradients."""

import pytest
import torch

import flag_train
from flag_train.reference_candidate import get_candidate
from tests.megatron._weighted_swiglu_reference import (
    DTYPES,
    OVERRIDE_KEY,
    SHAPES,
    make_inputs,
    reference,
)


@pytest.mark.weighted_swiglu_funciton
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_weighted_swiglu_funciton(shape, dtype):
    candidate = get_candidate(OVERRIDE_KEY, reference=reference)

    inputs = make_inputs(shape, dtype, flag_train.device)
    candidate_inputs = tuple(tensor.clone().requires_grad_() for tensor in inputs)
    reference_inputs = tuple(tensor.clone().requires_grad_() for tensor in inputs)
    actual = candidate(*candidate_inputs)
    expected = reference(*reference_inputs)
    torch.testing.assert_close(actual, expected)
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
    for actual_input, original in zip(candidate_inputs, inputs):
        torch.testing.assert_close(actual_input, original, rtol=0, atol=0)
