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


"""Check the explicit TransformerEngine L2 backward interface."""

import pytest
import torch

import flag_train
from flag_train.reference_candidate import get_candidate
from tests.transformer_engine._l2_backward_reference import (
    DTYPES,
    OP_NAME,
    SHAPES,
    make_inputs,
    reference,
)


@pytest.mark.l2_normalization_backward
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_l2_normalization_backward(shape, dtype):
    candidate = get_candidate(f"transformer_engine.{OP_NAME}", reference=reference)
    inputs = make_inputs(shape, dtype, flag_train.device)
    expected = reference(*inputs)
    actual = candidate(*inputs)
    assert actual.device == inputs[1].device
    torch.testing.assert_close(actual, expected)
