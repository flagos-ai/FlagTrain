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


"""Check explicit Megatron non-interleaved mRoPE inputs and layout."""

import pytest
import torch

import flag_train
from flag_train.reference_candidate import get_candidate
from tests.megatron._multi_modal_rotary_reference import (
    CORRECTNESS_WORKLOADS,
    OVERRIDE_KEY,
    WORKLOADS,
    make_inputs,
    reference,
)


@pytest.mark.multi_modal_rotary_embedding
@pytest.mark.parametrize("workload", CORRECTNESS_WORKLOADS)
def test_multi_modal_rotary_embedding(workload):
    candidate = get_candidate(OVERRIDE_KEY, reference=reference)

    position_ids, inv_freq, sections, interpolation = make_inputs(
        workload, flag_train.device
    )
    original_ids, original_freqs = position_ids.clone(), inv_freq.clone()
    expected = reference(original_ids, original_freqs, sections, interpolation)
    actual = candidate(position_ids, inv_freq, sections, interpolation)

    assert actual.shape == (workload.length, workload.batch, 1, 2 * inv_freq.numel())
    assert actual.dtype == torch.float32
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(position_ids, original_ids, rtol=0, atol=0)
    torch.testing.assert_close(inv_freq, original_freqs, rtol=0, atol=0)
    actual_storage = actual.untyped_storage().data_ptr()
    assert actual_storage != position_ids.untyped_storage().data_ptr()
    assert actual_storage != inv_freq.untyped_storage().data_ptr()

    if workload == WORKLOADS[0]:
        first_position = actual.new_tensor([1, 20, 300, 400, 1, 20, 300, 400])
        torch.testing.assert_close(actual[0, 0, 0], first_position, rtol=0, atol=0)
