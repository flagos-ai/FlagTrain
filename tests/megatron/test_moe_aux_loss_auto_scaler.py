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


import pytest
import torch

import flag_train
from flag_train.testing import assert_equal


@pytest.mark.moe_aux_loss_auto_scaler
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["default", "scalar", "updated_tensor"])
def test_moe_aux_loss_auto_scaler(dtype, mode):
    device = flag_train.device
    values = torch.arange(12, device=device, dtype=torch.float32)
    output = values.to(dtype).reshape(3, 4).t().requires_grad_()
    aux_loss = torch.ones((2, 1), device=device, dtype=dtype, requires_grad=True)
    if mode == "default":
        scale, expected_scale = None, 1.0
    elif mode == "scalar":
        scale, expected_scale = 2.5, 2.5
    else:
        scale = torch.tensor(1.0, device=device)
        expected_scale = 3.0

    actual = flag_train.megatron.moe_aux_loss_auto_scaler(output, aux_loss, scale)

    assert actual.data_ptr() == output.data_ptr()
    assert actual.stride() == output.stride()
    assert_equal(actual, output)
    if mode == "updated_tensor":
        scale.fill_(expected_scale)
    grad = values.to(dtype).reshape(actual.shape)
    actual.backward(grad)
    assert_equal(output.grad, grad)
    assert_equal(aux_loss.grad, torch.full_like(aux_loss, expected_scale))
