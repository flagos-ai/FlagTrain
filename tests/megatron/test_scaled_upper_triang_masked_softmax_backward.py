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
import megatron.core.fusions.fused_softmax
import pytest
import torch

import flag_train

from .. import accuracy_utils as utils

# As in the forward suite: widths from one warp to a row far past the widest
# tile, including the odd sizes that leave a row guard on the last program.
_SHAPES = [
    (16, 16, 16),
    (16, 32, 32),
    (8, 33, 33),
    (8, 64, 64),
    (16, 96, 96),
    (32, 128, 128),
    (8, 256, 256),
    (16, 512, 512),
    (8, 1024, 1024),
    (8, 2048, 2048),
    (8, 4096, 4096),
    (8, 4160, 4160),
    (8, 8192, 8192),
]

# Megatron's CUDA fused kernel rejects fp32.
_DTYPES = [torch.float16, torch.bfloat16]

# On Ascend the Megatron reference calls torch_npu.npu_scaled_masked_softmax,
# which supports fp32.
if flag_train.vendor_name == "ascend":
    _DTYPES.append(torch.float32)

_SCALES = [0.0, 0.125, 1.0, -0.5, 2]


@pytest.mark.scaled_upper_triang_masked_softmax_backward
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("scale", _SCALES)
def test_scaled_upper_triang_masked_softmax_backward(shape, dtype, scale):
    utils.init_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device).requires_grad_(True)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_train.device)

    output = megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax.apply(
        x, scale
    )
    (ref,) = torch.autograd.grad(output, x, grad_output.clone())
    res = flag_train.megatron.scaled_upper_triang_masked_softmax_backward(
        grad_output, output.detach(), scale
    )

    # Megatron needs device inputs and returns device tensors; only the
    # reference follows ``--ref cpu``.
    if utils.TO_CPU:
        ref = utils.to_reference(ref)

    utils.train_assert_close(res, ref, dtype)


@pytest.mark.scaled_upper_triang_masked_softmax_backward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_scaled_upper_triang_masked_softmax_backward_zero_grad(dtype):
    """A zero upstream gradient must give a zero gradient, masked half included."""
    shape = (16, 128, 128)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device).requires_grad_(True)
    grad_output = torch.zeros(shape, dtype=dtype, device=flag_train.device)

    output = megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax.apply(
        x, 0.125
    )
    (ref,) = torch.autograd.grad(output, x, grad_output.clone())
    res = flag_train.megatron.scaled_upper_triang_masked_softmax_backward(
        grad_output, output.detach(), 0.125
    )

    if utils.TO_CPU:
        ref = utils.to_reference(ref)

    utils.train_assert_close(res, ref, dtype)


@pytest.mark.scaled_upper_triang_masked_softmax_backward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_scaled_upper_triang_masked_softmax_backward_non_contiguous(dtype):
    utils.init_seed(2)

    # The forward runs on a contiguous copy; the operator is handed strided
    # views of both of its arguments, the way a downstream gradient arrives.
    x = torch.randn((16, 32, 64), dtype=dtype, device=flag_train.device)[..., ::2]
    x = x.contiguous().requires_grad_(True)

    wide_grads = torch.randn((16, 32, 64), dtype=dtype, device=flag_train.device)
    grad_output = wide_grads[..., ::2]

    output = megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax.apply(
        x, 0.125
    )

    wide_output = torch.zeros((16, 32, 64), dtype=dtype, device=flag_train.device)
    wide_output[..., ::2] = output
    strided_output = wide_output[..., ::2]

    assert not grad_output.is_contiguous()
    assert not strided_output.is_contiguous()

    (ref,) = torch.autograd.grad(output, x, grad_output.clone())
    res = flag_train.megatron.scaled_upper_triang_masked_softmax_backward(
        grad_output, strided_output, 0.125
    )

    if utils.TO_CPU:
        ref = utils.to_reference(ref)

    utils.train_assert_close(res, ref, dtype)
