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
"""Correctness tests for Megatron Quick GELU.

The reference is the formula in Megatron-LM ``megatron/core/activations.py``:

    quick_gelu(x) = x * sigmoid(1.702 * x)

It is evaluated with PyTorch, independently of the Triton kernels. Under
``--ref cpu`` the reference operands are moved to the CPU and upcast, so the
comparison is against a higher-precision CPU result rather than a second run
of the device kernel. Tolerances are ``train_assert_close``'s defaults
(``atol=1e-4``, dtype rtol from ``flag_train.testing.RESOLUTION``).
"""

import pytest
import torch

import flag_train
from flag_train.megatron import quick_gelu

from .. import accuracy_utils as utils

# Megatron-LM coefficient. Kept here, not imported from the kernel, so a wrong
# constant in the implementation cannot satisfy this reference.
_ALPHA = 1.702

# Scalar, empty, unaligned, and the ranks pointwise_dynamic specializes on.
_SHAPES = [
    (),
    (0,),
    (0, 8),
    (1,),
    (17,),
    (1024,),
    (64, 128),
    (2, 19, 7),
    (4, 8, 32, 16),
]


def quick_gelu_ref(x):
    """Megatron Quick GELU, composed from PyTorch ops."""
    return x * torch.sigmoid(_ALPHA * x)


def _reference(tensor):
    return utils.to_reference(tensor.detach(), upcast=True)


def _assert_close(actual, expected, dtype):
    utils.train_assert_close(utils.to_reference(actual), expected, dtype)


def _check_forward_and_backward(x):
    dtype = x.dtype
    y = quick_gelu(x)
    assert y.shape == x.shape
    assert y.dtype == dtype

    x_ref = _reference(x)
    _assert_close(y, quick_gelu_ref(x_ref), dtype)

    grad_out = torch.randn_like(y)
    x_train = x.detach().requires_grad_(True)
    quick_gelu(x_train).backward(grad_out)

    x_ref = _reference(x).requires_grad_(True)
    quick_gelu_ref(x_ref).backward(_reference(grad_out))
    _assert_close(x_train.grad, x_ref.grad, dtype)


@pytest.mark.quick_gelu
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_quick_gelu(shape, dtype):
    utils.init_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    _check_forward_and_backward(x)


@pytest.mark.quick_gelu
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_quick_gelu_values(dtype):
    """Positive, negative, and near-zero inputs, plus a few large magnitudes."""
    values = [
        -50.0,
        -8.0,
        -1.702,
        -1.0,
        -1e-3,
        -1e-6,
        0.0,
        1e-6,
        1e-3,
        0.5,
        1.0,
        1.702,
        8.0,
        50.0,
    ]
    x = torch.tensor(values, dtype=torch.float32, device=flag_train.device).to(dtype)
    _check_forward_and_backward(x)


@pytest.mark.quick_gelu
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_quick_gelu_noncontiguous(dtype):
    utils.init_seed(1)
    base = torch.randn((32, 64), dtype=dtype, device=flag_train.device)
    x = base.t()
    assert not x.is_contiguous()
    _check_forward_and_backward(x)

    cube = torch.randn((4, 8, 16), dtype=dtype, device=flag_train.device)
    transposed = cube.transpose(0, 2)
    assert not transposed.is_contiguous()
    _check_forward_and_backward(transposed)
