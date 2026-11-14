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
"""Forward contract: dtype-rounded bias addition, layouts and special values.

The formula reference preserves the existing precision tolerances; a separate
case checks the installed Megatron forward. --ref cpu remains supported.
"""

import pytest
import torch

import flag_train

from .. import accuracy_utils as utils

pytestmark = pytest.mark.bias_gelu_impl
_SHAPES = [(), (0,), (0, 17), (3, 0), (1,), (17,), (2, 19, 7), (64, 128)]
if not utils.QUICK_MODE:
    _SHAPES += [
        (64, 64),
        (1024, 1024),
        (4096, 4096),
        (16384, 4096),
        (2048, 11008),
        (16, 128, 8192),
        (256, 32768),
        (2, 1024, 4097),
    ]


def _check(x, b):
    y = flag_train.megatron.bias_gelu_impl(x, b)
    assert y.shape == torch.broadcast_shapes(x.shape, b.shape)
    assert y.dtype == x.dtype and y.device == x.device
    assert y.is_contiguous()
    # Round the add before upcasting, even for the CPU reference.
    z = (utils.to_reference(x.detach()) + utils.to_reference(b.detach())).to(x.dtype)
    z = utils.to_reference(z, upcast=True)
    t = torch.tanh(0.79788456 * z * (1 + 0.044715 * z * z))
    ref = (0.5 * z * (1 + t)).to(x.dtype)
    tolerances = {
        torch.float16: (3e-3, 3e-3),
        torch.bfloat16: (2e-2, 2e-2),
        torch.float32: (2e-5, 2e-5),
    }
    rtol, atol = tolerances[x.dtype]
    torch.testing.assert_close(utils.to_reference(y), ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_gelu_impl(shape, dtype):
    utils.init_seed(0)
    x = torch.randn(shape, device=flag_train.device, dtype=dtype)
    b = torch.randn(shape[-1:] if shape else (), device=x.device, dtype=dtype)
    _check(x, b)


@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("bias_shape", [(), (7,), (1, 7), (2, 1, 7), (2, 3, 7)])
def test_bias_gelu_broadcast(dtype, bias_shape):
    utils.init_seed(1)
    x = torch.randn((2, 3, 7), device=flag_train.device, dtype=dtype)
    b = torch.randn(bias_shape, device=x.device, dtype=dtype)
    _check(x, b)
    # Broadcasting may expand the input as well as the bias.
    _check(b, x)


@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_gelu_noncontiguous(dtype):
    utils.init_seed(2)
    x = torch.randn((17, 9), device=flag_train.device, dtype=dtype).t()
    b = torch.randn((34,), device=x.device, dtype=dtype)[::2]
    assert not x.is_contiguous() and not b.is_contiguous()
    _check(x, b)
    _check(x[:, ::2], b[::2])


@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_gelu_values_and_empty(dtype):
    values = [-1000, -20, -8, -3, -1, -1e-4, 0, 1e-4, 1, 3, 8, 20, 1000]
    x = torch.tensor(values, device=flag_train.device, dtype=dtype)
    _check(x, torch.zeros_like(x))
    _check(
        torch.empty((0, 17), device=x.device, dtype=dtype),
        torch.ones((1, 17), device=x.device, dtype=dtype),
    )
    _check(
        torch.empty((2, 0, 7), device=x.device, dtype=dtype),
        torch.ones((), device=x.device, dtype=dtype),
    )


def test_bias_gelu_bad_arguments():
    x = torch.randn((4, 8), device=flag_train.device)
    for a, b in [(None, x), (x, None)]:
        with pytest.raises(TypeError, match="tensor"):
            flag_train.megatron.bias_gelu_impl(a, b)
    for dtype in [torch.int32, torch.float64, torch.bool]:
        # torch_npu silently casts requested FP64 to FP32. Keep a real FP64
        # operand on CPU: dtype validation deliberately precedes device checks.
        invalid = (
            x.cpu().to(dtype)
            if flag_train.vendor_name == "ascend" and dtype == torch.float64
            else x.to(dtype)
        )
        with pytest.raises(ValueError, match="dtype"):
            flag_train.megatron.bias_gelu_impl(x, invalid)
    with pytest.raises(ValueError, match="same dtype"):
        flag_train.megatron.bias_gelu_impl(x, x.to(torch.float16))
    for a, b in [(x, torch.randn(8)), (torch.randn(4, 8), x)]:
        with pytest.raises(ValueError, match="accelerator"):
            flag_train.megatron.bias_gelu_impl(a, b)
    with pytest.raises(ValueError, match="broadcast"):
        flag_train.megatron.bias_gelu_impl(x, x.new_empty(3))
    with pytest.raises(ValueError, match="broadcast"):
        flag_train.megatron.bias_gelu_impl(x.new_empty(0, 8), x.new_empty(3))


@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_gelu_nonfinite(dtype):
    x = torch.tensor(
        [float("inf"), -float("inf"), float("nan")],
        device=flag_train.device,
        dtype=dtype,
    )
    b = torch.zeros_like(x)
    y = flag_train.megatron.bias_gelu_impl(x, b)
    assert torch.isposinf(y[0])
    assert torch.isnan(y[1:]).all()


@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_gelu_installed_megatron(dtype):
    module = pytest.importorskip("megatron.core.fusions.fused_bias_gelu")
    utils.init_seed(3)
    x = torch.randn((16, 127), device=flag_train.device, dtype=dtype)
    b = torch.randn((127,), device=x.device, dtype=dtype)
    y = flag_train.megatron.bias_gelu_impl(x, b)
    ref = module.bias_gelu_impl(x, b)
    # Eager Megatron rounds intermediate low-precision arithmetic; the kernel
    # computes activation in fp32, so equality is tolerance based.
    rtol, atol = {
        torch.float16: (5e-3, 1e-2),
        torch.bfloat16: (4e-2, 8e-2),
        torch.float32: (2e-5, 2e-5),
    }[dtype]
    torch.testing.assert_close(y, ref, rtol=rtol, atol=atol)
