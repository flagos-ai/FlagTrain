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
"""Forward contract: Megatron dtype promotion, layouts and special values.

Compare the installed Megatron forward and an independent formula at
reference precision after applying Megatron's promoted bias addition.
"""

from functools import cache

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


@cache
def _load_megatron():
    try:
        from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "megatron",
            "megatron.core",
            "megatron.core.fusions",
            "megatron.core.fusions.fused_bias_gelu",
        }:
            raise
        return None
    return bias_gelu_impl


def _check(x, b):
    y = flag_train.megatron.bias_gelu_impl(x, b)
    assert y.shape == torch.broadcast_shapes(x.shape, b.shape)
    assert y.dtype == x.dtype and y.device == x.device
    assert y.is_contiguous()

    # Apply Megatron's promoted bias add, then evaluate GELU in reference
    # precision on the device selected by --ref.
    z = utils.to_reference(x.detach()) + utils.to_reference(b.detach())
    z = utils.to_reference(z, upcast=True)
    result = utils.to_reference(y)

    megatron_impl = _load_megatron()
    if megatron_impl is not None:
        # Bias is already included in z, so use a scalar zero.
        utils.train_assert_close(result, megatron_impl(z, z.new_zeros(())), x.dtype)

    t = torch.tanh(0.79788456 * z * (1 + 0.044715 * z * z))
    ref = 0.5 * z * (1 + t)
    utils.train_assert_close(result, ref, x.dtype)


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
    for a, b in [(x, torch.randn(8)), (torch.randn(4, 8), x)]:
        with pytest.raises(ValueError, match="same device"):
            flag_train.megatron.bias_gelu_impl(a, b)
    with pytest.raises(ValueError, match="unsupported"):
        flag_train.megatron.bias_gelu_impl(x.to(torch.complex64), x.to(torch.complex64))
    with pytest.raises(ValueError, match="unsupported"):
        flag_train.megatron.bias_gelu_impl(x, 1 + 2j)
    with pytest.raises(TypeError, match="one of input and bias"):
        flag_train.megatron.bias_gelu_impl(1.0, 2.0)
    with pytest.raises(ValueError, match="broadcast"):
        flag_train.megatron.bias_gelu_impl(x, x.new_empty(3))
    with pytest.raises(ValueError, match="broadcast"):
        flag_train.megatron.bias_gelu_impl(x.new_empty(0, 8), x.new_empty(3))


_EXTENDED_DTYPES = [
    torch.int8,
    torch.int16,
    torch.int32,
    torch.uint8,
    torch.bool,
]
if utils.int64_is_supported:
    _EXTENDED_DTYPES.append(torch.int64)
if utils.fp64_is_supported:
    _EXTENDED_DTYPES.append(torch.float64)


def _random_operand(dtype, shape):
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, device=flag_train.device, dtype=dtype)
    if dtype.is_floating_point:
        return torch.randn(shape, device=flag_train.device, dtype=dtype)
    return torch.randint(0, 4, shape, device=flag_train.device, dtype=dtype)


def _is_bool_operand(value):
    """Whether ``value`` takes part in a bool + bool sum."""
    return isinstance(value, bool) or (
        torch.is_tensor(value) and value.dtype == torch.bool
    )


def _assert_matches_megatron(x, b):
    megatron_impl = _load_megatron()
    if megatron_impl is None:
        pytest.skip("Megatron bias_gelu_impl is unavailable")
    y = flag_train.megatron.bias_gelu_impl(x, b)
    if _is_bool_operand(x) and _is_bool_operand(b):
        # torch defines bool + bool as logical or, which is what the kernel
        # implements; torch-npu adds bool tensors as integers instead
        # (True + True == 2), so the reference is evaluated where Megatron's
        # own semantics hold and the result is moved back for the checks below.
        expected = megatron_impl(x.cpu(), b.cpu() if torch.is_tensor(b) else b).to(
            y.device
        )
    else:
        expected = megatron_impl(x, b)
    assert y.shape == expected.shape
    assert y.dtype == expected.dtype
    assert y.device == expected.device
    atol = {
        torch.float16: 2e-3,
        torch.bfloat16: 3.2e-2,
        torch.float64: 1e-6,
    }.get(y.dtype, 1e-3)
    utils.train_assert_close(
        utils.to_reference(y), utils.to_reference(expected), y.dtype, atol=atol
    )
    return y, expected


@pytest.mark.parametrize("dtype", _EXTENDED_DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_bias_gelu_extended_dtypes(dtype):
    """Megatron promotes integer/bool sums to FP32."""
    if dtype is torch.bool:
        x = torch.randint(0, 2, (4, 8), device=flag_train.device, dtype=dtype)
        b = torch.randint(0, 2, (8,), device=flag_train.device, dtype=dtype)
    elif dtype.is_floating_point:
        x = torch.randn((4, 8), device=flag_train.device, dtype=dtype)
        b = torch.randn((8,), device=flag_train.device, dtype=dtype)
    else:
        x = torch.randint(0, 4, (4, 8), device=flag_train.device, dtype=dtype)
        b = torch.randint(0, 4, (8,), device=flag_train.device, dtype=dtype)
    _assert_matches_megatron(x, b)


@pytest.mark.parametrize(
    "dtype,values",
    [
        (torch.uint8, [0, 15, 16, 20, 50, 200, 255]),
        (torch.int8, [-128, -12, 0, 12, 20, 100, 127]),
        (torch.int16, [-30000, -200, 0, 200, 1000, 30000]),
        (torch.int32, [-2000000, -50000, 0, 50000, 100000, 2000000]),
    ]
    + (
        [(torch.int64, [-4000000000, 0, 4000000000])]
        if utils.int64_is_supported
        else []
    ),
    ids=lambda value: (
        str(value).split(".")[-1] if isinstance(value, torch.dtype) else ""
    ),
)
def test_bias_gelu_integral_edges(dtype, values):
    """The GELU square is promoted to FP32 before it can overflow."""
    x = torch.tensor(values, device=flag_train.device, dtype=dtype)
    _assert_matches_megatron(x, torch.zeros((), device=x.device, dtype=dtype))


@pytest.mark.parametrize(
    "dtype,input, bias",
    [
        (torch.uint8, [200, 100, 255, 128], [100, 200, 1, 128]),
        (torch.int8, [120, 100, 127, -128], [10, 100, 1, -1]),
        (torch.int16, [32000, 30000, 32767, -32768], [1000, 3000, 1, -1]),
    ],
)
def test_bias_gelu_integral_add_wraps(dtype, input, bias):
    """The bias add keeps the promoted integer dtype and wraps."""
    x = torch.tensor(input, device=flag_train.device, dtype=dtype)
    b = torch.tensor(bias, device=flag_train.device, dtype=dtype)
    _assert_matches_megatron(x, b)


@pytest.mark.parametrize(
    "input_dtype, bias_dtype",
    [
        (torch.float16, torch.float32),
        (torch.float32, torch.float16),
        (torch.float16, torch.bfloat16),
        (torch.int32, torch.float16),
        (torch.int32, torch.float32),
        (torch.bool, torch.float32),
        (torch.bool, torch.int32),
        (torch.int64, torch.int32),
        (torch.uint8, torch.int32),
    ],
    ids=lambda value: str(value).split(".")[-1],
)
def test_bias_gelu_dtype_promotion(input_dtype, bias_dtype):
    """The result dtype follows Megatron/torch promotion, not the input dtype."""
    if torch.int64 in (input_dtype, bias_dtype) and not utils.int64_is_supported:
        pytest.skip("device does not support int64")

    def make(dtype, shape):
        if dtype is torch.bool:
            return torch.randint(0, 2, shape, device=flag_train.device, dtype=dtype)
        if dtype.is_floating_point:
            return torch.randn(shape, device=flag_train.device, dtype=dtype)
        return torch.randint(0, 4, shape, device=flag_train.device, dtype=dtype)

    _assert_matches_megatron(make(input_dtype, (4, 8)), make(bias_dtype, (8,)))


def test_bias_gelu_cpu_input():
    """CPU operands are staged through the accelerator and returned on CPU."""
    megatron_impl = _load_megatron()
    if megatron_impl is None:
        pytest.skip("Megatron bias_gelu_impl is unavailable")
    x = torch.randn((4, 8), device="cpu", dtype=torch.float32)
    b = torch.randn((8,), device="cpu", dtype=torch.float32)
    y = flag_train.megatron.bias_gelu_impl(x, b)
    expected = megatron_impl(x, b)
    assert y.device.type == "cpu"
    utils.train_assert_close(y, expected, y.dtype)


@pytest.mark.parametrize("scalar", [1.0, 1, True, 300, -1])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.int8, torch.uint8, torch.int32]
)
def test_bias_gelu_scalar_operand(dtype, scalar):
    """Megatron promotes one Python-number operand with the tensor operand."""
    x = _random_operand(dtype, (4, 8))
    _assert_matches_megatron(x, scalar)
    _assert_matches_megatron(scalar, x)


def test_bias_gelu_cpu_scalar_tensor():
    """A 0-dim CPU tensor pairs with an accelerator tensor, as torch does."""
    megatron_impl = _load_megatron()
    if megatron_impl is None:
        pytest.skip("Megatron bias_gelu_impl is unavailable")
    b = torch.randn((8,), device=flag_train.device, dtype=torch.float16)
    for x in (
        torch.tensor(1.0, dtype=torch.float16),
        torch.tensor(1.0, device=b.device, dtype=torch.float16),
    ):
        y = flag_train.megatron.bias_gelu_impl(x, b)
        expected = megatron_impl(x, b)
        assert y.device == b.device
        utils.train_assert_close(
            utils.to_reference(y), utils.to_reference(expected), y.dtype, atol=2e-3
        )
        y = flag_train.megatron.bias_gelu_impl(b, x)
        expected = megatron_impl(b, x)
        assert y.device == b.device
        utils.train_assert_close(
            utils.to_reference(y), utils.to_reference(expected), y.dtype, atol=2e-3
        )


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
    if _load_megatron() is None:
        pytest.skip("Megatron bias_gelu_impl is unavailable")
    utils.init_seed(3)
    x = torch.randn((16, 127), device=flag_train.device, dtype=dtype)
    b = torch.randn((127,), device=x.device, dtype=dtype)
    _check(x, b)
