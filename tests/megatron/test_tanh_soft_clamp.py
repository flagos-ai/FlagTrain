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

import math
import warnings

import pytest
import torch

import flag_train

from .. import accuracy_utils as utils

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

_SCALES = [1.0, 0.5, 4.0, -2.5, 1e-3, 1e3]

_EXTENDED_DTYPES = [torch.int8, torch.int16, torch.int32, torch.uint8, torch.bool]
if utils.int64_is_supported:
    _EXTENDED_DTYPES.append(torch.int64)
_WIDE_UNSIGNED_DTYPES = [torch.uint16, torch.uint32, torch.uint64]


def tanh_soft_clamp_ref(x, scale):
    return (scale * torch.tanh(x.float() / scale)).to(x.dtype)


def base_op():
    try:
        import megatron.core.activations
    except ImportError:
        warnings.warn(
            "Megatron-LM is unavailable; using the PyTorch Tanh Soft Clamp "
            "composition as the correctness reference",
            RuntimeWarning,
            stacklevel=2,
        )
        return tanh_soft_clamp_ref

    compiled = getattr(megatron.core.activations, "tanh_soft_clamp", None)
    if compiled is None:
        warnings.warn(
            "Megatron-LM predates tanh_soft_clamp (needs megatron-core >= "
            "0.20.0); using the PyTorch Tanh Soft Clamp composition as the "
            "correctness reference",
            RuntimeWarning,
            stacklevel=2,
        )
        return tanh_soft_clamp_ref

    eager = getattr(compiled, "__wrapped__", compiled)
    chosen = compiled

    def reference(*args, **kwargs):
        nonlocal chosen
        if chosen is tanh_soft_clamp_ref:
            return chosen(*args, **kwargs)
        try:
            return chosen(*args, **kwargs)
        except Exception:
            if chosen is not eager:
                warnings.warn(
                    "Megatron tanh_soft_clamp failed to compile on this "
                    "platform; falling back to its eager implementation",
                    RuntimeWarning,
                    stacklevel=2,
                )
                chosen = eager
            else:
                warnings.warn(
                    "Megatron tanh_soft_clamp is unusable on this platform; "
                    "falling back to the PyTorch composition",
                    RuntimeWarning,
                    stacklevel=2,
                )
                chosen = tanh_soft_clamp_ref
            return chosen(*args, **kwargs)

    return reference


reference_tanh_soft_clamp = base_op()


def _reference(tensor):
    return utils.to_reference(tensor.detach(), upcast=True)


def _assert_close(out, ref_out, dtype, equal_nan=False):
    utils.train_assert_close(
        utils.to_reference(out), ref_out, dtype, equal_nan=equal_nan
    )


def _reference_same_dtype(x, scale):
    """Keep the input dtype so an integer cast happens at the same point.

    Integer/unsigned casts of NaN differ between compiled and eager Megatron
    and between CPU and device, so use the exact upstream expression there.
    """
    x_ref = utils.to_reference(x)
    if isinstance(scale, torch.Tensor):
        scale = utils.to_reference(scale)
    if not x.dtype.is_floating_point:
        return tanh_soft_clamp_ref(x_ref, scale)
    return reference_tanh_soft_clamp(x_ref, scale)


def _check_forward(x, scale, check_stride=False):
    dtype = x.dtype
    out = flag_train.megatron.tanh_soft_clamp(x, scale)
    assert out.shape == x.shape
    assert out.dtype == dtype
    if check_stride:
        assert out.stride() == x.stride()

    ref_out = reference_tanh_soft_clamp(_reference(x), scale)
    _assert_close(out, ref_out, dtype)
    return out


def _check_integral(x, scale, check_stride=False):
    """Compare integer/bool casts without upcasting the reference input."""
    dtype = x.dtype
    out = flag_train.megatron.tanh_soft_clamp(x, scale)
    assert out.shape == x.shape
    assert out.dtype == dtype
    assert out.device == x.device
    if check_stride:
        assert out.stride() == x.stride()

    _assert_close(out, _reference_same_dtype(x, scale), dtype)
    return out


def _integral_tensor(dtype, shape, device):
    n = 1
    for dim in shape:
        n *= dim
    values = torch.arange(n, device=device, dtype=torch.int64)
    if dtype in _WIDE_UNSIGNED_DTYPES or dtype is torch.uint8:
        pass
    elif dtype is torch.bool:
        values = values.remainder(2)
    else:
        values = values - n // 2
    if dtype is torch.bool:
        values = values.to(torch.bool)
    else:
        values = values.to(dtype)
    return values.view(shape)


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("scale", _SCALES)
def test_tanh_soft_clamp(shape, dtype, scale):
    utils.init_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    _check_forward(x, scale)


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("scale", _SCALES)
def test_tanh_soft_clamp_values(dtype, scale):
    """Zero, near-zero, curve landmarks, and both tanh saturation regions."""
    values = [
        -1e30,
        -50.0,
        -8.0,
        -1.0,
        -1e-3,
        -1e-6,
        0.0,
        1e-6,
        1e-3,
        1.0,
        8.0,
        50.0,
        1e30,
    ]
    x = torch.tensor(values, dtype=torch.float32, device=flag_train.device).to(dtype)
    _check_forward(x, scale)


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tanh_soft_clamp_noncontiguous(dtype):
    utils.init_seed(1)
    base = torch.randn((32, 64), dtype=dtype, device=flag_train.device)
    x = base.t()
    assert not x.is_contiguous()
    _check_forward(x, 0.5, check_stride=True)

    cube = torch.randn((4, 8, 16), dtype=dtype, device=flag_train.device)
    transposed = cube.transpose(0, 2)
    assert not transposed.is_contiguous()
    _check_forward(transposed, 0.5, check_stride=True)


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize("scale", [1.0, -2.0, 0.0])
def test_tanh_soft_clamp_special_values(scale):
    """inf/nan and division by scale == 0 must match the baseline."""
    values = [
        0.0,
        1.0,
        -1.0,
        1e30,
        -1e30,
        float("inf"),
        float("-inf"),
        float("nan"),
    ]
    x = torch.tensor(values, dtype=torch.float32, device=flag_train.device)
    out = flag_train.megatron.tanh_soft_clamp(x, scale)
    ref_out = reference_tanh_soft_clamp(_reference(x), scale)
    _assert_close(out, ref_out, torch.float32, equal_nan=True)


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize("dtype", _EXTENDED_DTYPES)
@pytest.mark.parametrize("scale", [0.5, 1e3, 0.0, -2.5, float("inf"), float("nan")])
def test_tanh_soft_clamp_extended_dtypes(dtype, scale):
    """Integer and bool tensors cast back to their input dtype, as upstream."""
    if utils.TO_CPU and (scale == 0.0 or not math.isfinite(scale)):
        pytest.skip("NaN-to-integer casts differ between CPU and device")
    x = _integral_tensor(dtype, (2, 3, 7), flag_train.device)
    _check_integral(x, scale)


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize("device", [flag_train.device, "cpu"])
@pytest.mark.parametrize("dtype", _WIDE_UNSIGNED_DTYPES)
@pytest.mark.parametrize("scale", [0.5, 1e3, 0.0, float("nan")])
def test_tanh_soft_clamp_wide_unsigned(dtype, device, scale):
    """The local comparator has no resolution entry for 16/32/64-bit unsigned."""
    if (
        utils.TO_CPU
        and torch.device(device).type != "cpu"
        and (scale == 0.0 or not math.isfinite(scale))
    ):
        pytest.skip("NaN-to-integer casts differ between CPU and device")
    x = _integral_tensor(dtype, (2, 3, 7), device)
    out = flag_train.megatron.tanh_soft_clamp(x, scale)
    ref_out = _reference_same_dtype(x, scale)
    assert out.shape == x.shape
    assert out.dtype == dtype
    assert out.device == x.device
    utils.train_assert_equal(out.to(torch.int64), ref_out.to(torch.int64))


_CPU_DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
    torch.float64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.bool,
]


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize("dtype", _CPU_DTYPES)
@pytest.mark.parametrize("scale", [0.5, 0.0, -2.5])
def test_tanh_soft_clamp_cpu(dtype, scale):
    """CPU tensors use the eager composition and stay on CPU."""
    if dtype.is_floating_point:
        x = torch.randn((2, 3, 7), dtype=dtype)
        out = _check_forward(x, scale)
    else:
        x = _integral_tensor(dtype, (2, 3, 7), "cpu")
        out = _check_integral(x, scale)
    assert out.device.type == "cpu"


@pytest.mark.tanh_soft_clamp
def test_tanh_soft_clamp_cpu_empty():
    x = torch.empty((0, 8), dtype=torch.float32, device="cpu")
    out = _check_forward(x, 0.5)
    assert out.device.type == "cpu"
    assert out.stride() == x.stride()


@pytest.mark.tanh_soft_clamp
def test_tanh_soft_clamp_cpu_noncontiguous():
    x = torch.randn((8, 16), device="cpu").t()
    assert not x.is_contiguous()
    out = _check_forward(x, 0.5, check_stride=True)
    assert out.device.type == "cpu"


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize("device", [flag_train.device, "cpu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_tanh_soft_clamp_tensor_scale(dtype, device):
    """A broadcastable tensor scale follows upstream's eager expression."""
    if dtype.is_floating_point:
        x = torch.randn((2, 1), dtype=dtype, device=device)
    else:
        x = torch.tensor([[1], [2]], dtype=dtype, device=device)
    scale = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32, device=device)
    out = flag_train.megatron.tanh_soft_clamp(x, scale)
    ref_out = _reference_same_dtype(x, scale)
    assert out.shape == (2, 3)
    assert out.dtype == dtype
    assert out.device == x.device
    utils.train_assert_close(
        utils.to_reference(out), utils.to_reference(ref_out), dtype
    )


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize("device", [flag_train.device, "cpu"])
def test_tanh_soft_clamp_scalar_tensor_scale(device):
    x = torch.randn((2, 3), dtype=torch.float32, device=device)
    scale = torch.tensor(0.5, dtype=torch.float32, device=device)
    out = flag_train.megatron.tanh_soft_clamp(x, scale)
    ref_out = _reference_same_dtype(x, scale)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device
    utils.train_assert_close(
        utils.to_reference(out), utils.to_reference(ref_out), x.dtype
    )


@pytest.mark.tanh_soft_clamp
@pytest.mark.skipif(not utils.fp64_is_supported, reason="device has no fp64")
def test_tanh_soft_clamp_float64():
    x = torch.randn((2, 3, 7), dtype=torch.float64, device=flag_train.device)
    _check_forward(x, 0.5)


_FLOAT8_DTYPES = [
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    )
    if dtype is not None
]


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize("device", [flag_train.device, "cpu"])
@pytest.mark.parametrize("dtype", _FLOAT8_DTYPES)
def test_tanh_soft_clamp_float8(dtype, device):
    values = [-100.0, -10.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 10.0, 100.0]
    x = torch.tensor(values, device=device).to(dtype)
    for scale in (0.5, 1e3):
        out = flag_train.megatron.tanh_soft_clamp(x, scale)
        ref_out = _reference_same_dtype(x, scale)
        assert out.shape == x.shape
        assert out.dtype == dtype
        assert out.device == x.device
        utils.train_assert_equal(out.float(), ref_out.float())


@pytest.mark.tanh_soft_clamp
def test_tanh_soft_clamp_forward_only():
    x = torch.randn((2, 3), device=flag_train.device, requires_grad=True)
    assert not flag_train.megatron.tanh_soft_clamp(x, 0.5).requires_grad
    x_cpu = torch.randn((2, 3), requires_grad=True)
    assert not flag_train.megatron.tanh_soft_clamp(x_cpu, 0.5).requires_grad
    scale_cpu = torch.tensor(0.5, requires_grad=True)
    assert not flag_train.megatron.tanh_soft_clamp(x_cpu, scale_cpu).requires_grad


@pytest.mark.tanh_soft_clamp
def test_tanh_soft_clamp_bad_arguments():
    with pytest.raises(TypeError, match="tensor"):
        flag_train.megatron.tanh_soft_clamp([1.0], 0.5)
    with pytest.raises(TypeError, match="complex"):
        flag_train.megatron.tanh_soft_clamp(
            torch.tensor([1 + 0j], device=flag_train.device), 0.5
        )
