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
"""Correctness tests for fused bias + GeGLU (forward only).

The reference is Megatron-LM's own ``bias_geglu_impl`` when megatron-core is
installed. Otherwise, ``bias_geglu_ref`` evaluates the eager tanh-GELU formula
with the bias added by ordinary broadcasting.

Under ``--ref cpu`` the reference operands are moved to the CPU; floating
operands are upcast for a higher-precision result, while integer/bool
operands keep their dtype so wrap/OR semantics are preserved. Tolerances are
``train_assert_close``'s defaults.

The operator follows Megatron's forward interface for the real dtypes the
Triton kernel can load and store: rank 2/3, optional bias with torch
broadcasting and dtype promotion, integer/bool inputs, CPU staging through the
accelerator, and the non-contiguous layouts Megatron's ``view`` accepts.
Complex and 16/32/64-bit unsigned operands are rejected. Only the forward
output is checked; autograd/backward is out of scope.
"""

import importlib

import pytest
import torch

import flag_train

from .. import accuracy_utils as utils


def _accelerator_available():
    if flag_train.runtime.device.name == "cpu":
        return False
    available = getattr(flag_train.runtime.torch_device_fn, "is_available", None)
    return True if not callable(available) else bool(available())


pytestmark = pytest.mark.skipif(
    not _accelerator_available(),
    reason="bias_geglu needs a usable accelerator for its Triton kernel",
)


def _load_megatron_impl():
    """Installed ``bias_geglu_impl``, or ``None`` when it is unavailable.

    Older Megatron installs may not provide the fusions module or this public
    function; those installations use the formula reference. Other import
    failures still raise instead of silently hiding a broken installation.
    """
    try:
        importlib_metadata_version = __import__(
            "importlib.metadata", fromlist=["version"]
        )
    except ImportError:
        return None
    try:
        importlib_metadata_version.version("megatron-core")
    except Exception:
        return None
    try:
        from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
    except ModuleNotFoundError as exc:
        if exc.name in {
            "megatron.core.fusions",
            "megatron.core.fusions.fused_bias_geglu",
        }:
            return None
        raise RuntimeError(
            "megatron-core is installed but importing bias_geglu_impl failed: "
            f"{exc!r}"
        ) from exc
    except ImportError as exc:
        if "cannot import name 'bias_geglu_impl'" in str(exc):
            return None
        raise RuntimeError(
            "megatron-core is installed but importing bias_geglu_impl failed: "
            f"{exc!r}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "megatron-core is installed but importing bias_geglu_impl failed: "
            f"{exc!r}"
        ) from exc
    return bias_geglu_impl


_megatron_impl = _load_megatron_impl()

# Empty, unaligned (half = 3 and 17), and MLP-like ranks. The last dim is even.
_SHAPES = [
    (0, 8),
    (0, 4, 16),
    (1, 2),
    (8, 6),
    (5, 34),
    (17, 32),
    (64, 128),
    (2, 8, 64),
    (4, 7, 32),
]

# Bias is orthogonal to the shapes above, so it is sampled on representative
# cases only.
_BIAS_SHAPES = [
    (1, 2),
    (17, 32),
    (64, 128),
    (4, 7, 32),
]

# Dtypes the Triton kernel can load and store, and the ones it cannot.
_INTEGRAL_DTYPES = [
    dtype
    for dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        getattr(torch, "uint8", None),
        torch.bool,
    )
    if dtype is not None
]
_UNSIGNED_DTYPES = tuple(
    dtype for dtype in (getattr(torch, "uint8", None),) if dtype is not None
)
_UNSUPPORTED_DTYPES = [
    dtype
    for dtype in (
        getattr(torch, "uint16", None),
        getattr(torch, "uint32", None),
        getattr(torch, "uint64", None),
    )
    if dtype is not None
]
_SUPPORTED_TEST_DTYPES = (
    list(utils.FLOAT_DTYPES)
    + _INTEGRAL_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
)
_MIXED_DTYPES = [
    (torch.float16, torch.float32),
    (torch.float16, torch.bfloat16),
    (torch.float32, torch.float16),
    (torch.int32, torch.float16),
    (torch.int64, torch.float32),
    (torch.int32, torch.bool),
    (torch.bool, torch.float16),
    (getattr(torch, "uint8", None), torch.int8),
    (torch.bool, getattr(torch, "uint8", None)),
]
_MIXED_DTYPES = [pair for pair in _MIXED_DTYPES if None not in pair]
if utils.fp64_is_supported:
    _MIXED_DTYPES.append((torch.float32, torch.float64))


def _fp64_dtype_supported(dtype):
    if dtype is torch.float64:
        return utils.fp64_is_supported or flag_train.device == "cpu"
    return True


def bias_geglu_ref(x, bias=None):
    y = x if bias is None else x + bias
    gate, linear = torch.chunk(y, 2, dim=-1)
    # tanh approximation of GELU, the same constants as Megatron-LM.
    return (
        gate
        * 0.5
        * (1.0 + torch.tanh(0.79788456 * gate * (1 + 0.044715 * gate * gate)))
    ) * linear


def _reference(tensor):
    if tensor is None:
        return None
    if tensor.dtype.is_floating_point:
        return utils.to_reference(tensor.detach(), upcast=True)
    # Integers and bool must keep their dtype: upcasting changes integer wrap
    # and turns bool ``+`` from OR into arithmetic addition.
    return utils.to_reference(tensor.detach(), upcast=False)


def _assert_close(actual, expected, dtype):
    utils.train_assert_close(utils.to_reference(actual), expected, dtype)


def _expected_dtype(x, bias):
    """Output dtype of Megatron's expression for these operands."""
    dtype = x.dtype if bias is None else torch.promote_types(x.dtype, bias.dtype)
    if dtype.is_floating_point:
        return dtype
    return torch.get_default_dtype()


def _dtype_tensor(dtype, shape, device=None):
    """A small-valued tensor of ``dtype`` for the dtype-support matrix."""
    device = device or flag_train.device
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, device=device).to(dtype)
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype in _UNSIGNED_DTYPES:
        return torch.randint(0, 5, shape, device=device).to(dtype)
    return torch.randint(-3, 4, shape, device=device).to(dtype)


def _expected_output(x, bias):
    """Use Megatron-LM when available, otherwise use the eager formula.

    The Megatron side gets contiguous operands for the value comparison; the
    operator's own acceptance of non-contiguous layouts is exercised
    separately. Empty inputs also use the formula: Megatron restores a rank-3
    result with ``view(..., -1)``, which is ambiguous when the output has zero
    elements.
    """
    promoted = x.dtype if bias is None else torch.promote_types(x.dtype, bias.dtype)
    if promoted is torch.bool:
        # torch's bool ``+`` is logical OR on CPU/CUDA but an integer add on
        # NPU, so evaluate the reference where Megatron's OR semantics hold and
        # bring the result back for the comparison.
        device = x.device
        x = x.cpu()
        bias = None if bias is None else bias.cpu()
    if _megatron_impl is not None and not utils.TO_CPU and x.numel() > 0:
        output = _megatron_impl(
            x.contiguous(),
            None if bias is None else bias.contiguous(),
        )
    else:
        ori_shape = x.shape
        flat = x.view(-1, ori_shape[-1])
        output = bias_geglu_ref(flat, bias)
        if len(ori_shape) != 2:
            output = output.view(ori_shape[0], ori_shape[1], -1)
    return output.to(device) if promoted is torch.bool else output


def _check_bias_geglu(x, bias):
    # Restoring an empty rank-3 result with view(..., -1) is ambiguous.
    if x.ndim == 3 and x.numel() == 0:
        with pytest.raises(RuntimeError):
            flag_train.megatron.bias_geglu_impl(x, bias)
        return

    dtype = _expected_dtype(x, bias)
    y = flag_train.megatron.bias_geglu_impl(x, bias)
    expected = _expected_output(_reference(x), _reference(bias))

    assert y.shape == expected.shape
    assert y.dtype is dtype
    _assert_close(y, expected, dtype)


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_geglu_impl(shape, dtype):
    utils.init_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    _check_bias_geglu(x, None)


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize("shape", _BIAS_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_geglu_impl_bias(shape, dtype):
    utils.init_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    bias = torch.randn(shape[-1], dtype=dtype, device=flag_train.device)
    _check_bias_geglu(x, bias)


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_geglu_impl_bias_shapes(dtype):
    utils.init_seed(4)
    x = torch.randn((8, 32), dtype=dtype, device=flag_train.device)
    _check_bias_geglu(x, torch.randn((1, 32), dtype=dtype, device=flag_train.device))
    _check_bias_geglu(x, torch.randn((8, 32), dtype=dtype, device=flag_train.device))
    _check_bias_geglu(x, torch.tensor(0.25, dtype=dtype, device=flag_train.device))


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_geglu_impl_large_values(dtype):
    """Extreme magnitudes: check GELU/tanh saturation."""
    values = [-1000.0, -90.0, -20.0, -1.0, 0.0, 1.0, 20.0, 90.0, 1000.0, -8.0]
    # 10 gate values and 10 linear values.
    data = values + [-1000.0, -4.0, -1.0, 0.0, 0.5, 1.0, 4.0, 20.0, 90.0, 1000.0]
    x = torch.tensor(data, dtype=torch.float32, device=flag_train.device).to(dtype)
    x = x.view(1, -1)
    _check_bias_geglu(x, None)


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_geglu_impl_noncontiguous(dtype):
    """Layouts Megatron's ``view`` accepts are materialized for the kernel."""
    utils.init_seed(5)
    base = torch.randn((32, 64), dtype=dtype, device=flag_train.device)
    x = base.t()
    bias = torch.randn((64,), dtype=dtype, device=flag_train.device)[::2]

    assert not x.is_contiguous()
    _check_bias_geglu(x, bias)

    cube = torch.randn((4, 8, 16), dtype=dtype, device=flag_train.device)
    transposed = cube.transpose(0, 2)

    # Megatron's public view rejects this layout; so does this operator. The
    # check goes through aten.view, so the Megatron-LM-FL patch that would copy
    # a non-contiguous tensor on NPU cannot make it pass.
    with pytest.raises(ValueError, match="view"):
        flag_train.megatron.bias_geglu_impl(transposed, None)


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_geglu_impl_wide_vector_bias(dtype):
    """A row wider than a column tile keeps the per-row bias load."""
    # hidden = 32768 is wider than every backend tile ceiling, so the launch
    # that hoists a shared bias (one tile covers the row) cannot apply.
    utils.init_seed(7)
    x = torch.randn((8, 32768), dtype=dtype, device=flag_train.device)
    bias = torch.randn((32768,), dtype=dtype, device=flag_train.device)
    _check_bias_geglu(x, bias)


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_rank_validation():
    device = flag_train.device
    with pytest.raises(AssertionError):
        flag_train.megatron.bias_geglu_impl(torch.randn((8,), device=device), None)
    with pytest.raises(AssertionError):
        flag_train.megatron.bias_geglu_impl(
            torch.randn((1, 1, 2, 4), device=device), None
        )


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_bias_broadcast_validation():
    x = torch.randn((4, 8), device=flag_train.device)
    with pytest.raises(RuntimeError):
        flag_train.megatron.bias_geglu_impl(x, torch.randn((3,), device=x.device))


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_geglu_impl_bias_dtype_promotion(dtype):
    """Mixed-dtype bias follows torch's promotion, as upstream does."""
    other = torch.float32 if dtype != torch.float32 else torch.float16
    x = torch.randn((4, 8), dtype=dtype, device=flag_train.device)
    bias = torch.randn((8,), dtype=other, device=flag_train.device)
    _check_bias_geglu(x, bias)


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize("shape", [(4, 16), (2, 3, 16)])
@pytest.mark.parametrize("bias_mode", ["none", "vector", "full"])
def test_bias_geglu_impl_fp64(shape, bias_mode):
    """FP64 inputs use FP64 arithmetic throughout the Triton kernel."""
    if not flag_train.runtime.device.support_fp64:
        pytest.skip("the active device does not support FP64")

    utils.init_seed(6)
    x = torch.randn(shape, dtype=torch.float64, device=flag_train.device)
    flat = x.view(-1, shape[-1])
    if bias_mode == "none":
        bias = None
    elif bias_mode == "vector":
        bias = torch.randn(shape[-1], dtype=torch.float64, device=flag_train.device)
    else:
        bias = torch.randn_like(flat)

    _check_bias_geglu(x, bias)


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize("dtype", _SUPPORTED_TEST_DTYPES)
def test_bias_geglu_impl_dtype_matrix(dtype):
    """Every supported real dtype, with and without bias."""
    if not _fp64_dtype_supported(dtype):
        pytest.skip("the active device does not support FP64")
    x = _dtype_tensor(dtype, (4, 8))
    _check_bias_geglu(x, None)
    _check_bias_geglu(x, _dtype_tensor(dtype, (1, 8)))


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize(("x_dtype", "bias_dtype"), _MIXED_DTYPES)
def test_bias_geglu_impl_mixed_dtype(x_dtype, bias_dtype):
    """Input/bias dtype promotion follows torch, as upstream does."""
    if not (_fp64_dtype_supported(x_dtype) and _fp64_dtype_supported(bias_dtype)):
        pytest.skip("the active device does not support FP64")
    x = _dtype_tensor(x_dtype, (4, 8))
    bias = _dtype_tensor(bias_dtype, (1, 8))
    _check_bias_geglu(x, bias)


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_integer_add_semantics():
    """Integer add wraps in the promoted width; bool add is OR."""
    x = torch.tensor(
        [[120, 120, -120, -120]], dtype=torch.int8, device=flag_train.device
    )
    bias = torch.tensor(
        [[120, 120, -120, -120]], dtype=torch.int8, device=flag_train.device
    )
    _check_bias_geglu(x, bias)

    xu = torch.tensor([[200, 200, 10, 10]], dtype=torch.uint8, device=flag_train.device)
    bu = torch.tensor([[200, 200, 10, 10]], dtype=torch.uint8, device=flag_train.device)
    _check_bias_geglu(xu, bu)

    xb = torch.tensor([[True, False, True, False]], device=flag_train.device)
    bb = torch.tensor([[True, True, False, False]], device=flag_train.device)
    _check_bias_geglu(xb, bb)


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.int32,
        torch.bool,
    ],
)
def test_bias_geglu_impl_cpu(dtype):
    """CPU tensors are staged through the accelerator and copied back."""
    x = _dtype_tensor(dtype, (2, 3, 8), device="cpu")
    _check_bias_geglu(x, None)
    _check_bias_geglu(x, _dtype_tensor(dtype, (8,), device="cpu"))


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_portable_launch():
    """The portable launch keeps matching the registered backend operator.

    On a backend with an override the rest of this module never runs
    ``_bias_geglu_forward_kernel``, so pin it against the registered operator
    on the same operands.
    """
    portable = importlib.import_module("flag_train.megatron.bias_geglu_impl")
    registered = flag_train.megatron.bias_geglu_impl
    if registered is portable.bias_geglu_impl:
        pytest.skip("the registered operator is the portable implementation")

    device = flag_train.device
    utils.init_seed(8)
    fp16 = torch.float16
    cases = [
        (
            torch.randn((64, 128), dtype=fp16, device=device),
            torch.randn((128,), dtype=fp16, device=device),
        ),
        (torch.randn((16, 4096), device=device), torch.randn((4096,), device=device)),
        (torch.randn((8, 16), device=device), torch.randn((8, 16), device=device)),
        (torch.randn((64, 128), device=device), None),
        (torch.randn((2, 3, 8), device=device), torch.randn((8,), device=device)),
        (torch.randn((64, 64), device=device).t(), torch.randn((64,), device=device)),
        (torch.empty((0, 8), device=device), torch.randn((8,), device=device)),
        (
            torch.randint(0, 4, (64, 32), device=device).to(torch.int8),
            torch.randint(0, 4, (32,), device=device).to(torch.int16),
        ),
    ]
    for x, bias in cases:
        expected = portable.bias_geglu_impl(x, bias)
        actual = registered(x, bias)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected), (tuple(x.shape), x.dtype)


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_leading_broadcast():
    """A bias may expand leading dimensions, matching Megatron's broadcast."""
    x = torch.randn((4, 8), device=flag_train.device)
    _check_bias_geglu(x, torch.randn((1, 1, 8), device=flag_train.device))

    x3 = torch.randn((2, 3, 8), device=flag_train.device)
    bias3 = torch.randn((5, 1, 8), device=flag_train.device)
    _check_bias_geglu(x3, bias3)


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_scalar_cpu_bias():
    """A CPU 0-dimensional tensor is a scalar for torch, as it is upstream."""
    if flag_train.device == "cpu":
        pytest.skip("requires distinct CPU and active devices")
    x = torch.randn((4, 8), device=flag_train.device)
    _check_bias_geglu(x, torch.tensor(0.5))


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_scalar_bias_validation():
    x = torch.randn((4, 8), dtype=torch.float16, device=flag_train.device)
    with pytest.raises(TypeError, match="Tensor or None"):
        flag_train.megatron.bias_geglu_impl(x, 0.5)


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_odd_last_dimension():
    """Odd split sizes fail in the same place as Megatron's chunking."""
    x = torch.randn((4, 7), device=flag_train.device)
    with pytest.raises(RuntimeError):
        flag_train.megatron.bias_geglu_impl(x, None)


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_empty_3d_error():
    x = torch.empty((2, 0, 8), device=flag_train.device)
    with pytest.raises(RuntimeError):
        flag_train.megatron.bias_geglu_impl(x, None)


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_forward_only():
    """The forward does not build an autograd graph on either path."""
    x = torch.randn((4, 8), device=flag_train.device, requires_grad=True)
    assert not flag_train.megatron.bias_geglu_impl(x, None).requires_grad

    base = torch.randn((8, 4), device=flag_train.device, requires_grad=True)
    assert not flag_train.megatron.bias_geglu_impl(base.t(), None).requires_grad


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_non_tensor_input():
    with pytest.raises(TypeError, match="tensor"):
        flag_train.megatron.bias_geglu_impl([1.0], None)


@pytest.mark.bias_geglu_impl
@pytest.mark.skipif(not _UNSUPPORTED_DTYPES, reason="no unsupported dtype here")
@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPES)
def test_bias_geglu_impl_unsupported_dtypes(dtype):
    """16/32/64-bit unsigned operands are rejected."""
    # Built on the CPU: the accelerator cannot even allocate these dtypes, and
    # the dtype check runs before any CPU staging.
    x = torch.zeros((4, 8), dtype=dtype)
    with pytest.raises(ValueError, match="unsupported dtype"):
        flag_train.megatron.bias_geglu_impl(x, None)


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_bias_device_validation():
    if flag_train.device == "cpu":
        pytest.skip("requires distinct CPU and active devices")
    x = torch.randn((4, 8), device=flag_train.device)
    bias = torch.randn((8,), device="cpu")
    with pytest.raises(RuntimeError):
        flag_train.megatron.bias_geglu_impl(x, bias)
