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
"""Correctness tests for fused bias + SwiGLU.

The reference is the eager formula Megatron-LM checks its own fusion against:
SiLU (or SiTU) of the two halves, with the bias added by ordinary
broadcasting. A separate differential test compares the plain path against
``megatron.core.fusions.fused_bias_swiglu`` when Megatron-LM is installed, so
the kernel is not only checked against a formula written for this file.
Under ``--ref cpu`` those operands are moved to the CPU and upcast, so the
comparison is against a higher-precision CPU result.
Tolerances are ``train_assert_close``'s defaults.

Megatron-LM's ``BiasSwiGLUFunction.backward`` returns the unreduced
pre-activation gradient for the bias as well, and autograd reduces it for
broadcast-compatible ``bias`` shapes. These tests expect the broadcast
gradient, reduced to ``bias.shape``, which is also what the eager reference
produces. ``bias`` must have the same dtype as ``input``.
"""

import warnings

import pytest
import torch

import flag_train

from .. import accuracy_utils as utils


def base_op():
    """Megatron-LM's SwiGLU module, or ``None`` when it is not installed.

    Mirrors ``base_op`` in ``test_quick_gelu.py``: the import happens here and
    a missing Megatron-LM is a warning, not a collection error.
    """
    try:
        import megatron.core.fusions.fused_bias_swiglu
    except ImportError:
        warnings.warn(
            "Megatron-LM is unavailable; skipping the differential tests",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return megatron.core.fusions.fused_bias_swiglu


reference_megatron = base_op()

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

# bias 与 shape / variant 正交，只在代表形状和变体上做带 bias 的抽查。
_BIAS_SHAPES = [
    (1, 2),
    (17, 32),
    (64, 128),
    (4, 7, 32),
]

_VARIANTS = {
    "plain": {},
    "clamped": {"clamp_value": 1.5},
    "situ": {"gate_clamp_scale": 4.0},
    "situ_linear": {"gate_clamp_scale": 4.0, "linear_clamp_scale": 2.5},
}


def bias_swiglu_ref(
    x,
    bias=None,
    fp8_input_store=False,
    cpu_offload_input=False,
    clamp_value=None,
    gate_clamp_scale=None,
    linear_clamp_scale=None,
):
    del fp8_input_store, cpu_offload_input
    y = x if bias is None else x + bias
    gate, linear = torch.chunk(y, 2, dim=-1)
    if gate_clamp_scale is not None:
        gate_v = gate_clamp_scale * torch.tanh(gate / gate_clamp_scale)
        gate_v = gate_v * torch.sigmoid(gate)
        if linear_clamp_scale is not None:
            linear = linear_clamp_scale * torch.tanh(linear / linear_clamp_scale)
        return gate_v * linear
    if clamp_value is not None and clamp_value > 0:
        return torch.nn.functional.silu(gate.clamp(max=clamp_value)) * linear.clamp(
            min=-clamp_value, max=clamp_value
        )
    return torch.nn.functional.silu(gate) * linear


def weighted_bias_swiglu_ref(
    x,
    bias,
    weights,
    fp8_input_store=False,
    clamp_value=None,
    gate_clamp_scale=None,
    linear_clamp_scale=None,
):
    if bias is not None:
        raise NotImplementedError("Bias is not supported for weighted swiglu fusion")
    activated = bias_swiglu_ref(
        x,
        None,
        clamp_value=clamp_value,
        gate_clamp_scale=gate_clamp_scale,
        linear_clamp_scale=linear_clamp_scale,
    )
    del fp8_input_store
    return activated * weights


def _reference(tensor):
    if tensor is None:
        return None
    return utils.to_reference(tensor.detach(), upcast=True)


def _assert_close(actual, expected, dtype):
    utils.train_assert_close(utils.to_reference(actual), expected, dtype)


# Megatron-LM's own tests use dtype-specific tolerances, because the reference
# and the fused path round independently. Its unreduced bias gradient is also
# rounded to the input dtype before autograd sums it, while this kernel sums
# the fp32 values, so the bias slot needs the same headroom.
_MEGATRON_TOLERANCES = {
    torch.float32: {"rtol": 1.0e-6, "atol": 1.0e-6},
    torch.float16: {"rtol": 1.0e-3, "atol": 1.0e-3},
    torch.bfloat16: {"rtol": 2.0e-2, "atol": 1.0e-2},
}


def _assert_megatron_close(actual, expected, dtype):
    torch.testing.assert_close(
        actual.detach().float().cpu(),
        expected.detach().float().cpu(),
        **_MEGATRON_TOLERANCES[dtype],
    )


def _variant_kwargs(name):
    return dict(_VARIANTS[name])


def _scale_for(name):
    # Push a share of the values into the clamp or the tanh saturation region.
    if name == "plain":
        return 1.0
    if name == "clamped":
        return 3.0
    return 5.0


def _check_bias_swiglu(x, bias, **kwargs):
    dtype = x.dtype
    y = flag_train.megatron.bias_swiglu_impl(x, bias, **kwargs)
    assert y.shape[-1] * 2 == x.shape[-1]
    assert y.shape[:-1] == x.shape[:-1]
    assert y.dtype == dtype

    x_ref = _reference(x)
    bias_ref = _reference(bias)
    _assert_close(y, bias_swiglu_ref(x_ref, bias_ref, **kwargs), dtype)

    grad_out = torch.randn_like(y)
    x_train = x.detach().requires_grad_(True)
    bias_train = None if bias is None else bias.detach().requires_grad_(True)
    flag_train.megatron.bias_swiglu_impl(x_train, bias_train, **kwargs).backward(
        grad_out
    )

    x_ref = _reference(x).requires_grad_(True)
    bias_ref = None if bias is None else _reference(bias).requires_grad_(True)
    bias_swiglu_ref(x_ref, bias_ref, **kwargs).backward(_reference(grad_out))
    _assert_close(x_train.grad, x_ref.grad, dtype)
    if bias is not None:
        assert bias_train.grad.shape == bias.shape
        _assert_close(bias_train.grad, bias_ref.grad, bias.dtype)


def _check_weighted(x, weights, **kwargs):
    dtype = x.dtype
    y = flag_train.megatron.weighted_bias_swiglu_impl(x, None, weights, **kwargs)
    assert y.shape[-1] * 2 == x.shape[-1]
    assert y.dtype == dtype

    x_ref = _reference(x)
    w_ref = _reference(weights)
    _assert_close(y, weighted_bias_swiglu_ref(x_ref, None, w_ref, **kwargs), dtype)

    grad_out = torch.randn_like(y)
    x_train = x.detach().requires_grad_(True)
    w_train = weights.detach().requires_grad_(True)
    flag_train.megatron.weighted_bias_swiglu_impl(
        x_train, None, w_train, **kwargs
    ).backward(grad_out)

    x_ref = _reference(x).requires_grad_(True)
    w_ref = _reference(weights).requires_grad_(True)
    weighted_bias_swiglu_ref(x_ref, None, w_ref, **kwargs).backward(
        _reference(grad_out)
    )
    _assert_close(x_train.grad, x_ref.grad, dtype)
    assert w_train.grad.shape == weights.shape
    _assert_close(w_train.grad, w_ref.grad, weights.dtype)


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_swiglu_impl(shape, dtype):
    utils.init_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    _check_bias_swiglu(x, None)


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("shape", _BIAS_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_swiglu_impl_bias(shape, dtype):
    utils.init_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    bias = torch.randn(shape[-1], dtype=dtype, device=flag_train.device)
    _check_bias_swiglu(x, bias)


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("variant", list(_VARIANTS))
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_swiglu_impl_variants(variant, dtype):
    utils.init_seed(1)
    scale = _scale_for(variant)
    x = torch.randn((16, 64), dtype=dtype, device=flag_train.device) * scale
    _check_bias_swiglu(x, None, **_variant_kwargs(variant))


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("variant", list(_VARIANTS))
def test_bias_swiglu_impl_variants_bias(variant):
    utils.init_seed(1)
    scale = _scale_for(variant)
    x = torch.randn((16, 64), dtype=torch.float16, device=flag_train.device) * scale
    bias = torch.randn((64,), dtype=torch.float16, device=flag_train.device)
    _check_bias_swiglu(x, bias, **_variant_kwargs(variant))


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("variant", list(_VARIANTS))
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_weighted_bias_swiglu_impl(variant, dtype):
    utils.init_seed(2)
    scale = _scale_for(variant)
    x = torch.randn((16, 64), dtype=dtype, device=flag_train.device) * scale
    weights = torch.randn((16, 1), dtype=torch.float32, device=flag_train.device)
    _check_weighted(x, weights, **_variant_kwargs(variant))


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_weighted_bias_swiglu_impl_broadcast(dtype):
    """Weights shaped like the unflattened output, a scalar, and a full output."""
    utils.init_seed(3)
    x = torch.randn((2, 4, 32), dtype=dtype, device=flag_train.device)
    per_token = torch.randn((2, 4, 1), dtype=torch.float32, device=flag_train.device)
    _check_weighted(x, per_token)

    scalar = torch.tensor([0.5], dtype=torch.float32, device=flag_train.device)
    _check_weighted(x, scalar)

    full = torch.randn((2, 4, 16), dtype=dtype, device=flag_train.device)
    _check_weighted(x, full, clamp_value=2.0)


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_swiglu_impl_bias_shapes(dtype):
    utils.init_seed(4)
    x = torch.randn((8, 32), dtype=dtype, device=flag_train.device)
    _check_bias_swiglu(x, torch.randn((1, 32), dtype=dtype, device=flag_train.device))
    _check_bias_swiglu(x, torch.randn((8, 32), dtype=dtype, device=flag_train.device))
    _check_bias_swiglu(
        x,
        torch.tensor(0.25, dtype=dtype, device=flag_train.device),
        gate_clamp_scale=3.0,
    )


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_swiglu_impl_clamp_boundary(dtype):
    """Values that land exactly on the clamp, just inside, and just outside."""
    clamp = 1.0
    gate = torch.tensor(
        [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0], dtype=torch.float32
    )
    linear = torch.tensor(
        [-3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 1.5, 2.0], dtype=torch.float32
    )
    values = torch.cat((gate, linear)).to(dtype=dtype, device=flag_train.device)
    x = values.view(1, -1)
    bias = torch.zeros((x.shape[-1],), dtype=dtype, device=flag_train.device)
    _check_bias_swiglu(x, bias, clamp_value=clamp)


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_swiglu_impl_large_values(dtype):
    values = [-1000.0, -90.0, -20.0, -1.0, 0.0, 1.0, 20.0, 90.0, 1000.0, -8.0]
    # 10 gate values and 10 linear values.
    data = values + [-1000.0, -4.0, -1.0, 0.0, 0.5, 1.0, 4.0, 20.0, 90.0, 1000.0]
    x = torch.tensor(data, dtype=torch.float32, device=flag_train.device).to(dtype)
    x = x.view(1, -1)
    _check_bias_swiglu(x, None, gate_clamp_scale=5.0, linear_clamp_scale=3.0)
    _check_bias_swiglu(x, None)


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_swiglu_impl_noncontiguous(dtype):
    utils.init_seed(5)
    base = torch.randn((32, 64), dtype=dtype, device=flag_train.device)
    x = base.t()
    assert x.shape == (64, 32)
    assert not x.is_contiguous()
    bias = torch.randn((64,), dtype=dtype, device=flag_train.device)[::2]
    assert bias.shape == (32,)
    assert not bias.is_contiguous()
    _check_bias_swiglu(x, bias, clamp_value=2.0)

    cube = torch.randn((4, 8, 16), dtype=dtype, device=flag_train.device)
    transposed = cube.transpose(0, 2)
    assert not transposed.is_contiguous()
    assert transposed.shape[-1] % 2 == 0
    _check_bias_swiglu(transposed, None, gate_clamp_scale=2.0)


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_swiglu_impl_fp8_input_store(dtype):
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("float8_e4m3fn is not available")
    utils.init_seed(6)
    x = torch.randn((8, 32), dtype=dtype, device=flag_train.device)
    bias = torch.randn((32,), dtype=dtype, device=flag_train.device)
    y = flag_train.megatron.bias_swiglu_impl(x, bias, fp8_input_store=True)
    y_full = flag_train.megatron.bias_swiglu_impl(x, bias, fp8_input_store=False)
    _assert_close(y, _reference(y_full), dtype)

    x_q = x.detach().to(torch.float8_e4m3fn).to(dtype)
    bias_train = bias.detach().requires_grad_(True)
    x_train = x.detach().requires_grad_(True)
    grad_out = torch.randn_like(y)
    flag_train.megatron.bias_swiglu_impl(
        x_train, bias_train, fp8_input_store=True
    ).backward(grad_out)

    x_ref = _reference(x_q).requires_grad_(True)
    bias_ref = _reference(bias).requires_grad_(True)
    bias_swiglu_ref(x_ref, bias_ref).backward(_reference(grad_out))
    _assert_close(x_train.grad, x_ref.grad, dtype)
    _assert_close(bias_train.grad, bias_ref.grad, bias.dtype)


@pytest.mark.bias_swiglu_impl
def test_bias_swiglu_impl_cpu_offload_marks_saved_tensors():
    dtype = torch.float32
    x = torch.randn((4, 8), dtype=dtype, device=flag_train.device, requires_grad=True)
    bias = torch.randn((8,), dtype=dtype, device=flag_train.device, requires_grad=True)
    y = flag_train.megatron.bias_swiglu_impl(x, bias, cpu_offload_input=True)
    assert getattr(x, "activation_offloading", False) is True
    assert getattr(bias, "activation_offloading", False) is True
    y.backward(torch.ones_like(y))
    assert x.grad is not None
    assert bias.grad is not None


@pytest.mark.bias_swiglu_impl
def test_bias_swiglu_impl_rejects_bad_arguments():
    device = flag_train.device
    x = torch.randn((4, 8), device=device)
    with pytest.raises(ValueError, match="rank 2 or 3"):
        flag_train.megatron.bias_swiglu_impl(torch.randn((8,), device=device), None)
    with pytest.raises(ValueError, match="rank 2 or 3"):
        flag_train.megatron.bias_swiglu_impl(
            torch.randn((1, 1, 2, 4), device=device), None
        )
    with pytest.raises(ValueError, match="even"):
        flag_train.megatron.bias_swiglu_impl(torch.randn((4, 7), device=device), None)
    with pytest.raises(ValueError, match="CPU"):
        flag_train.megatron.bias_swiglu_impl(torch.randn((4, 8)), None)
    with pytest.raises(ValueError, match="cannot both"):
        flag_train.megatron.bias_swiglu_impl(
            x, None, clamp_value=1.0, gate_clamp_scale=1.0
        )
    with pytest.raises(ValueError, match="requires gate_clamp_scale"):
        flag_train.megatron.bias_swiglu_impl(x, None, linear_clamp_scale=1.0)
    with pytest.raises(ValueError, match="greater than zero"):
        flag_train.megatron.bias_swiglu_impl(x, None, clamp_value=0.0)
    with pytest.raises(ValueError, match="greater than zero"):
        flag_train.megatron.bias_swiglu_impl(x, None, gate_clamp_scale=-1.0)
    with pytest.raises(NotImplementedError, match="Bias is not supported"):
        flag_train.megatron.weighted_bias_swiglu_impl(
            x, x.new_zeros(8), x.new_ones((4, 1))
        )
    with pytest.raises(ValueError, match="broadcast"):
        flag_train.megatron.bias_swiglu_impl(x, torch.randn((3,), device=device))


@pytest.mark.bias_swiglu_impl
def test_bias_swiglu_impl_empty_input_validates_bias():
    """An empty input must not skip the bias checks."""
    device = flag_train.device
    x = torch.empty((0, 8), device=device)
    with pytest.raises(ValueError, match="broadcast"):
        flag_train.megatron.bias_swiglu_impl(x, torch.randn((3,), device=device))


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_swiglu_impl_requires_matching_bias_dtype(dtype):
    """The kernel computes the add in fp32; a mixed-dtype bias would silently
    change the promoted dtype eager PyTorch produces, so it is rejected."""
    device = flag_train.device
    other = torch.float32 if dtype != torch.float32 else torch.float16
    x = torch.randn((4, 8), dtype=dtype, device=device)
    with pytest.raises(ValueError, match="input dtype"):
        flag_train.megatron.bias_swiglu_impl(
            x, torch.randn((8,), dtype=other, device=device)
        )


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_weighted_bias_swiglu_impl_noncontiguous_weights(dtype):
    utils.init_seed(8)
    x = torch.randn((8, 64), dtype=dtype, device=flag_train.device)

    per_token = torch.randn((8, 2), dtype=torch.float32, device=flag_train.device)[
        :, ::2
    ]
    assert per_token.shape == (8, 1)
    assert not per_token.is_contiguous()
    _check_weighted(x, per_token)

    elementwise = torch.randn((32, 8), dtype=dtype, device=flag_train.device).t()
    assert elementwise.shape == (8, 32)
    assert not elementwise.is_contiguous()
    _check_weighted(x, elementwise, clamp_value=2.0)


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_weighted_bias_swiglu_impl_zero_stride_weights(dtype):
    """A scalar expanded to the output shape exercises the stride-0 path."""
    utils.init_seed(9)
    x = torch.randn((8, 64), dtype=dtype, device=flag_train.device)
    weights = torch.tensor(0.5, dtype=torch.float32, device=flag_train.device).expand(
        8, 32
    )
    assert weights.stride() == (0, 0)
    _check_weighted(x, weights)


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_weighted_bias_swiglu_impl_fp8_input_store(dtype):
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("float8_e4m3fn is not available")
    utils.init_seed(10)
    x = torch.randn((8, 32), dtype=dtype, device=flag_train.device)
    weights = torch.randn((8, 1), dtype=torch.float32, device=flag_train.device)

    y = flag_train.megatron.weighted_bias_swiglu_impl(
        x, None, weights, fp8_input_store=True
    )
    y_full = flag_train.megatron.weighted_bias_swiglu_impl(
        x, None, weights, fp8_input_store=False
    )
    _assert_close(y, _reference(y_full), dtype)

    x_q = x.detach().to(torch.float8_e4m3fn).to(dtype)
    x_train = x.detach().requires_grad_(True)
    w_train = weights.detach().requires_grad_(True)
    grad_out = torch.randn_like(y)
    flag_train.megatron.weighted_bias_swiglu_impl(
        x_train, None, w_train, fp8_input_store=True
    ).backward(grad_out)

    x_ref = _reference(x_q).requires_grad_(True)
    w_ref = _reference(weights).requires_grad_(True)
    weighted_bias_swiglu_ref(x_ref, None, w_ref).backward(_reference(grad_out))
    _assert_close(x_train.grad, x_ref.grad, dtype)
    _assert_close(w_train.grad, w_ref.grad, weights.dtype)


@pytest.mark.bias_swiglu_impl
def test_bias_swiglu_impl_skips_unneeded_parameter_grads():
    device = flag_train.device
    x = torch.randn((4, 8), device=device, requires_grad=True)
    bias = torch.randn((8,), device=device)
    flag_train.megatron.bias_swiglu_impl(x, bias).sum().backward()
    assert x.grad is not None
    assert bias.grad is None

    x = torch.randn((4, 8), device=device, requires_grad=True)
    weights = torch.randn((4, 1), device=device)
    flag_train.megatron.weighted_bias_swiglu_impl(x, None, weights).sum().backward()
    assert x.grad is not None
    assert weights.grad is None


@pytest.mark.bias_swiglu_impl
@pytest.mark.skipif(reference_megatron is None, reason="Megatron-LM is not installed")
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("with_bias", [False, True])
def test_bias_swiglu_impl_matches_megatron(dtype, with_bias):
    """Differential test against Megatron-LM's own plain SwiGLU path."""
    utils.init_seed(11)
    x = torch.randn((16, 64), dtype=dtype, device=flag_train.device)
    bias = (
        torch.randn((64,), dtype=dtype, device=flag_train.device) if with_bias else None
    )
    grad_out = torch.randn((16, 32), dtype=dtype, device=flag_train.device)

    x_actual = x.detach().requires_grad_(True)
    bias_actual = None if bias is None else bias.detach().requires_grad_(True)
    y_actual = flag_train.megatron.bias_swiglu_impl(x_actual, bias_actual)
    y_actual.backward(grad_out)

    x_ref = x.detach().requires_grad_(True)
    bias_ref = None if bias is None else bias.detach().requires_grad_(True)
    y_ref = reference_megatron.bias_swiglu_impl(x_ref, bias_ref)
    y_ref.backward(grad_out)

    _assert_megatron_close(y_actual, _reference(y_ref), dtype)
    _assert_megatron_close(x_actual.grad, _reference(x_ref.grad), dtype)
    if bias is not None:
        _assert_megatron_close(bias_actual.grad, _reference(bias_ref.grad), dtype)


@pytest.mark.bias_swiglu_impl
@pytest.mark.skipif(
    reference_megatron is None
    or not hasattr(reference_megatron, "weighted_bias_swiglu_impl"),
    reason="Megatron-LM weighted SwiGLU is not available",
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_weighted_bias_swiglu_impl_matches_megatron(dtype):
    """Differential test against Megatron-LM's weighted path."""
    utils.init_seed(12)
    x = torch.randn((16, 64), dtype=dtype, device=flag_train.device)
    weights = torch.randn((16, 1), dtype=torch.float32, device=flag_train.device)
    grad_out = torch.randn((16, 32), dtype=dtype, device=flag_train.device)

    x_actual = x.detach().requires_grad_(True)
    weights_actual = weights.detach().requires_grad_(True)
    y_actual = flag_train.megatron.weighted_bias_swiglu_impl(
        x_actual, None, weights_actual
    )
    y_actual.backward(grad_out)

    x_ref = x.detach().requires_grad_(True)
    weights_ref = weights.detach().requires_grad_(True)
    y_ref = reference_megatron.weighted_bias_swiglu_impl(x_ref, None, weights_ref)
    y_ref.backward(grad_out)

    _assert_megatron_close(y_actual, _reference(y_ref), dtype)
    _assert_megatron_close(x_actual.grad, _reference(x_ref.grad), dtype)
    _assert_megatron_close(
        weights_actual.grad, _reference(weights_ref.grad), weights.dtype
    )
