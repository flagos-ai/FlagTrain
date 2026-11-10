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
import types

import pytest
import torch

import flag_train

from .. import accuracy_utils as utils

# Megatron's class imports its Apex extension inside forward/backward, so probe
# the two independently.
try:
    import megatron.core.fusions.fused_softmax

    _HAS_MEGATRON = True
except ImportError:
    _HAS_MEGATRON = False

try:
    import scaled_upper_triang_masked_softmax_cuda  # noqa: F401

    _HAS_APEX = True
except ImportError:
    _HAS_APEX = False

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

_SCALES = [0.0, 0.125, 1.0, -0.5, 2]

# On Ascend the reference runs on torch_npu.npu_scaled_masked_softmax: it takes
# fp32, and its ``apply`` is differentiable where the Apex-backed ``backward``
# is not importable.
_ASCEND_AUTOGRAD = False
if flag_train.vendor_name == "ascend":
    _DTYPES.append(torch.float32)
    _ASCEND_AUTOGRAD = _HAS_MEGATRON

if _ASCEND_AUTOGRAD:
    _BASELINE = (
        "megatron-core ScaledUpperTriangMaskedSoftmax.apply + torch.autograd "
        "(Ascend)"
    )
elif _HAS_MEGATRON and _HAS_APEX:
    _BASELINE = "megatron-core ScaledUpperTriangMaskedSoftmax.backward"
else:
    _BASELINE = "scaled_upper_triang_masked_softmax_backward_ref (torch)"


def _mask(x):
    """True above the diagonal of each ``sq x sk`` matrix."""
    rows = torch.arange(x.shape[-2], device=x.device).unsqueeze(-1)
    cols = torch.arange(x.shape[-1], device=x.device).unsqueeze(0)
    return cols > rows


def scaled_upper_triang_masked_softmax_forward_ref(x, scale):
    """Forward composition, the same one the forward suite falls back to."""
    scores = x.to(torch.float32) * scale
    masked = _mask(x)
    scores = scores.masked_fill(masked, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return probs.masked_fill(masked, 0.0).to(x.dtype)


def scaled_upper_triang_masked_softmax_backward_ref(grad_output, output, scale):
    """``scale * y * (dy - rowsum(dy * y))`` over the kept prefix, zero above it."""
    masked = _mask(output)
    dy = grad_output.to(torch.float32)
    # Masked entries are zero already; fill anyway in case they are not.
    y = output.to(torch.float32).masked_fill(masked, 0.0)
    total = (dy * y).sum(dim=-1, keepdim=True)
    dx = scale * y * (dy - total)
    return dx.masked_fill(masked, 0.0).to(grad_output.dtype)


def _reference(x, grad_output, scale):
    """The forward output ``y`` and the gradient of its baseline w.r.t. ``x``."""
    if _ASCEND_AUTOGRAD:
        # Autograd through ``apply``; the leaf is a detached view of ``x``.
        leaf = x.contiguous().detach().requires_grad_(True)
        output = (
            megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax.apply(
                leaf, scale
            )
        )
        (ref,) = torch.autograd.grad(output, leaf, grad_output.clone())
        return output, ref

    if _HAS_MEGATRON and _HAS_APEX:
        output = (
            megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax.apply(
                x.contiguous(), scale
            )
        )
        # Stand in for the engine's ctx, holding what ``forward`` saved.
        ctx = types.SimpleNamespace(saved_tensors=(output, torch.tensor([scale])))
        # Megatron's backward writes in place, hence the clone.
        ref, _ = (
            megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax.backward(
                ctx, grad_output.clone()
            )
        )
        return output, ref

    output = scaled_upper_triang_masked_softmax_forward_ref(x, scale)
    ref = scaled_upper_triang_masked_softmax_backward_ref(grad_output, output, scale)
    return output, ref


@pytest.mark.scaled_upper_triang_masked_softmax_backward
def test_baseline_source():
    assert _BASELINE
    print(f"\nBaseline: {_BASELINE}")


@pytest.mark.scaled_upper_triang_masked_softmax_backward
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("scale", _SCALES)
def test_scaled_upper_triang_masked_softmax_backward(shape, dtype, scale):
    utils.init_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_train.device)

    output, ref = _reference(x, grad_output, scale)
    res = flag_train.megatron.scaled_upper_triang_masked_softmax_backward(
        grad_output, output.detach(), scale
    )

    # Only the reference follows ``--ref cpu``.
    if utils.TO_CPU:
        ref = utils.to_reference(ref)

    utils.train_assert_close(res, ref, dtype)


@pytest.mark.scaled_upper_triang_masked_softmax_backward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_scaled_upper_triang_masked_softmax_backward_zero_grad(dtype):
    """A zero upstream gradient must give a zero gradient, masked half included."""
    shape = (16, 128, 128)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    grad_output = torch.zeros(shape, dtype=dtype, device=flag_train.device)

    output, ref = _reference(x, grad_output, 0.125)
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

    # The operator gets strided views of both arguments, as a downstream
    # gradient would arrive.
    x = torch.randn((16, 32, 64), dtype=dtype, device=flag_train.device)[..., ::2]
    x = x.contiguous()

    wide_grads = torch.randn((16, 32, 64), dtype=dtype, device=flag_train.device)
    grad_output = wide_grads[..., ::2]

    output, ref = _reference(x, grad_output, 0.125)

    wide_output = torch.zeros((16, 32, 64), dtype=dtype, device=flag_train.device)
    wide_output[..., ::2] = output
    strided_output = wide_output[..., ::2]

    assert not grad_output.is_contiguous()
    assert not strided_output.is_contiguous()

    res = flag_train.megatron.scaled_upper_triang_masked_softmax_backward(
        grad_output, strided_output, 0.125
    )

    if utils.TO_CPU:
        ref = utils.to_reference(ref)

    utils.train_assert_close(res, ref, dtype)
