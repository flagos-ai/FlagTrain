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

from .. import base

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

# Megatron's CUDA fused kernel rejects fp32.
_DTYPES = [torch.float16, torch.bfloat16]

# Scales are not comparable across runs if they differ, so the 1/sqrt(d) used
# for attention scores is fixed here.
_SCALE = 0.125

_SHAPES = [
    (64, 64, 64),
    (16, 128, 128),
    (64, 128, 128),
    (16, 256, 256),
    (32, 256, 256),
    (32, 512, 512),
    (16, 1024, 1024),
    (8, 2048, 2048),
    (16, 2048, 2048),
    (8, 4096, 4096),
    (32, 4096, 4096),
    (8, 8192, 8192),
]

# On Ascend the reference runs on torch_npu.npu_scaled_masked_softmax: it takes
# fp32, and its ``apply`` is differentiable where the Apex-backed ``backward``
# is not importable.
_ASCEND_AUTOGRAD = False
if flag_train.vendor_name == "ascend":
    _DTYPES.append(torch.float32)
    _ASCEND_AUTOGRAD = _HAS_MEGATRON


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


def _forward(x, scale):
    """The forward output the case hands to the operator."""
    if _ASCEND_AUTOGRAD or (_HAS_MEGATRON and _HAS_APEX):
        return megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax.apply(
            x.contiguous(), scale
        )
    return scaled_upper_triang_masked_softmax_forward_ref(x, scale)


if _ASCEND_AUTOGRAD:
    _BASELINE = (
        "megatron-core ScaledUpperTriangMaskedSoftmax.apply + torch.autograd "
        "(Ascend)"
    )
elif _HAS_MEGATRON and _HAS_APEX:
    _BASELINE = "megatron-core ScaledUpperTriangMaskedSoftmax.backward"
else:
    _BASELINE = "scaled_upper_triang_masked_softmax_backward_ref (torch)"


class _BaselineBackward:
    """Megatron's backward, or the torch composition without the Apex kernels.

    With Apex the static ``backward`` runs behind a stand-in ctx and overwrites
    ``grad_output``, so repeats see the previous result; only shape and dtype
    reach the timing. On Ascend it is not importable, so the gradient comes from
    ``apply`` + ``torch.autograd``, graphed once per case. ``_source`` keeps that
    case's ``output`` alive.
    """

    def __init__(self):
        self._source = None

    def __call__(self, grad_output, output, scale):
        if _ASCEND_AUTOGRAD:
            if output is not self._source:
                self._source = output
                self._leaf = torch.randn(
                    output.shape, dtype=output.dtype, device=output.device
                ).requires_grad_(True)
                self._output = megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax.apply(
                    self._leaf, scale
                )
            return torch.autograd.grad(
                self._output, self._leaf, grad_output, retain_graph=True
            )[0]

        if not (_HAS_MEGATRON and _HAS_APEX):
            return scaled_upper_triang_masked_softmax_backward_ref(
                grad_output, output, scale
            )
        if output is not self._source:
            self._source = output
            self._ctx = types.SimpleNamespace(
                saved_tensors=(output, torch.tensor([scale]))
            )
        return (
            megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax.backward(
                self._ctx, grad_output
            )[0]
        )


class _ScaledUpperTriangMaskedSoftmaxBackwardBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _SHAPES
    DEFAULT_SHAPE_DESC = "attn_batches, sq, sk"

    def set_shapes(self, shape_file=None):
        self.shapes = list(_SHAPES)


def scaled_upper_triang_masked_softmax_backward_input_fn(shape, dtype, device):
    """``(grad_output, output, scale)``; ``output`` is the reference's forward."""
    x = torch.randn(shape, dtype=dtype, device=device)
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    yield grad_output, _forward(x, _SCALE), _SCALE


@pytest.mark.scaled_upper_triang_masked_softmax_backward
def test_scaled_upper_triang_masked_softmax_backward():
    print(f"\nBaseline: {_BASELINE}")
    bench = _ScaledUpperTriangMaskedSoftmaxBackwardBenchmark(
        input_fn=scaled_upper_triang_masked_softmax_backward_input_fn,
        op_name="scaled_upper_triang_masked_softmax_backward",
        torch_op=_BaselineBackward(),
        dtypes=_DTYPES,
    )
    bench.set_train(flag_train.megatron.scaled_upper_triang_masked_softmax_backward)
    bench.run()
