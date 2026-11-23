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
import warnings

import pytest
import torch

import flag_train

from .. import base

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

# Megatron's CUDA fused kernel rejects fp32.
_DTYPES = [torch.float16, torch.bfloat16]

# On Ascend the reference runs on torch_npu.npu_scaled_masked_softmax, which
# supports fp32.
if flag_train.vendor_name == "ascend":
    _DTYPES.append(torch.float32)

_EXTENSION = "scaled_upper_triang_masked_softmax_cuda"

_NO_BASELINE = "none (no on-machine baseline; --base-data is the only comparison)"


class _SavedForBackward:
    """Stand-in for the context ``ScaledUpperTriangMaskedSoftmax.backward`` takes.

    Megatron's forward saves ``(softmax_results, scale_t)``; the backward only
    reads ``ctx.saved_tensors``, so a two-attribute object runs the installed
    body without a forward pass in the timed region.
    """

    def __init__(self, softmax_results, scale):
        self.saved_tensors = (softmax_results, torch.tensor([scale]))


def _vendored_backward(grad_output, output, scale):
    """``ScaledUpperTriangMaskedSoftmax.backward``'s body, minus the ``ctx``."""
    import scaled_upper_triang_masked_softmax_cuda

    return scaled_upper_triang_masked_softmax_cuda.backward(grad_output, output, scale)


class _MegatronClassBackward:
    """Megatron's static ``backward`` behind a saved-tensor ctx.

    The ctx is built once per case: ``torch.tensor([scale])`` inside the timed
    region would charge the baseline for work Megatron does in its forward.
    Megatron's CUDA backward overwrites ``grad_output``, so repeats run on the
    previous result; only shape and dtype reach the timing.
    """

    def __init__(self, cls):
        self._cls = cls
        self._source = None
        self._ctx = None

    def __call__(self, grad_output, output, scale):
        if output is not self._source:
            self._source = output
            self._ctx = _SavedForBackward(output, scale)
        return self._cls.backward(self._ctx, grad_output)[0]


class _MegatronAutogradBackward:
    """``torch.autograd`` through Megatron's Ascend ``apply``, graphed once per case.

    The graph is built outside the timed region and re-run with
    ``retain_graph=True``.
    """

    def __init__(self, apply):
        self._apply = apply
        self._source = None

    def __call__(self, grad_output, output, scale):
        if output is not self._source:
            self._source = output
            self._leaf = torch.randn(
                output.shape, dtype=output.dtype, device=output.device
            ).requires_grad_(True)
            self._output = self._apply(self._leaf, scale)
        return torch.autograd.grad(
            self._output, self._leaf, grad_output, retain_graph=True
        )[0]


def _warn_missing_baseline(reason):
    """Report a missing on-machine baseline: no torch composition stands in."""
    warnings.warn(
        f"{reason} This run has no on-machine baseline; a torch composition is "
        "not timed, so --base-data is the only comparison.",
        RuntimeWarning,
        stacklevel=2,
    )


def _resolve_megatron_baseline():
    """``(op, label)``: Megatron's backward, or ``None`` when this machine has none."""
    try:
        from megatron.core.fusions.fused_softmax import ScaledUpperTriangMaskedSoftmax
    except Exception as exc:  # pragma: no cover - depends on the installation
        cls = None
        import_error = exc
    else:
        cls = ScaledUpperTriangMaskedSoftmax
        import_error = None

    if flag_train.vendor_name == "ascend":
        if cls is None:
            _warn_missing_baseline(
                "Megatron-LM's ScaledUpperTriangMaskedSoftmax could not be "
                f"imported ({import_error!r})."
            )
            return None, _NO_BASELINE
        # Megatron's ``backward`` imports the Apex extension, which Ascend does
        # not ship; its NPU plugin points ``apply`` at
        # ``torch_npu.npu_scaled_masked_softmax``, which is differentiable.
        return (
            _MegatronAutogradBackward(cls.apply),
            "megatron-core ScaledUpperTriangMaskedSoftmax.apply + torch.autograd "
            "(Ascend)",
        )

    try:
        import scaled_upper_triang_masked_softmax_cuda  # noqa: F401
    except ImportError:
        _warn_missing_baseline(
            "Megatron-LM's fused upper-triangular softmax needs the Apex CUDA "
            f"extension {_EXTENSION!r}, which is not installed."
        )
        return None, _NO_BASELINE

    if cls is not None:
        return (
            _MegatronClassBackward(cls),
            "megatron-core ScaledUpperTriangMaskedSoftmax.backward",
        )

    warnings.warn(
        "megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax "
        f"could not be imported ({import_error!r}); using its vendored body "
        f"against the same Apex extension {_EXTENSION!r}.",
        RuntimeWarning,
        stacklevel=2,
    )
    return (
        _vendored_backward,
        "Megatron-LM ScaledUpperTriangMaskedSoftmax.backward (vendored body)",
    )


# Resolved once, so the first-use import is not counted in the measurement.
# ``None`` is the benchmark harness's "no on-machine baseline" sentinel; the
# harness checks ``torch_op is not None`` before timing anything.
torch_op, _BASELINE = _resolve_megatron_baseline()


class _ScaledUpperTriangMaskedSoftmaxBackwardBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _SHAPES
    DEFAULT_SHAPE_DESC = "attn_batches, sq, sk"

    def set_shapes(self, shape_file=None):
        self.shapes = list(_SHAPES)


def scaled_upper_triang_masked_softmax_backward_input_fn(shape, dtype, device):
    """``(grad_output, output, scale)`` for one ``[attn_batches, sq, sk]`` case.

    ``output`` is a plausible forward result: non-negative rows that sum to one
    over the kept prefix and are zero above the diagonal. The kernels' latency
    does not depend on the values, so no forward pass is needed here.
    """
    rows = torch.arange(shape[-2], device=device).unsqueeze(-1)
    cols = torch.arange(shape[-1], device=device).unsqueeze(0)
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    output = torch.rand(shape, dtype=dtype, device=device)
    output = output.masked_fill(cols > rows, 0.0)
    output = output / output.sum(dim=-1, keepdim=True)
    yield grad_output, output, _SCALE


@pytest.mark.scaled_upper_triang_masked_softmax_backward
def test_scaled_upper_triang_masked_softmax_backward():
    print(f"\nBaseline: {_BASELINE}")
    bench = _ScaledUpperTriangMaskedSoftmaxBackwardBenchmark(
        input_fn=scaled_upper_triang_masked_softmax_backward_input_fn,
        op_name="scaled_upper_triang_masked_softmax_backward",
        torch_op=torch_op,
        dtypes=_DTYPES,
    )
    bench.set_train(flag_train.megatron.scaled_upper_triang_masked_softmax_backward)
    bench.run()
