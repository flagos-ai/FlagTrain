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

"""NVIDIA baseline for scaled_upper_triang_masked_softmax_backward.

Self-contained module; depends only on torch and megatron, and does not import
``flag_train`` or ``benchmark``.

``native()`` returns Megatron-LM's ``ScaledUpperTriangMaskedSoftmax.backward``
(``megatron/core/fusions/fused_softmax.py``) -- the operator
``FlagTrain/src/flag_train/megatron/scaled_upper_triang_masked_softmax_backward.py``
ports -- whose whole body is one call into Megatron-LM's Apex CUDA extension
``scaled_upper_triang_masked_softmax_cuda``::

    softmax_results, scale_t = ctx.saved_tensors
    input_grads = scaled_upper_triang_masked_softmax_cuda.backward(
        output_grads, softmax_results, scale_t[0]
    )
    return input_grads, None

The baseline is Megatron's ``ScaledUpperTriangMaskedSoftmax`` when megatron-core
provides it, and otherwise the same body vendored below against the same
extension. The Apex extension is a hard dependency. Without it this module
raises instead of measuring the benchmark's torch composition, because the
record is meant to describe Megatron's fused kernel.

The operands are passed directly rather than obtained by differentiating a
forward: the operator is the backward kernel. Its output is ``input_grads``;
upstream's ``None`` placeholder for ``scale`` is dropped. ``ctx`` is stood in
for by ``_SavedForBackward``, which holds the ``(softmax_results, scale_t)``
pair the upstream forward saves, so no forward pass lands in the timed region.

Shapes, the dtype sweep and the single scale are copied from
``FlagTrain/benchmark/megatron/test_scaled_upper_triang_masked_softmax_backward.py``
(``_SHAPES``, ``_DTYPES`` and ``_SCALE``).
"""

import importlib.util
import warnings

import torch

OP_NAME = "scaled_upper_triang_masked_softmax_backward"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False

# benchmark/megatron/test_scaled_upper_triang_masked_softmax_backward.py::_SHAPES
# [attn_batches, sq, sk]; sq == sk because the mask is upper-triangular.
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

_SCALE = 0.125
_EXTENSION = "scaled_upper_triang_masked_softmax_cuda"


def _require_extension():
    if importlib.util.find_spec(_EXTENSION) is None:
        raise RuntimeError(
            "scaled_upper_triang_masked_softmax_backward baseline requires "
            f"Megatron-LM's Apex CUDA extension '{_EXTENSION}', which is not "
            "installed. The record describes that fused kernel, so this module "
            "does not fall back to a torch composition. Install Megatron-LM's "
            "Apex extensions on this machine."
        )


class _SavedForBackward:
    """Stand-in for the context ``ScaledUpperTriangMaskedSoftmax.backward`` takes.

    Megatron's forward saves ``(softmax_results, scale_t)``; the backward only
    reads ``ctx.saved_tensors``, so a two-attribute object runs the copied body
    -- and the installed one -- without a forward pass in the timed region.
    """

    def __init__(self, softmax_results, scale):
        self.saved_tensors = (softmax_results, torch.tensor([scale]))


def _vendored_backward(output_grads, softmax_results, scale):
    """``ScaledUpperTriangMaskedSoftmax.backward``'s body."""
    import scaled_upper_triang_masked_softmax_cuda

    softmax_results, scale_t = _SavedForBackward(softmax_results, scale).saved_tensors
    return scaled_upper_triang_masked_softmax_cuda.backward(
        output_grads, softmax_results, scale_t[0]
    )


_SCALED_UPPER_TRIANG_MASKED_SOFTMAX = None
_NATIVE = None


def _class_backward(output_grads, softmax_results, scale):
    """``ScaledUpperTriangMaskedSoftmax.backward`` behind the saved-tensor context.

    Upstream answers one value per forward input -- ``(input_grads, None)``, the
    ``None`` standing for ``scale``, which has no gradient. Here only the
    gradient is the operator's output.
    """
    return _SCALED_UPPER_TRIANG_MASKED_SOFTMAX.backward(
        _SavedForBackward(softmax_results, scale), output_grads
    )[0]


def _megatron_backward():
    """Megatron's baseline backward, resolved once on first use.

    Imported lazily: ``megatron.core``'s ``__init__`` pulls in the whole core
    tree, which the ops-discovery pass must not trigger.
    """
    global _NATIVE, _SCALED_UPPER_TRIANG_MASKED_SOFTMAX
    if _NATIVE is None:
        _require_extension()
        try:
            from megatron.core.fusions.fused_softmax import (
                ScaledUpperTriangMaskedSoftmax,
            )

            _SCALED_UPPER_TRIANG_MASKED_SOFTMAX = ScaledUpperTriangMaskedSoftmax
            _NATIVE = _class_backward
        except Exception as exc:  # megatron-core absent, or without the class
            warnings.warn(
                "megatron.core.fusions.fused_softmax.ScaledUpperTriangMasked"
                f"Softmax could not be imported ({exc!r}); using the vendored "
                "body against the same Apex extension.",
                RuntimeWarning,
                stacklevel=2,
            )
            _NATIVE = _vendored_backward
    return _NATIVE


def torch_op(grad_output, output, scale):
    """Megatron's scaled upper-triangular masked softmax gradient, on the Apex kernel."""
    return _megatron_backward()(grad_output, output, scale)


def native():
    # Resolve eagerly so an unavailable extension or megatron-core fails here
    # instead of being discovered mid-collection.
    _megatron_backward()
    return torch_op


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    """``(grad_output, output, scale)``; ``output`` is a plausible forward result."""
    shape = binding["shape"]
    # A dense softmax: non-negative rows that sum to one. The kernel only reads
    # the kept prefix of each row, so the masked half's values do not matter.
    output = torch.softmax(torch.randn(shape, dtype=dtype, device=device), dim=-1)
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    return (grad_output, output, _SCALE), {}


def key_shape(binding):
    # The first tensor's shape determines the operator; everything else follows.
    return list(binding["shape"])


def config(binding, dtype):
    attn_batches, sq, sk = binding["shape"]
    return {
        "inputs": {
            "grad_output": {"shape": list(binding["shape"]), "dtype": str(dtype)},
            "output": {"shape": list(binding["shape"]), "dtype": str(dtype)},
            "scale": {"scalar": _SCALE},
        },
        "dims": {"attn_batches": attn_batches, "sq": sq, "sk": sk},
    }
