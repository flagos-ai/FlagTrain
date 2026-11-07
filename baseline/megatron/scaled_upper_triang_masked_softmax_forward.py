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

"""NVIDIA baseline for scaled_upper_triang_masked_softmax_forward.

Self-contained module; depends only on torch and megatron, and does not import
``flag_train`` or ``benchmark``.

``native()`` returns Megatron-LM's ``ScaledUpperTriangMaskedSoftmax.forward``
(``megatron/core/fusions/fused_softmax.py``) -- the operator
``FlagTrain/src/flag_train/megatron/scaled_upper_triang_masked_softmax_forward.py``
ports -- whose whole body is one call into Megatron-LM's Apex CUDA extension
``scaled_upper_triang_masked_softmax_cuda``::

    scale_t = torch.tensor([scale])
    return scaled_upper_triang_masked_softmax_cuda.forward(inputs, scale_t[0])

The baseline is Megatron's ``ScaledUpperTriangMaskedSoftmax`` when megatron-core
provides it, and otherwise the same body vendored below against the same
extension. The Apex extension is a hard dependency. Without it this module
raises instead of measuring a torch composition, because the record is meant to
describe Megatron's fused kernel.

Shapes, the dtype sweep and the single scale are copied from
``FlagTrain/benchmark/megatron/test_scaled_upper_triang_masked_softmax_forward.py``
(``_SHAPES``, ``_DTYPES`` and ``_SCALE``).
"""

import importlib.util
import warnings

import torch

OP_NAME = "scaled_upper_triang_masked_softmax_forward"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False

# benchmark/megatron/test_scaled_upper_triang_masked_softmax_forward.py::_SHAPES
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
            "scaled_upper_triang_masked_softmax_forward baseline requires "
            f"Megatron-LM's Apex CUDA extension '{_EXTENSION}', which is not "
            "installed. The record describes that fused kernel, so this module "
            "does not fall back to a torch composition. Install Megatron-LM's "
            "Apex extensions on this machine."
        )


def _vendored_forward(inputs, scale):
    """``ScaledUpperTriangMaskedSoftmax.forward``'s body."""
    import scaled_upper_triang_masked_softmax_cuda

    scale_t = torch.tensor([scale])
    return scaled_upper_triang_masked_softmax_cuda.forward(inputs, scale_t[0])


_NATIVE = None


def _megatron_forward():
    """Megatron's baseline forward, resolved once on first use.

    Imported lazily: ``megatron.core``'s ``__init__`` pulls in the whole core
    tree, which the ops-discovery pass must not trigger.
    """
    global _NATIVE
    if _NATIVE is None:
        _require_extension()
        try:
            from megatron.core.fusions.fused_softmax import (
                ScaledUpperTriangMaskedSoftmax,
            )

            _NATIVE = ScaledUpperTriangMaskedSoftmax.apply
        except Exception as exc:  # megatron-core absent, or without the class
            warnings.warn(
                "megatron.core.fusions.fused_softmax.ScaledUpperTriangMasked"
                f"Softmax could not be imported ({exc!r}); using the vendored "
                "body against the same Apex extension.",
                RuntimeWarning,
                stacklevel=2,
            )
            _NATIVE = _vendored_forward
    return _NATIVE


def torch_op(inputs, scale):
    """Megatron's scaled upper-triangular masked softmax, on the Apex kernel."""
    return _megatron_forward()(inputs, scale)


def native():
    # Resolve eagerly so an unavailable extension or megatron-core fails here
    # instead of being discovered mid-collection.
    _megatron_forward()
    return torch_op


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    """``(inputs, scale)`` exactly as the benchmark's input function builds them."""
    inputs = torch.randn(binding["shape"], dtype=dtype, device=device)
    return (inputs, _SCALE), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    attn_batches, sq, sk = binding["shape"]
    return {
        "inputs": {
            "inputs": {"shape": list(binding["shape"]), "dtype": str(dtype)},
            "scale": {"scalar": _SCALE},
        },
        "dims": {"attn_batches": attn_batches, "sq": sq, "sk": sk},
    }
