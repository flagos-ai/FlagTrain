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

"""NVIDIA baseline for scaled_masked_softmax_cuda_forward.

Self-contained module; depends only on torch and megatron, and does not import
``flag_train`` or ``benchmark``.

``native()`` returns Megatron-LM's ``ScaledMaskedSoftmax.forward``
(``megatron/core/fusions/fused_softmax.py``) -- the operator
``FlagTrain/src/flag_train/megatron/scaled_masked_softmax_cuda_forward.py``
ports -- whose whole body is one call into Megatron-LM's Apex CUDA extension
``scaled_masked_softmax_cuda``::

    scale_t = torch.tensor([scale])
    return scaled_masked_softmax_cuda.forward(inputs, mask, scale_t[0])

The baseline is selected the same way as in the FlagTrain benchmark: the
installed ``ScaledMaskedSoftmax`` when megatron-core is >= 0.20, and otherwise
the same body vendored below against the same extension. The Apex extension is
a hard dependency. Without it this module raises instead of measuring the
benchmark's ``masked_softmax_ref`` torch composition, because the record is
meant to describe Megatron's fused kernel.

Shapes, dtypes and the input construction are copied from
``FlagTrain/benchmark/megatron/test_scaled_masked_softmax_cuda_forward.py``
(``_SHAPES`` and ``masked_forward_input_fn``).
"""

import importlib.metadata
import importlib.util
import re

import torch

OP_NAME = "scaled_masked_softmax_cuda_forward"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False

# benchmark/megatron/test_scaled_masked_softmax_cuda_forward.py::_SHAPES
# [b, np, sq, sk] scores; the mask is [b, 1, sq, sk] bool.
_SHAPES = [
    (8, 16, 128, 128),
    (8, 16, 256, 256),
    (8, 16, 512, 512),
    (8, 16, 1024, 1024),
    (4, 16, 2048, 2048),
    (4, 16, 4096, 4096),
]

_SCALE = 0.125
_EXTENSION = "scaled_masked_softmax_cuda"

# The version from which megatron-core's own ``ScaledMaskedSoftmax`` is the
# baseline, as in the benchmark.
_MIN_NATIVE_VERSION = (0, 20)


def _version_tuple(version):
    """``(major, minor)`` of a distribution version, or ``None`` without one."""
    match = re.match(r"\s*(\d+)(?:\.(\d+))?", version)
    return None if match is None else (int(match.group(1)), int(match.group(2) or 0))


def _megatron_version():
    for distribution in ("megatron-core", "megatron"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _require_extension():
    if importlib.util.find_spec(_EXTENSION) is None:
        raise RuntimeError(
            "scaled_masked_softmax_cuda_forward baseline requires Megatron-LM's "
            f"Apex CUDA extension '{_EXTENSION}', which is not installed. The "
            "record describes that fused kernel, so this module does not fall "
            "back to the benchmark's masked_softmax_ref (torch) composition. "
            "Install Megatron-LM's Apex extensions on this machine."
        )


def _vendored_forward(inputs, mask, scale):
    """``ScaledMaskedSoftmax.forward``'s body, for an older megatron-core."""
    import scaled_masked_softmax_cuda

    scale_t = torch.tensor([scale])
    return scaled_masked_softmax_cuda.forward(inputs, mask, scale_t[0])


_NATIVE = None


def _megatron_forward():
    """Megatron's baseline forward, resolved once on first use.

    Imported lazily: ``megatron.core``'s ``__init__`` pulls in the whole core
    tree, which the ops-discovery pass must not trigger.
    """
    global _NATIVE
    if _NATIVE is None:
        _require_extension()
        version = _megatron_version()
        current = _version_tuple(version) if version is not None else None
        if current is None or current < _MIN_NATIVE_VERSION:
            _NATIVE = _vendored_forward
        else:
            try:
                from megatron.core.fusions.fused_softmax import ScaledMaskedSoftmax
            except Exception as exc:
                raise RuntimeError(
                    f"megatron-core {version} should provide "
                    "megatron.core.fusions.fused_softmax.ScaledMaskedSoftmax, "
                    f"but importing it failed: {exc!r}"
                ) from exc
            _NATIVE = ScaledMaskedSoftmax.apply
    return _NATIVE


def torch_op(inputs, mask, scale):
    """Megatron's scaled masked softmax forward, on the Apex CUDA kernel."""
    return _megatron_forward()(inputs, mask, scale)


def native():
    # Resolve eagerly so an unavailable extension or megatron-core fails here
    # instead of being discovered mid-collection.
    _megatron_forward()
    return torch_op


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    """``(inputs, mask, scale)`` exactly as ``masked_forward_input_fn`` builds them."""
    b, _, sq, sk = binding["shape"]
    inputs = torch.randn(binding["shape"], dtype=dtype, device=device)
    # Hide the second half of the keys: a padding mask, as in the benchmark.
    mask = torch.zeros((b, 1, sq, sk), dtype=torch.bool, device=device)
    mask[..., sk // 2 :] = True
    return (inputs, mask, _SCALE), {}


def key_shape(binding):
    # The first tensor's shape determines the operator; the mask follows from it.
    return list(binding["shape"])


def config(binding, dtype):
    b, np_, sq, sk = binding["shape"]
    return {
        "inputs": {
            "inputs": {"shape": list(binding["shape"]), "dtype": str(dtype)},
            "mask": {"shape": [b, 1, sq, sk], "dtype": "torch.bool"},
            "scale": {"scalar": _SCALE},
        },
        "dims": {"b": b, "np": np_, "sq": sq, "sk": sk},
    }
