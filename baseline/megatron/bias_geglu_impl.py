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

"""NVIDIA baseline for bias_geglu_impl.

Self-contained module; depends only on torch and megatron, and does not import
``flag_train`` or ``benchmark``. ``native()`` returns Megatron-LM's
``megatron.core.fusions.fused_bias_geglu.bias_geglu_impl`` and raises when
Megatron is unavailable instead of falling back to a torch composition.

Shapes and dtypes are copied from
``FlagTrain/benchmark/megatron/test_bias_geglu_impl.py`` (``_CORE_SHAPES`` and
``_dtypes``). The last input dimension is the concatenated ``[gate | linear]``,
so the output keeps half of it. The bias is the ``skip_bias_add`` ``[hidden]``
tensor added to both halves. Only the standard GEGLU activation is collected;
the QuickGEGLU variants are additional FlagTrain benchmark cases and are not
arguments of Megatron-LM's public ``bias_geglu_impl``.

Megatron decorates the inner helpers of that module with ``@jit_fuser``, which
is ``torch.compile`` on supported PyTorch versions. This module disables the
fuser so the collector profiles the ATen kernels rather than Inductor output.
Because ``import megatron.core`` may already have imported and wrapped the
fusion helpers, the module is reloaded once after the fuser is disabled.
Megatron is imported lazily on the first ``native()`` call, so operator
discovery does not require Megatron to be installed.
"""

import math

import torch

OP_NAME = "bias_geglu_impl"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False

# benchmark/megatron/test_bias_geglu_impl.py::_CORE_SHAPES
# (tokens, 2 * hidden); the output shape is [..., hidden].
_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (16384, 4096),
    (2048, 11008),
    (16, 128, 8192),
    (256, 32768),
]

_MEGATRON_BIAS_GEGLU_IMPL = None


def _megatron_bias_geglu_impl():
    """Megatron-LM's bias_geglu_impl (eager); raise when it cannot be imported."""
    global _MEGATRON_BIAS_GEGLU_IMPL
    if _MEGATRON_BIAS_GEGLU_IMPL is None:
        try:
            import importlib

            import megatron.core.jit

            # The collector profiles native kernels, not Inductor output.
            megatron.core.jit.disable_jit_fuser()
            import megatron.core.fusions.fused_bias_geglu

            # megatron.core imports the fusions package at import time, so the
            # helpers may already be wrapped by torch.compile. Reloading after
            # disable_jit_fuser() rebinds @jit_fuser to the no-op decorator.
            if hasattr(
                megatron.core.fusions.fused_bias_geglu.geglu,
                "_torchdynamo_orig_callable",
            ):
                importlib.reload(megatron.core.fusions.fused_bias_geglu)
        except Exception as exc:
            raise RuntimeError(
                "bias_geglu_impl baseline requires Megatron-LM's "
                "megatron.core.fusions.fused_bias_geglu.bias_geglu_impl, "
                f"which could not be imported: {exc!r}. Install megatron-core "
                "on this machine."
            ) from exc
        _MEGATRON_BIAS_GEGLU_IMPL = (
            megatron.core.fusions.fused_bias_geglu.bias_geglu_impl
        )
    return _MEGATRON_BIAS_GEGLU_IMPL


def native():
    # Resolve eagerly so an unavailable Megatron fails here instead of being
    # discovered mid-collection.
    return _megatron_bias_geglu_impl()


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    hidden = binding["shape"][-1]
    input = torch.randn(binding["shape"], dtype=dtype, device=device)
    bias = torch.randn((hidden,), dtype=dtype, device=device)
    return (input, bias), {}


def key_shape(binding):
    # First tensor: the [tokens, 2 * hidden] input; unique per deployment.
    return list(binding["shape"])


def config(binding, dtype):
    """Describe the real inputs/outputs for baseline matching."""
    shape = list(binding["shape"])
    hidden = shape[-1]
    dtype_str = str(dtype)
    return {
        "inputs": {
            "input": {"shape": shape, "dtype": dtype_str},
            # One [2 * d] bias vector is shared by all rows.
            "bias": {"shape": [hidden], "dtype": dtype_str},
        },
        "outputs": {
            # The forward splits the last dimension in half: 2 * d -> d.
            "out": {
                "shape": shape[:-1] + [hidden // 2],
                "dtype": dtype_str,
            },
        },
        "dims": {
            "num_tokens": math.prod(shape[:-1]),
            "hidden": hidden,
            "d": hidden // 2,
            "shape": shape,
        },
    }
