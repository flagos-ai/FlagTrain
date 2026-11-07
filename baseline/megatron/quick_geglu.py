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

"""NVIDIA baseline for quick_geglu (forward).

Self-contained module; depends only on torch and megatron. ``native()`` returns
Megatron-LM's eager ``megatron.core.fusions.fused_bias_geglu.quick_geglu``
(Megatron's ``jit_fuser``, i.e. ``torch.compile``, is disabled so the collector
profiles ATen kernels rather than Inductor output); when Megatron is
unavailable it raises instead of collecting a baseline or falling back to a
torch composition. Shapes and dtypes are copied from
``FlagTrain/benchmark/megatron/test_quick_geglu.py``.
"""

import math

import torch

OP_NAME = "quick_geglu"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False

# benchmark/megatron/test_quick_geglu.py::_CORE_SHAPES
_SHAPES = [
    (2,),
    (1024,),
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (1024, 1024, 1024),
]


def base_op():
    """Megatron-LM quick_geglu (eager); raises when Megatron is unavailable."""
    try:
        import megatron.core.jit as megatron_jit

        # Megatron's @jit_fuser is torch.compile by default. The collector must
        # profile Megatron's eager kernels rather than Inductor output, so
        # disable the fuser before fused_bias_geglu binds the decorator.
        megatron_jit.disable_jit_fuser()
        import megatron.core.fusions.fused_bias_geglu as fused_bias_geglu

        kernel = fused_bias_geglu.quick_geglu
        # fused_bias_geglu may have been imported before the fuser was
        # disabled; in that case unwrap the torch.compile wrapper instead of
        # compiling here.
        return getattr(kernel, "_torchdynamo_orig_callable", kernel)
    except Exception as exc:
        raise RuntimeError(
            "quick_geglu baseline requires Megatron-LM's "
            "megatron.core.fusions.fused_bias_geglu.quick_geglu, which could "
            f"not be imported: {exc!r}. Install megatron-core on this machine."
        ) from exc


def native():
    return base_op()


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    return (torch.randn(binding["shape"], dtype=dtype, device=device),), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    """Describe the real inputs/outputs for baseline matching."""
    shape = list(binding["shape"])
    hidden = shape[-1]
    dtype_str = str(dtype)
    return {
        "inputs": {
            "y": {"shape": shape, "dtype": dtype_str},
            # The benchmark calls quick_geglu(y); linear_offset keeps its default.
            "linear_offset": {"scalar": 0.0},
        },
        "outputs": {
            # The forward splits the last dimension in half: 2 * d -> d.
            "out": {"shape": shape[:-1] + [hidden // 2], "dtype": dtype_str},
        },
        "dims": {
            "num_tokens": math.prod(shape[:-1]),
            "hidden": hidden,
            "d": hidden // 2,
            "shape": shape,
        },
    }
