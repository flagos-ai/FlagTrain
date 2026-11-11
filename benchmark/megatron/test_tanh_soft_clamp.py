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
"""Performance benchmark for the tanh_soft_clamp operator.

The baseline is Megatron-LM's ``megatron.core.activations.tanh_soft_clamp``,
the vendor implementation this operator ports. Megatron defines it as the torch
composition ``scale * torch.tanh(x / scale)`` wrapped in ``jit_fuser``
(``torch.compile``); the composition is the vendor's own implementation, so it
is a valid on-machine baseline. When Megatron-LM is unavailable or predates the
operator, that exact composition is timed instead. Timing, warmup and
synchronization come from ``benchmark/base.py`` (``triton.testing.do_bench`` in
kernel mode).
"""

import warnings

import pytest
import torch

import flag_train

from .. import base

# Small, medium, and a Megatron-MLP-sized activation tensor (tokens, hidden).
_CORE_SHAPES = [
    (1,),
    (1024,),
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (4, 8192, 4096),
    (8, 4096, 11008),
]

_SCALE = 0.5


def tanh_soft_clamp_ref(x, scale):
    """Megatron-LM's body; timed when Megatron is unavailable."""
    return (scale * torch.tanh(x.float() / scale)).to(x.dtype)


def _load_mcore_tanh_soft_clamp():
    """Megatron's ``tanh_soft_clamp``, or the composition it is defined as."""
    try:
        import megatron.core.activations
    except ImportError:
        warnings.warn(
            "Megatron-LM is unavailable; timing its torch composition as the "
            "benchmark baseline",
            RuntimeWarning,
            stacklevel=2,
        )
        return tanh_soft_clamp_ref, "torch composition of tanh_soft_clamp"

    compiled = getattr(megatron.core.activations, "tanh_soft_clamp", None)
    if compiled is None:
        warnings.warn(
            "Megatron-LM predates tanh_soft_clamp (needs megatron-core >= "
            "0.20.0); timing its torch composition as the benchmark baseline",
            RuntimeWarning,
            stacklevel=2,
        )
        return tanh_soft_clamp_ref, "torch composition of tanh_soft_clamp"
    return compiled, "megatron-core tanh_soft_clamp (@jit_fuser)"


torch_op, _BASELINE = _load_mcore_tanh_soft_clamp()


def _dtypes():
    dtypes = [torch.float16, torch.float32]
    if flag_train.runtime.device.support_bf16:
        dtypes.append(torch.bfloat16)
    return dtypes


class TanhSoftClampBenchmark(base.GenericBenchmark):

    DEFAULT_SHAPES = _CORE_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_CORE_SHAPES)


def tanh_soft_clamp_input_fn(shape, dtype, device):
    yield (torch.randn(shape, dtype=dtype, device=device), _SCALE)


@pytest.mark.tanh_soft_clamp
def test_tanh_soft_clamp_perf():
    print(f"\nBaseline: {_BASELINE}")

    bench = TanhSoftClampBenchmark(
        input_fn=tanh_soft_clamp_input_fn,
        op_name="tanh_soft_clamp",
        torch_op=torch_op,
        dtypes=_dtypes(),
    )
    bench.set_train(flag_train.megatron.tanh_soft_clamp)
    bench.run()
