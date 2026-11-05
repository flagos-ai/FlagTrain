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
"""Performance benchmark for Megatron Quick GEGLU.

The baseline is Megatron-LM's
``megatron.core.fusions.fused_bias_geglu.quick_geglu``, the vendor
implementation this operator ports. Megatron defines it as a torch composition
(``quick_gelu(y_1) * (y_2 + linear_offset)``) wrapped in ``jit_fuser``
(``torch.compile``); the composition is the vendor's own implementation, so it
is a valid on-machine baseline. When Megatron-LM is not installed, that exact
composition is timed instead. The module-wide fixture raises Dynamo's recompile
limit above this suite's case count so the compiled baseline does not silently
fall back to eager halfway through. Timing, warmup and synchronization come
from ``benchmark/base.py`` (``triton.testing.do_bench`` in kernel mode).
"""

import warnings

import pytest
import torch

import flag_train

from .. import base

# Small, medium and large tensors, in 1-D, 2-D and 3-D layouts.
_CORE_SHAPES = [
    (2,),
    (1024,),
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (1024, 1024, 1024),
]

_COMPILE_BUDGET = 256


@pytest.fixture(scope="module", autouse=True)
def _full_compile_budget():
    config = torch._dynamo.config
    names = [
        name
        for name in ("recompile_limit", "cache_size_limit")
        if hasattr(config, name)
    ]
    saved = {name: getattr(config, name) for name in names}
    for name in names:
        setattr(config, name, max(saved[name], _COMPILE_BUDGET))
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(config, name, value)


def quick_geglu_ref(y, linear_offset=0.0):
    """Megatron-LM's ``quick_geglu`` body; timed when Megatron is unavailable."""
    y_1, y_2 = torch.chunk(y, 2, dim=-1)
    return y_1 * torch.sigmoid(1.702 * y_1) * (y_2 + linear_offset)


def _load_mcore_quick_geglu():
    """Megatron's ``quick_geglu``, or the composition it is defined as."""
    try:
        from megatron.core.fusions.fused_bias_geglu import quick_geglu
    except ImportError:
        warnings.warn(
            "Megatron-LM is unavailable; timing its torch composition as the "
            "benchmark baseline",
            RuntimeWarning,
            stacklevel=2,
        )
        return quick_geglu_ref, "torch composition of megatron-core quick_geglu"

    chosen = quick_geglu

    def torch_op(*args, **kwargs):
        nonlocal chosen
        if chosen is quick_geglu_ref:
            return chosen(*args, **kwargs)
        try:
            return chosen(*args, **kwargs)
        except Exception:
            warnings.warn(
                "Megatron quick_geglu failed to compile on this platform; "
                "falling back to its torch composition",
                RuntimeWarning,
                stacklevel=2,
            )
            chosen = quick_geglu_ref
            return chosen(*args, **kwargs)

    return torch_op, "megatron-core quick_geglu (@jit_fuser)"


torch_op, _BASELINE = _load_mcore_quick_geglu()


def _dtypes():
    dtypes = [torch.float16, torch.float32]
    if flag_train.runtime.device.support_bf16:
        dtypes.append(torch.bfloat16)
    return dtypes


class QuickGegluBenchmark(base.GenericBenchmark):
    """Shapes are fixed here because ``benchmark/core_shapes.yaml`` does not exist yet.

    ``Benchmark.set_shapes`` opens that file, so an override is required for an
    operator that is not listed in one. This matches the other benchmarks that
    keep their shapes in the test module.
    """

    DEFAULT_SHAPES = _CORE_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_CORE_SHAPES)


def quick_geglu_input_fn(shape, dtype, device):
    yield (torch.randn(shape, dtype=dtype, device=device),)


def _run():
    print(f"\nBaseline: {_BASELINE}")

    bench = QuickGegluBenchmark(
        input_fn=quick_geglu_input_fn,
        op_name="quick_geglu",
        torch_op=torch_op,
        dtypes=_dtypes(),
    )
    bench.set_train(flag_train.megatron.quick_geglu)
    bench.run()


@pytest.mark.quick_geglu
def test_quick_geglu_perf():
    _run()
