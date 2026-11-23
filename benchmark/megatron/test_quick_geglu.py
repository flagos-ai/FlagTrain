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

The baseline is Megatron-LM's official
``megatron.core.fusions.fused_bias_geglu.quick_geglu`` interface. When
Megatron-LM is unavailable, the benchmark has no native baseline (``None``)
instead of falling back to a PyTorch composition. Timing, warmup and
synchronization come from ``benchmark/base.py`` (``triton.testing.do_bench``
in kernel mode).
"""

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


def base_op():
    try:
        from megatron.core.fusions.fused_bias_geglu import quick_geglu
    except Exception:
        return None

    return quick_geglu


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
    torch_op = base_op()
    baseline = "megatron-core quick_geglu" if torch_op is not None else "none"
    print(f"\nBaseline: {baseline}")

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
