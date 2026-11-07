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

The preferred baseline is Megatron-LM's official
``megatron.core.fusions.fused_bias_geglu.quick_geglu`` interface. If
Megatron-LM is not installed, the benchmark logs a warning and falls back to
the equivalent PyTorch composition. Timing, warmup and synchronization come
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
    y_1, y_2 = torch.chunk(y, 2, dim=-1)
    return y_1 * torch.sigmoid(1.702 * y_1) * (y_2 + linear_offset)


def base_op():
    try:
        from megatron.core.fusions.fused_bias_geglu import quick_geglu
    except ImportError:
        warnings.warn(
            "Megatron-LM is unavailable; using the PyTorch Quick GEGLU "
            "composition as the correctness reference",
            RuntimeWarning,
            stacklevel=2,
        )
        return quick_geglu_ref

    chosen = quick_geglu

    def reference(*args, **kwargs):
        nonlocal chosen
        if chosen is quick_geglu_ref:
            return chosen(*args, **kwargs)
        try:
            return chosen(*args, **kwargs)
        except Exception:
            warnings.warn(
                "Megatron quick_geglu failed to compile on this platform; "
                "falling back to the PyTorch Quick GEGLU composition",
                RuntimeWarning,
                stacklevel=2,
            )
            chosen = quick_geglu_ref
            return chosen(*args, **kwargs)

    return reference


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
    bench = QuickGegluBenchmark(
        input_fn=quick_geglu_input_fn,
        op_name="quick_geglu",
        torch_op=base_op(),
        dtypes=_dtypes(),
    )
    bench.set_train(flag_train.megatron.quick_geglu)
    bench.run()


@pytest.mark.quick_geglu
def test_quick_geglu_perf():
    _run()
