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
"""Performance benchmark for bias_geglu_impl.

The baseline is Megatron-LM's ``bias_geglu_impl`` when it can be imported. Its
forward helpers use ``jit_fuser``, which is ``torch.compile`` on supported
PyTorch versions. Without Megatron, ``torch_op`` is ``None`` and no on-machine
baseline is timed -- ``--base-data`` is then the only comparison, rather than a
vendored torch composition. The ``Baseline:`` line printed by the test records
which one ran.

Timing, warmup and synchronization come from ``benchmark/base.py``
(``triton.testing.do_bench`` in kernel mode). ``--level core`` runs the shapes
below. The operator is forward-only, so there are no backward cases.

The measured path is the fused Triton fast path: contiguous active-device
floating inputs with a same-dtype bias. CPU staging, non-contiguous or
broadcast-expanded layouts and mixed/integer/bool promotion are covered by the
functional suite and are not performance targets here.

``torch.compile`` recompiles per shape/dtype pattern. The default Dynamo
recompile limit is smaller than this benchmark's case count, so the
module-scoped fixture below raises it; otherwise the baseline could silently
fall back to eager halfway through the table.
"""

import pytest
import torch

import flag_train

from .. import base, consts

# Token counts and hidden widths used by a fused MLP activation. Last dim is even.
_CORE_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (16384, 4096),
    (2048, 11008),
    (16, 128, 8192),
    (256, 32768),
]


def _dtypes():
    """The floating dtypes the fused fast path is benchmarked with."""
    dtypes = [dtype for dtype in consts.FLOAT_DTYPES if dtype != torch.bfloat16]
    if flag_train.runtime.device.support_bf16:
        dtypes.append(torch.bfloat16)
    return dtypes


class BiasGegluImplBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _CORE_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_CORE_SHAPES)


def bias_geglu_input_fn(shape, dtype, device):
    # Same-dtype, contiguous operands with a [hidden] bias: the kernel fast path.
    x = torch.randn(shape, dtype=dtype, device=device)
    bias = torch.randn((shape[-1],), dtype=dtype, device=device)
    yield (x, bias)


def base_op():
    try:
        from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
    except Exception:
        return None
    return bias_geglu_impl


# Enough variants for 7 shapes x 3 dtypes of forward cases without Dynamo
# falling back to eager halfway through the table.
_DYNAMO_RECOMPILE_LIMIT = 128


@pytest.fixture(scope="module", autouse=True)
def _configure_torch_compile():
    """Keep the compiled baseline working across this module's case count."""
    import torch._dynamo
    import torch._inductor.config as inductor_config

    # One compile thread keeps Inductor's compilation in-process; vendor Triton
    # builds may fail to pickle the compiled kernel back from a worker.
    saved_threads = inductor_config.compile_threads
    inductor_config.compile_threads = 1
    saved_autotune = inductor_config.triton.autotune_pointwise
    # torch_musa's Inductor autotune timer imports
    # triton.backends.mtgpu.musa_testing, which the FlagTree Triton build does
    # not ship. Disable the pointwise config search on MUSA; the compiled
    # baseline itself still runs.
    if flag_train.vendor_name == "mthreads":
        inductor_config.triton.autotune_pointwise = False

    dynamo = torch._dynamo.config
    limit_name = next(
        (
            name
            for name in ("recompile_limit", "cache_size_limit")
            if hasattr(dynamo, name)
        ),
        None,
    )
    saved_limit = getattr(dynamo, limit_name) if limit_name else None
    if limit_name is not None:
        setattr(dynamo, limit_name, max(saved_limit, _DYNAMO_RECOMPILE_LIMIT))

    try:
        yield
    finally:
        inductor_config.compile_threads = saved_threads
        inductor_config.triton.autotune_pointwise = saved_autotune
        if limit_name is not None:
            setattr(dynamo, limit_name, saved_limit)


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_perf():
    """Benchmark GEGLU for every shape and dtype in one table."""
    torch_op = base_op()
    _BASELINE = "megatron-core bias_geglu_impl" if torch_op is not None else "none"
    print(f"\nBaseline: {_BASELINE}")

    bench = BiasGegluImplBenchmark(
        input_fn=bias_geglu_input_fn,
        op_name="bias_geglu_impl",
        torch_op=torch_op,
        dtypes=_dtypes(),
        is_backward=False,
    )
    bench.set_train(flag_train.megatron.bias_geglu_impl)
    bench.run()
