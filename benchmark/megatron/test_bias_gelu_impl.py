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
"""Benchmark Bias + GELU against Megatron's native GeLUFunction.apply.

Megatron's inner bias_gelu (not the outer apply) owns the torch.compile
wrapper. Precompile fixed cases before timing to avoid measuring JIT work.
NVIDIA requires compiled Megatron; other backends use the native eager
implementation if JIT is disabled or compilation fails.
"""

import importlib
import importlib.metadata
from contextlib import nullcontext

import pytest
import torch

import flag_train
from flag_train.runtime import torch_device_fn

from .. import base, conftest

_CORE_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (16384, 4096),
    (2048, 11008),
    (16, 128, 8192),
    (256, 32768),
]


def _load_megatron():
    try:
        return importlib.import_module("megatron.core.fusions.fused_bias_gelu")
    except ImportError as exc:
        raise RuntimeError(
            "bias_gelu_impl benchmark requires installed megatron-core"
        ) from exc


def _dtypes():
    dtypes = [torch.float16, torch.float32]
    if flag_train.runtime.device.support_bf16:
        dtypes.append(torch.bfloat16)
    return dtypes


class BiasGeluBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _CORE_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_CORE_SHAPES)


def bias_gelu_input_fn(shape, dtype, device):
    yield (
        torch.randn(shape, dtype=dtype, device=device),
        torch.randn(shape[-1], dtype=dtype, device=device),
    )


def _warmup(bench, torch_op):
    for dtype in bench.to_bench_dtypes:
        for shape in bench.shapes:
            args = next(bias_gelu_input_fn(shape, dtype, bench.device))
            torch_op(*args)
            torch_device_fn.synchronize()


def _run():
    module = _load_megatron()
    torch_op = module.bias_gelu_impl
    jit_wrapped = hasattr(module.bias_gelu, "_torchdynamo_orig_callable")

    bench = BiasGeluBenchmark(
        input_fn=bias_gelu_input_fn,
        op_name="bias_gelu_impl",
        torch_op=torch_op,
        dtypes=_dtypes(),
    )
    bench.set_train(flag_train.megatron.bias_gelu_impl)

    if base.Config.query or base.Config.list_cases:
        return bench.run()
    if base.Config.skip_native:
        raise RuntimeError("Megatron benchmark requires the native baseline")

    # Keep the 21 fixed shape/dtype specializations within the compile budget.
    config = torch._dynamo.config
    limits = {
        name: max(getattr(config, name), 128)
        for name in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit")
        if hasattr(config, name)
    }
    limits["suppress_errors"] = False
    if hasattr(config, "automatic_dynamic_shapes"):
        limits["automatic_dynamic_shapes"] = False

    mode = "compiled" if jit_wrapped else "eager"
    fallback_reason = None
    with config.patch(limits), torch.no_grad():
        bench.init_user_config()
        if jit_wrapped:
            from torch._dynamo.exc import BackendCompilerFailed, Unsupported
            from torch._dynamo.utils import counters

            before = counters["stats"].get("unique_graphs", 0)
            try:
                _warmup(bench, torch_op)
            except (BackendCompilerFailed, Unsupported) as exc:
                if flag_train.vendor_name == "nvidia":
                    raise RuntimeError("NVIDIA requires compiled Megatron") from exc
                mode = "eager"
                fallback_reason = f"{type(exc).__name__}: {exc}"
            else:
                if counters["stats"].get("unique_graphs", 0) <= before:
                    raise RuntimeError("Megatron did not produce a compiled graph")
        elif flag_train.vendor_name == "nvidia":
            raise RuntimeError("NVIDIA requires Megatron default torch.compile")

        if mode == "eager" and jit_wrapped:
            stance = getattr(torch.compiler, "set_stance", None)
            if stance is None:
                raise RuntimeError("Native Megatron eager fallback requires set_stance")
            execution = stance("force_eager")
        else:
            execution = nullcontext()

        print(f"\nBaseline: megatron-core {importlib.metadata.version('megatron-core')}")
        print(f"Baseline Mode: {mode}" + (
            " (Megatron default torch.compile)" if mode == "compiled" else ""
        ))
        print("Operator: bias_gelu_impl (native GeLUFunction.apply)")
        if fallback_reason:
            print(f"Fallback Reason: {fallback_reason}")
        if base.Config.record_json:
            conftest.TEST_RESULTS.setdefault(bench.op_name, {})["baseline"] = {
                "package": "megatron-core",
                "version": importlib.metadata.version("megatron-core"),
                "mode": mode,
                "jit_wrapped": jit_wrapped,
                "fallback_reason": fallback_reason,
            }

        with execution:
            if mode == "eager":
                _warmup(bench, torch_op)
            # Fail on a new graph rather than compile while measuring.
            freeze = (
                config.patch(error_on_recompile=True)
                if mode == "compiled"
                else nullcontext()
            )
            with freeze:
                return bench.run()


@pytest.mark.bias_gelu_impl
def test_bias_gelu_impl_perf():
    _run()
