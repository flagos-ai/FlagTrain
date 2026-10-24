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
"""Performance benchmark for bias_swiglu_impl / weighted_bias_swiglu_impl.

The baseline is Megatron-LM's own ``bias_swiglu_impl`` /
``weighted_bias_swiglu_impl`` when it is installed. Those functions are
PyTorch compositions decorated with ``@jit_fuser``, which is ``torch.compile``
on torch >= 2.2, so the baseline is the compiled native implementation rather
than eager PyTorch. The eager composition is only a fallback (with a warning)
when Megatron-LM is missing or does not support a variant.

Timing, warmup and synchronization come from ``benchmark/base.py``
(``triton.testing.do_bench`` in kernel mode). ``--level core`` runs the shapes
below. The plain path adds a ``[hidden]`` bias. The weighted path multiplies
by a fp32 ``[*batch, 1]`` scale and passes ``bias=None``, matching
Megatron-LM, which does not implement the weighted-with-bias combination.
Megatron-LM wants flattened ``[token, 1]`` weights, so the wrapper reshapes
``[*batch, 1]`` to ``[token, 1]`` first; that is the same view its own callers
pass. The clamped and SiTU-GLU variants are benchmarked separately so their
extra tanh/clamp work is visible instead of being folded into the plain
numbers.

``torch.compile`` recompiles per shape/dtype/requires-grad pattern. The
default Dynamo recompile limit is smaller than this benchmark's case count, so
the module raises it; otherwise the baseline silently falls back to eager and
the comparison is meaningless.
"""

import functools
import inspect
import os
import warnings

import pytest
import torch

import flag_train

from .. import base, consts

# Token counts and hidden widths used by a fused MLP activation. Last dim is even.
_CORE_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (4, 8192, 4096),
    (8, 4096, 11008),
]


def bias_swiglu_ref(
    x,
    bias=None,
    clamp_value=None,
    gate_clamp_scale=None,
    linear_clamp_scale=None,
):
    y = x if bias is None else x + bias
    gate, linear = torch.chunk(y, 2, dim=-1)
    if gate_clamp_scale is not None:
        gate_v = gate_clamp_scale * torch.tanh(gate / gate_clamp_scale)
        gate_v = gate_v * torch.sigmoid(gate)
        if linear_clamp_scale is not None:
            linear = linear_clamp_scale * torch.tanh(linear / linear_clamp_scale)
        return gate_v * linear
    if clamp_value is not None and clamp_value > 0:
        return torch.nn.functional.silu(gate.clamp(max=clamp_value)) * linear.clamp(
            min=-clamp_value, max=clamp_value
        )
    return torch.nn.functional.silu(gate) * linear


def weighted_bias_swiglu_ref(x, bias, weights, **kwargs):
    if bias is not None:
        raise NotImplementedError("Bias is not supported for weighted swiglu fusion")
    return bias_swiglu_ref(x, None, **kwargs) * weights


# Enough variants for 5 shapes x 3 dtypes x forward/backward without Dynamo
# falling back to eager halfway through the suite.
_DYNAMO_RECOMPILE_LIMIT = 128


def _missing_kwargs(impl, kwargs):
    parameters = inspect.signature(impl).parameters
    return [name for name in kwargs if name not in parameters]


def _megatron_weighted(impl, x, bias, weights, **kwargs):
    # Megatron-LM expects already-flattened ``[token, 1]`` weights; its own
    # MoE callers pass that view. Reshaping here keeps the comparison equal.
    if weights.ndim == 3:
        weights = weights.reshape(-1, 1)
    return impl(x, bias, weights, **kwargs)


def base_op(weighted=False, **op_kwargs):
    """Megatron-LM's native op when available, eager composition otherwise.

    Mirrors ``base_op`` in ``test_quick_gelu.py``: the import happens here and
    a missing Megatron-LM is a warning plus a fallback, not a hard dependency.
    """
    eager = weighted_bias_swiglu_ref if weighted else bias_swiglu_ref
    name = "weighted_bias_swiglu_impl" if weighted else "bias_swiglu_impl"
    try:
        import megatron.core.fusions.fused_bias_swiglu
    except ImportError:
        warnings.warn(
            "Megatron-LM is unavailable; using the eager PyTorch composition "
            "as the benchmark baseline",
            RuntimeWarning,
            stacklevel=2,
        )
        return functools.partial(eager, **op_kwargs)
    impl = getattr(megatron.core.fusions.fused_bias_swiglu, name, None)
    if impl is None:
        warnings.warn(
            f"Megatron-LM has no {name}; using the eager PyTorch composition "
            "as the benchmark baseline",
            RuntimeWarning,
            stacklevel=2,
        )
        return functools.partial(eager, **op_kwargs)
    missing = _missing_kwargs(impl, op_kwargs)
    if missing:
        warnings.warn(
            f"installed Megatron-LM {name} does not accept {missing}; using "
            "the eager PyTorch composition as the benchmark baseline",
            RuntimeWarning,
            stacklevel=2,
        )
        return functools.partial(eager, **op_kwargs)
    if weighted:
        return functools.partial(_megatron_weighted, impl, **op_kwargs)
    return functools.partial(impl, **op_kwargs)


def _stabilize_megatron_compile():
    """Keep the compiled Megatron baseline working on every backend.

    ``megatron.core.jit`` sets ``jit_fuser = torch.compile`` on torch >= 2.2,
    so the native baseline recompiles per shape, dtype and requires-grad
    pattern; the default Dynamo limit is below this suite's case count. And
    Inductor's async compile has to pickle the compiled Triton kernel back
    from its worker, which some vendor Triton builds cannot do.
    """
    try:
        from megatron.core import jit
    except ImportError:
        return
    if jit.jit_fuser is not torch.compile:
        return

    # Inductor's async compile sends each Triton kernel to a worker process and
    # pickles the compiled kernel back. Vendor Triton builds whose JITFunction
    # cannot be pickled (it carries an RLock) fail there with
    # "TypeError: cannot pickle '_thread.RLock' object" while compiling the
    # Megatron baseline. One compile thread keeps compilation in-process; it
    # changes compile time only, not the measured kernel time.
    try:
        from torch._inductor import config as inductor_config
    except ImportError:
        inductor_config = None
    if (
        inductor_config is not None
        and getattr(inductor_config, "compile_threads", 1) != 1
    ):
        inductor_config.compile_threads = 1

    try:
        import torch._dynamo as dynamo
    except ImportError:
        return
    config = dynamo.config
    if hasattr(config, "recompile_limit"):
        current = config.recompile_limit
    elif hasattr(config, "cache_size_limit"):
        current = config.cache_size_limit
    else:
        return
    if current >= _DYNAMO_RECOMPILE_LIMIT:
        return
    if hasattr(config, "recompile_limit"):
        config.recompile_limit = _DYNAMO_RECOMPILE_LIMIT
    else:
        config.cache_size_limit = _DYNAMO_RECOMPILE_LIMIT
    warnings.warn(
        "raised the torch._dynamo recompile limit to "
        f"{_DYNAMO_RECOMPILE_LIMIT} so the torch.compile baseline does not "
        "silently fall back to eager",
        RuntimeWarning,
        stacklevel=2,
    )


_stabilize_megatron_compile()

_CLAMPED_KWARGS = {"clamp_value": 2.0}
_SITU_KWARGS = {"gate_clamp_scale": 4.0, "linear_clamp_scale": 2.5}


def _dtypes():
    """``consts.FLOAT_DTYPES`` with bf16 dropped where the backend has none.

    Same rule as ``utils.FLOAT_DTYPES`` in the correctness suite, so both see
    the same dtypes on every backend.
    """
    dtypes = [dtype for dtype in consts.FLOAT_DTYPES if dtype != torch.bfloat16]
    if flag_train.runtime.device.support_bf16:
        dtypes.append(torch.bfloat16)
    return dtypes


class _FixedShapeBenchmark(base.GenericBenchmark):
    """Shapes are fixed here because ``benchmark/core_shapes.yaml`` does not exist yet.

    ``Benchmark.set_shapes`` opens that file, so an override is required for an
    operator that is not listed in one. This matches ``QuickGeluBenchmark``.
    """

    DEFAULT_SHAPES = _CORE_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_CORE_SHAPES)


def bias_swiglu_input_fn(shape, dtype, device):
    yield (
        torch.randn(shape, dtype=dtype, device=device),
        torch.randn((shape[-1],), dtype=dtype, device=device),
    )


def weighted_bias_swiglu_input_fn(shape, dtype, device):
    weight_shape = shape[:-1] + (1,)
    yield (
        torch.randn(shape, dtype=dtype, device=device),
        None,
        torch.randn(weight_shape, dtype=torch.float32, device=device),
    )


def _run(is_backward, input_fn, torch_op, train_op, op_name, op_kwargs=None):
    # ``torch_op`` is already a configured baseline (native or fallback);
    # ``op_kwargs`` only selects the variant of the operator under test.
    if op_kwargs:
        train_op = functools.partial(train_op, **op_kwargs)
    bench = _FixedShapeBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=_dtypes(),
        is_backward=is_backward,
    )
    bench.set_train(train_op)
    # On Ascend, enable parallel scheduling of all blocks to avoid excessive logical cores
    # and reduce scheduling overhead when the logical core count exceeds 65,535.
    # See https://github.com/flagos-ai/FlagGems/pull/7304
    if flag_train.vendor_name == "ascend":
        os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"
    bench.run()


@pytest.mark.bias_swiglu_impl
def test_bias_swiglu_impl_perf():
    _run(
        False,
        bias_swiglu_input_fn,
        base_op(),
        flag_train.megatron.bias_swiglu_impl,
        "bias_swiglu_impl",
    )


@pytest.mark.bias_swiglu_impl
def test_bias_swiglu_impl_backward_perf():
    _run(
        True,
        bias_swiglu_input_fn,
        base_op(),
        flag_train.megatron.bias_swiglu_impl,
        "bias_swiglu_impl_backward",
    )


@pytest.mark.bias_swiglu_impl
def test_weighted_bias_swiglu_impl_perf():
    _run(
        False,
        weighted_bias_swiglu_input_fn,
        base_op(weighted=True),
        flag_train.megatron.weighted_bias_swiglu_impl,
        "weighted_bias_swiglu_impl",
    )


@pytest.mark.bias_swiglu_impl
def test_weighted_bias_swiglu_impl_backward_perf():
    _run(
        True,
        weighted_bias_swiglu_input_fn,
        base_op(weighted=True),
        flag_train.megatron.weighted_bias_swiglu_impl,
        "weighted_bias_swiglu_impl_backward",
    )


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("is_backward", [False, True], ids=["forward", "backward"])
def test_bias_swiglu_impl_clamped_perf(is_backward):
    _run(
        is_backward,
        bias_swiglu_input_fn,
        base_op(**_CLAMPED_KWARGS),
        flag_train.megatron.bias_swiglu_impl,
        (
            "bias_swiglu_impl_clamped_backward"
            if is_backward
            else "bias_swiglu_impl_clamped"
        ),
        op_kwargs=_CLAMPED_KWARGS,
    )


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("is_backward", [False, True], ids=["forward", "backward"])
def test_bias_swiglu_impl_situ_perf(is_backward):
    _run(
        is_backward,
        bias_swiglu_input_fn,
        base_op(**_SITU_KWARGS),
        flag_train.megatron.bias_swiglu_impl,
        "bias_swiglu_impl_situ_backward" if is_backward else "bias_swiglu_impl_situ",
        op_kwargs=_SITU_KWARGS,
    )
