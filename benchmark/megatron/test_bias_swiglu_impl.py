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
"""Performance benchmark for bias_swiglu_impl.

The baseline is Megatron-LM's own ``bias_swiglu_impl`` when it is installed.
That function is a PyTorch composition decorated with ``@jit_fuser``, which is
``torch.compile`` on torch >= 2.2, so the baseline is the compiled native
implementation rather than eager PyTorch. When the installed Megatron-LM
predates the clamp / SiTU variants, the main-branch implementation vendored in
the appendix at the end of this file is used instead, so those cases are still
compared against a compiled baseline.

Timing, warmup and synchronization come from ``benchmark/base.py``
(``triton.testing.do_bench`` in kernel mode). ``--level core`` runs the shapes
below. Each test covers one dtype and prints the three variants back to back, so
a report can be read dtype by dtype. The operator is forward-only, so there are
no backward cases. The plain path adds a ``[hidden]`` bias. The clamped and
SiTU-GLU variants are benchmarked separately so their extra tanh/clamp work is
visible instead of being folded into the plain numbers.

``torch.compile`` recompiles per shape/dtype/requires-grad pattern. The
default Dynamo recompile limit is smaller than this benchmark's case count, so
the module raises it; otherwise the baseline silently falls back to eager and
the comparison is meaningless.
"""

import functools
import inspect
import os
import warnings
from typing import Optional

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


# Enough variants for 5 shapes x 3 dtypes of forward cases without Dynamo
# falling back to eager halfway through the suite.
_DYNAMO_RECOMPILE_LIMIT = 128


def _missing_kwargs(impl, kwargs):
    parameters = inspect.signature(impl).parameters
    return [name for name in kwargs if name not in parameters]


def _missing_baseline(op_kwargs, reason):
    """Baseline for variants the installed Megatron-LM cannot serve.

    Uses the main-branch implementation vendored in the appendix at the end of
    this file so the clamp / SiTU cases are still compared against a compiled
    Megatron instead of the eager composition.

    Once the minimum supported megatron-core ships those variants, delete the
    appendix and this function and restore the call sites in ``base_op`` to
    ``functools.partial(eager, **op_kwargs)``.
    """
    warnings.warn(
        f"{reason}; using the vendored main-branch implementation as the "
        "benchmark baseline",
        RuntimeWarning,
        stacklevel=3,
    )
    return functools.partial(bias_swiglu_impl, **op_kwargs)


def base_op(**op_kwargs):
    """Megatron-LM's native op when it supports the variant, else the vendored one.

    Mirrors ``base_op`` in ``test_quick_gelu.py``: the import happens here and
    a missing Megatron-LM is a warning plus a fallback, not a hard dependency.
    An installed Megatron-LM that predates the clamp / SiTU variants is served
    by ``_missing_baseline``.
    """
    name = "bias_swiglu_impl"
    try:
        import megatron.core.fusions.fused_bias_swiglu
    except ImportError:
        return _missing_baseline(op_kwargs, "Megatron-LM is unavailable")
    impl = getattr(megatron.core.fusions.fused_bias_swiglu, name, None)
    if impl is None:
        return _missing_baseline(op_kwargs, f"Megatron-LM has no {name}")
    missing = _missing_kwargs(impl, op_kwargs)
    if missing:
        return _missing_baseline(
            op_kwargs,
            f"installed Megatron-LM {name} does not accept {missing}",
        )
    return functools.partial(impl, **op_kwargs)


def _stabilize_megatron_compile():
    """Keep the compiled Megatron baseline working on every backend.

    ``megatron.core.jit`` compiles the baseline per shape, dtype and
    requires-grad pattern, so the default Dynamo limit is below this suite's
    case count. Vendor Megatron builds may wrap ``torch.compile`` in their own
    helper, so do not gate on an identity check against ``torch.compile``.
    Inductor's async compile also has to pickle the compiled Triton kernel back
    from its worker, which some vendor Triton builds cannot do.
    """
    try:
        from megatron.core import jit
    except ImportError:
        return

    # Inductor's async compile sends each Triton kernel to a worker process and
    # pickles the compiled kernel back. Vendor Triton builds whose JITFunction
    # cannot be pickled (it carries an RLock) fail there with
    # "TypeError: cannot pickle '_thread.RLock' object" while compiling the
    # Megatron baseline. One compile thread keeps compilation in-process; it
    # changes compile time only, not the measured kernel time.
    try:
        import torch._inductor.config

        if getattr(torch._inductor.config, "compile_threads", 1) != 1:
            torch._inductor.config.compile_threads = 1
    except ImportError:
        pass

    if jit.jit_fuser is getattr(jit, "noop_decorator", None):
        return

    try:
        import torch._dynamo
    except ImportError:
        return
    if hasattr(torch._dynamo.config, "recompile_limit"):
        current = torch._dynamo.config.recompile_limit
    elif hasattr(torch._dynamo.config, "cache_size_limit"):
        current = torch._dynamo.config.cache_size_limit
    else:
        return
    if current >= _DYNAMO_RECOMPILE_LIMIT:
        return
    if hasattr(torch._dynamo.config, "recompile_limit"):
        torch._dynamo.config.recompile_limit = _DYNAMO_RECOMPILE_LIMIT
    else:
        torch._dynamo.config.cache_size_limit = _DYNAMO_RECOMPILE_LIMIT
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


def _run(input_fn, torch_op, train_op, op_name, op_kwargs=None, dtype=None):
    # ``torch_op`` is already a configured baseline (native or fallback);
    # ``op_kwargs`` only selects the variant of the operator under test.
    # The operator is forward-only, so there is no backward case to measure.
    if op_kwargs:
        train_op = functools.partial(train_op, **op_kwargs)
    bench = _FixedShapeBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=[dtype] if dtype is not None else _dtypes(),
        is_backward=False,
    )
    bench.set_train(train_op)
    # On Ascend, enable parallel scheduling of all blocks to avoid excessive logical cores
    # and reduce scheduling overhead when the logical core count exceeds 65,535.
    # See https://github.com/flagos-ai/FlagGems/pull/7304
    if flag_train.vendor_name == "ascend":
        os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"
    bench.run()


# Table order inside one dtype group: (op_name, input_fn, train_op, op_kwargs).
_VARIANTS = [
    (
        "bias_swiglu_impl",
        bias_swiglu_input_fn,
        flag_train.megatron.bias_swiglu_impl,
        {},
    ),
    (
        "bias_swiglu_impl_clamped",
        bias_swiglu_input_fn,
        flag_train.megatron.bias_swiglu_impl,
        _CLAMPED_KWARGS,
    ),
    (
        "bias_swiglu_impl_situ",
        bias_swiglu_input_fn,
        flag_train.megatron.bias_swiglu_impl,
        _SITU_KWARGS,
    ),
]


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("dtype", _dtypes(), ids=lambda d: str(d).split(".")[-1])
def test_bias_swiglu_impl_perf(dtype):
    """Every table for one dtype, so a report reads dtype by dtype."""
    for op_name, input_fn, train_op, op_kwargs in _VARIANTS:
        _run(
            input_fn,
            base_op(**op_kwargs),
            train_op,
            op_name,
            op_kwargs=op_kwargs or None,
            dtype=dtype,
        )


# ===========================================================================
# Appendix: vendored Megatron-LM baseline
#
# Source: NVIDIA Megatron-LM, **main branch**, fetched 2026-10-28 (no release
# tag yet; these variants are expected to reach a release with 0.20.0).
#   https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/fusions/fused_bias_swiglu.py
#
# The megatron-core installed on this machine is 0.19.2. Its fused_bias_swiglu
# has no clamp / SiTU variants, so those benchmark cases would otherwise fall
# back to the eager composition; vendoring the main-branch code keeps them
# compared against a compiled Megatron.
#
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# Changes from upstream: the three Megatron-internal imports are replaced
# (jit_fuser -> torch.compile, nvtx_decorator -> no-op, transformer_engine's
# propagate_te_grouped_tensor_marker -> identity), ``F.`` is expanded to
# ``torch.nn.functional.``, and the backward functions plus the saved-tensor
# bookkeeping are dropped because this benchmark measures forward only. The
# forward math is verbatim, which is why it sits under ``fmt: off``: black
# must not reformat it, so it can be re-synced with upstream. Benchmark
# baseline only, never a correctness oracle.
#
# When the minimum supported megatron-core ships these variants, delete this
# appendix and ``_missing_baseline``, and drop the dispatch branch in
# ``base_op``.
# ===========================================================================
# fmt: off

jit_fuser = torch.compile


def nvtx_decorator(message=None):
    """No-op stand-in for ``megatron.core.utils.nvtx_decorator``."""

    def wrap(fn):
        return fn

    return wrap


def propagate_te_grouped_tensor_marker(source, target):
    """Identity stand-in: the benchmark passes plain dense tensors."""
    return target


@jit_fuser
def swiglu(y):
    """Performs SwiGLU (Swish-Gated Linear Unit) activation function.

    Args:
        y (torch.Tensor): Input tensor to be split into two halves along the last dimension.

    Returns:
        torch.Tensor: Result of SwiGLU activation: SiLU(y1) * y2, where y1, y2 are the split halves.
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    return torch.nn.functional.silu(y_1) * y_2


@jit_fuser
def bias_swiglu(y, bias):
    """Performs SwiGLU activation with bias addition.

    Args:
        y (torch.Tensor): Input tensor.
        bias (torch.Tensor): Bias tensor to be added to input.

    Returns:
        torch.Tensor: Result of bias addition followed by SwiGLU activation.
    """
    y = y + bias
    return swiglu(y)


@jit_fuser
def clamped_swiglu(y, clamp_value):
    """Perform SwiGLU after clamping both halves of the input."""
    dtype = y.dtype
    y_1, y_2 = torch.chunk(y.to(torch.float32), 2, -1)
    y_1 = y_1.clamp(min=None, max=clamp_value)
    y_2 = y_2.clamp(min=-clamp_value, max=clamp_value)
    res = torch.nn.functional.silu(y_1) * y_2
    return res.to(dtype)


@jit_fuser
def bias_clamped_swiglu(y, bias, clamp_value):
    """Perform clamped SwiGLU after bias addition."""
    return clamped_swiglu(y + bias, clamp_value)


@jit_fuser
def _tanh_clamp_and_deriv(y: torch.Tensor, clamp_scale: float):
    """Soft-clamps ``y`` and returns the clamp derivative alongside it, both in fp32."""
    t = torch.tanh(y.float() / clamp_scale)
    return clamp_scale * t, 1 - t * t


@jit_fuser
def situ_glu(y, gate_clamp_scale: float, linear_clamp_scale: Optional[float]):
    """SiTU-GLU: ``s_g * tanh(y1 / s_g) * sigmoid(y1) * y2``, with ``y2`` optionally clamped."""
    dtype = y.dtype
    y_1, y_2 = torch.chunk(y, 2, -1)
    c, _ = _tanh_clamp_and_deriv(y_1, gate_clamp_scale)
    gate = c * torch.sigmoid(y_1)
    if linear_clamp_scale is not None:
        y_2, _ = _tanh_clamp_and_deriv(y_2, linear_clamp_scale)
    return (gate * y_2).to(dtype)


@jit_fuser
def bias_situ_glu(y, bias, gate_clamp_scale: float, linear_clamp_scale: Optional[float]):
    """Bias addition followed by SiTU-GLU. The clamps act on the full pre-activation."""
    return situ_glu(y + bias, gate_clamp_scale, linear_clamp_scale)


def bias_swiglu_impl(
    input,
    bias,
    fp8_input_store=False,
    cpu_offload_input=False,
    clamp_value=None,
    gate_clamp_scale=None,
    linear_clamp_scale=None,
):
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    assert gate_clamp_scale is not None or linear_clamp_scale is None
    assert gate_clamp_scale is None or clamp_value is None
    input = input.view(-1, ori_shape[-1])
    if gate_clamp_scale is not None:
        if bias is None:
            out = situ_glu(input, gate_clamp_scale, linear_clamp_scale)
        else:
            out = bias_situ_glu(input, bias, gate_clamp_scale, linear_clamp_scale)
    elif clamp_value is not None and clamp_value > 0:
        if bias is None:
            out = clamped_swiglu(input, clamp_value)
        else:
            out = bias_clamped_swiglu(input, bias, clamp_value)
    elif bias is None:
        out = swiglu(input)
    else:
        out = bias_swiglu(input, bias)
    return out if len(ori_shape) == 2 else out.view(ori_shape[0], ori_shape[1], -1)
