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
than eager PyTorch. When the installed Megatron-LM predates the clamp / SiTU
variants, the main-branch implementation vendored in the appendix at the end of
this file is used instead, so those cases are still compared against a compiled
baseline.

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
from functools import cache
from typing import Optional

import pytest
import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily

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


def _missing_baseline(weighted, op_kwargs, reason):
    """Baseline for variants the installed Megatron-LM cannot serve.

    Uses the main-branch implementation vendored in the appendix at the end of
    this file so the clamp / SiTU cases are still compared against a compiled
    Megatron instead of the eager composition.

    Once the minimum supported megatron-core ships those variants, delete the
    appendix and this function and restore the call sites in ``base_op`` to
    ``functools.partial(eager, **op_kwargs)``.
    """
    impl = weighted_bias_swiglu_impl if weighted else bias_swiglu_impl

    warnings.warn(
        f"{reason}; using the vendored main-branch implementation as the "
        "benchmark baseline",
        RuntimeWarning,
        stacklevel=3,
    )
    if weighted:
        return functools.partial(_megatron_weighted, impl, **op_kwargs)
    return functools.partial(impl, **op_kwargs)


def base_op(weighted=False, **op_kwargs):
    """Megatron-LM's native op when it supports the variant, else the vendored one.

    Mirrors ``base_op`` in ``test_quick_gelu.py``: the import happens here and
    a missing Megatron-LM is a warning plus a fallback, not a hard dependency.
    An installed Megatron-LM that predates the clamp / SiTU variants is served
    by ``_missing_baseline``; the eager composition is its own last resort.
    """
    name = "weighted_bias_swiglu_impl" if weighted else "bias_swiglu_impl"
    try:
        import megatron.core.fusions.fused_bias_swiglu
    except ImportError:
        return _missing_baseline(weighted, op_kwargs, "Megatron-LM is unavailable")
    impl = getattr(megatron.core.fusions.fused_bias_swiglu, name, None)
    if impl is None:
        return _missing_baseline(weighted, op_kwargs, f"Megatron-LM has no {name}")
    missing = _missing_kwargs(impl, op_kwargs)
    if missing:
        return _missing_baseline(
            weighted,
            op_kwargs,
            f"installed Megatron-LM {name} does not accept {missing}",
        )
    if weighted:
        return functools.partial(_megatron_weighted, impl, **op_kwargs)
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
# propagate_te_grouped_tensor_marker -> identity) and ``F.`` is expanded to
# ``torch.nn.functional.``. The rest is verbatim, which is why it sits under
# ``fmt: off``: black must not reformat it, so it can be re-synced with
# upstream. Benchmark baseline only, never a correctness oracle.
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


@cache
def _probe_clamp_boundaries():
    """Read the installed scalar-clamp subgradient using only CPU constants."""
    # PyTorch changed this within the 2.14 development series, so a version check
    # cannot distinguish builds. FunctionsManual.cpp at b2c75dd062 uses strict
    # inequalities; 4fdf77b940 uses inclusive ones. Probe both scalar-clamp forms
    # once on CPU, including calls under no_grad or inference_mode.
    # This zero-input probe needs real CPU constants, even during fake tracing.
    with unset_fake_temporarily(), torch.inference_mode(False), torch.enable_grad():
        gate = torch.tensor(1.0, dtype=torch.float32, device="cpu", requires_grad=True)
        linear = torch.tensor([-1.0, 1.0], dtype=torch.float32, device="cpu", requires_grad=True)
        gate_grad, linear_grad = torch.autograd.grad(
            gate.clamp(max=1.0) + linear.clamp(min=-1.0, max=1.0).sum(), (gate, linear)
        )
        gradients = [gate_grad.item(), *linear_grad.tolist()]
    if gradients not in ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]):
        raise RuntimeError(f"Unsupported scalar-clamp boundary gradients: {gradients}")
    return gradients[0] == 1.0


@torch.compiler.assume_constant_result
def _clamp_includes_boundaries():
    # Delay autograd initialization until this clamped backward is actually used.
    # Keep the cached probe outside tracing and specialize only its Python bool.
    return _probe_clamp_boundaries()


###### BIAS SWIGLU FUSION/ NO AUTOGRAD ################  # noqa: E266


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
def weighted_swiglu(y, weights):
    dtype = y.dtype
    res = swiglu(y) * weights
    return res.to(dtype)


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
def clamped_weighted_swiglu(y, weights, clamp_value):
    """Perform token-weighted clamped SwiGLU."""
    dtype = y.dtype
    res = clamped_swiglu(y, clamp_value) * weights
    return res.to(dtype)


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


@jit_fuser
def weighted_situ_glu(y, weights, gate_clamp_scale: float, linear_clamp_scale: Optional[float]):
    dtype = y.dtype
    res = situ_glu(y, gate_clamp_scale, linear_clamp_scale) * weights
    return res.to(dtype)


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@jit_fuser
def swiglu_back(g, y):
    """Computes the gradient for the SwiGLU activation function.

    Args:
        g (torch.Tensor): Gradient tensor from the subsequent layer.
        y (torch.Tensor): Input tensor that was used in the forward pass.

    Returns:
        torch.Tensor: Gradient with respect to the input tensor, computed using the
            chain rule and the derivative of the SiLU activation function.
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    return torch.cat(
        (g * torch.sigmoid(y_1) * (1 + y_1 * (1 - torch.sigmoid(y_1))) * y_2, g * torch.nn.functional.silu(y_1)), -1
    )


@jit_fuser
def bias_swiglu_back(g, y, bias):
    """Computes the gradient for the biased SwiGLU activation function.

    Args:
        g (torch.Tensor): Gradient tensor from the subsequent layer.
        y (torch.Tensor): Input tensor that was used in the forward pass.
        bias (torch.Tensor): Bias tensor that was added in the forward pass.

    Returns:
        torch.Tensor: Gradient with respect to the input tensor, computed after
            applying the bias addition.
    """
    y = y + bias
    return swiglu_back(g, y)


@jit_fuser
def weighted_swiglu_back(g, y, weights):
    input_dtype = y.dtype
    w_dtype = weights.dtype
    input_grad = swiglu_back(g * weights, y)
    # precison of w may be higher than y and g, so we need to cast g to w_dtype
    weights_grad = swiglu(y) * g.to(w_dtype)
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)
    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


@jit_fuser
def clamped_swiglu_back(g, y, clamp_value):
    """Compute the input gradient for clamped SwiGLU."""
    dtype = y.dtype
    y_1, y_2 = torch.chunk(y.to(torch.float32), 2, -1)
    y_1c = y_1.clamp(min=None, max=clamp_value)
    y_2c = y_2.clamp(min=-clamp_value, max=clamp_value)
    if _clamp_includes_boundaries():
        gate_mask = y_1 <= clamp_value
        linear_mask = (y_2 >= -clamp_value) & (y_2 <= clamp_value)
    else:
        gate_mask = y_1 < clamp_value
        linear_mask = (y_2 > -clamp_value) & (y_2 < clamp_value)
    res = torch.cat(
        (
            g
            * torch.sigmoid(y_1c)
            * (1 + y_1c * (1 - torch.sigmoid(y_1c)))
            * y_2c
            * gate_mask.to(g.dtype),
            g * torch.nn.functional.silu(y_1c) * linear_mask.to(g.dtype),
        ),
        -1,
    )
    return res.to(dtype)


@jit_fuser
def bias_clamped_swiglu_back(g, y, bias, clamp_value):
    """Compute the input gradient for clamped SwiGLU with bias."""
    return clamped_swiglu_back(g, y + bias, clamp_value)


@jit_fuser
def clamped_weighted_swiglu_back(g, y, weights, clamp_value):
    """Compute input and weight gradients for token-weighted clamped SwiGLU."""
    input_dtype = y.dtype
    w_dtype = weights.dtype
    input_grad = clamped_swiglu_back(g * weights, y, clamp_value)
    weights_grad = clamped_swiglu(y, clamp_value) * g.to(w_dtype)
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)
    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


@jit_fuser
def _situ_glu_grads(g, y, gate_clamp_scale: float, linear_clamp_scale: Optional[float]):
    """Gradients of SiTU-GLU w.r.t. the two halves of ``y``, kept in fp32."""
    y_1, y_2 = torch.chunk(y, 2, -1)
    c, sech2 = _tanh_clamp_and_deriv(y_1, gate_clamp_scale)
    sig = torch.sigmoid(y_1)
    gate = c * sig
    if linear_clamp_scale is not None:
        lin, lin_sech2 = _tanh_clamp_and_deriv(y_2, linear_clamp_scale)
        dy_2 = lin_sech2 * gate * g
    else:
        lin = y_2
        dy_2 = gate * g
    dy_1 = (sech2 + c * (1 - sig)) * sig * lin * g
    return dy_1, dy_2


@jit_fuser
def situ_glu_back(g, y, gate_clamp_scale: float, linear_clamp_scale: Optional[float]):
    input_dtype = y.dtype
    dy_1, dy_2 = _situ_glu_grads(g, y, gate_clamp_scale, linear_clamp_scale)
    return torch.cat((dy_1, dy_2), -1).to(input_dtype)


@jit_fuser
def bias_situ_glu_back(g, y, bias, gate_clamp_scale: float, linear_clamp_scale: Optional[float]):
    return situ_glu_back(g, y + bias, gate_clamp_scale, linear_clamp_scale)


@jit_fuser
def weighted_situ_glu_back(
    g, y, weights, gate_clamp_scale: float, linear_clamp_scale: Optional[float]
):
    input_dtype = y.dtype
    w_dtype = weights.dtype
    dy_1, dy_2 = _situ_glu_grads(g * weights, y, gate_clamp_scale, linear_clamp_scale)
    input_grad = torch.cat((dy_1, dy_2), -1)
    # precison of w may be higher than y and g, so we need to cast g to w_dtype
    weights_grad = situ_glu(y, gate_clamp_scale, linear_clamp_scale) * g.to(w_dtype)
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)
    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


class BiasSwiGLUFunction(torch.autograd.Function):
    """Custom autograd function for SwiGLU activation with bias support."""

    @staticmethod
    @nvtx_decorator()
    def forward(
        ctx,
        input,
        bias,
        fp8_input_store,
        cpu_offload_input,
        clamp_value,
        gate_clamp_scale,
        linear_clamp_scale,
    ):
        """Forward pass of biased SwiGLU activation.

        Args:
            ctx: Autograd context object for saving tensors for backward pass.
            input (torch.Tensor): Input tensor to apply SwiGLU to.
            bias (torch.Tensor): Bias tensor to be added to input before SwiGLU.
            fp8_input_store (bool): If True, stores intermediate values in FP8 format.
            clamp_value (Optional[float]): If set, hard-clamp both halves of the input before
                SwiGLU. Mutually exclusive with ``gate_clamp_scale``.
            gate_clamp_scale (Optional[float]): If set, use SiTU-GLU instead of SwiGLU, with the
                gate's linear factor soft-clamped to this scale.
            linear_clamp_scale (Optional[float]): If set, also soft-clamp the linear half to this
                scale. Requires ``gate_clamp_scale``.

        Returns:
            torch.Tensor: Result of applying bias addition followed by SwiGLU activation.
        """
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
            bias.activation_offloading = True
        ctx.save_for_backward(input_for_backward, bias)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        ctx.clamp_value = clamp_value
        ctx.gate_clamp_scale = gate_clamp_scale
        ctx.linear_clamp_scale = linear_clamp_scale
        if gate_clamp_scale is not None:
            return bias_situ_glu(input, bias, gate_clamp_scale, linear_clamp_scale)
        if clamp_value is not None and clamp_value > 0:
            return bias_clamped_swiglu(input, bias, clamp_value)
        return bias_swiglu(input, bias)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output):
        """Backward pass of biased SwiGLU activation.

        Args:
            ctx: Autograd context object containing saved tensors from forward pass.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            tuple: Tuple containing:
                - Gradient with respect to the input tensor
                - Gradient with respect to the bias tensor
                - None for fp8_input_store parameter
        """
        input, bias = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        if ctx.gate_clamp_scale is not None:
            tmp = bias_situ_glu_back(
                grad_output, input, bias, ctx.gate_clamp_scale, ctx.linear_clamp_scale
            )
        elif ctx.clamp_value is not None and ctx.clamp_value > 0:
            tmp = bias_clamped_swiglu_back(grad_output, input, bias, ctx.clamp_value)
        else:
            tmp = bias_swiglu_back(grad_output, input, bias)
        return tmp, tmp, None, None, None, None, None


class SwiGLUFunction(torch.autograd.Function):
    """Custom autograd function for SwiGLU activation without bias."""

    @staticmethod
    @nvtx_decorator()
    def forward(
        ctx,
        input,
        fp8_input_store,
        cpu_offload_input,
        clamp_value,
        gate_clamp_scale,
        linear_clamp_scale,
    ):
        """Forward pass of SwiGLU activation.

        Args:
            ctx: Autograd context object for saving tensors for backward pass.
            input (torch.Tensor): Input tensor to apply SwiGLU to.
            fp8_input_store (bool): If True, stores intermediate values in FP8 format.
            clamp_value (Optional[float]): If set, hard-clamp both halves of the input before
                SwiGLU. Mutually exclusive with ``gate_clamp_scale``.
            gate_clamp_scale (Optional[float]): If set, use SiTU-GLU instead of SwiGLU, with the
                gate's linear factor soft-clamped to this scale.
            linear_clamp_scale (Optional[float]): If set, also soft-clamp the linear half to this
                scale. Requires ``gate_clamp_scale``.

        Returns:
            torch.Tensor: Result of applying SwiGLU activation.
        """
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
        ctx.save_for_backward(input_for_backward)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        ctx.clamp_value = clamp_value
        ctx.gate_clamp_scale = gate_clamp_scale
        ctx.linear_clamp_scale = linear_clamp_scale
        if gate_clamp_scale is not None:
            return situ_glu(input, gate_clamp_scale, linear_clamp_scale)
        if clamp_value is not None and clamp_value > 0:
            return clamped_swiglu(input, clamp_value)
        return swiglu(input)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output):
        """Backward pass of SwiGLU activation.

        Args:
            ctx: Autograd context object containing saved tensors from forward pass.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            tuple: Tuple containing:
                - Gradient with respect to the input tensor
                - None for fp8_input_store parameter
        """
        input = ctx.saved_tensors[0]
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        if ctx.gate_clamp_scale is not None:
            tmp = situ_glu_back(grad_output, input, ctx.gate_clamp_scale, ctx.linear_clamp_scale)
        elif ctx.clamp_value is not None and ctx.clamp_value > 0:
            tmp = clamped_swiglu_back(grad_output, input, ctx.clamp_value)
        else:
            tmp = swiglu_back(grad_output, input)
        return tmp, None, None, None, None, None


class WeightedSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input, weights, fp8_input_store, clamp_value, gate_clamp_scale, linear_clamp_scale
    ):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        propagate_te_grouped_tensor_marker(input, input_for_backward)
        ctx.save_for_backward(input_for_backward, weights)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        ctx.clamp_value = clamp_value
        ctx.gate_clamp_scale = gate_clamp_scale
        ctx.linear_clamp_scale = linear_clamp_scale
        if gate_clamp_scale is not None:
            return weighted_situ_glu(input, weights, gate_clamp_scale, linear_clamp_scale)
        if clamp_value is not None and clamp_value > 0:
            return clamped_weighted_swiglu(input, weights, clamp_value)
        return weighted_swiglu(input, weights)

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        if ctx.gate_clamp_scale is not None:
            tmp, wgrad = weighted_situ_glu_back(
                grad_output, input, weights, ctx.gate_clamp_scale, ctx.linear_clamp_scale
            )
        elif ctx.clamp_value is not None and ctx.clamp_value > 0:
            tmp, wgrad = clamped_weighted_swiglu_back(grad_output, input, weights, ctx.clamp_value)
        else:
            tmp, wgrad = weighted_swiglu_back(grad_output, input, weights)
        return tmp, wgrad, None, None, None, None


def bias_swiglu_impl(
    input,
    bias,
    fp8_input_store=False,
    cpu_offload_input=False,
    clamp_value=None,
    gate_clamp_scale: Optional[float] = None,
    linear_clamp_scale: Optional[float] = None,
):
    """Implementation of biased SwiGLU that handles different input shapes.

    This function reshapes the input if necessary, applies the SwiGLU activation
    (with or without bias), and restores the original shape.

    Args:
        input (torch.Tensor): Input tensor to apply SwiGLU activation.
        bias (torch.Tensor, optional): Bias tensor to be added to input. If None,
            uses the bias-free SwiGLU variant.
        fp8_input_store (bool, optional): Whether to store intermediate values in FP8 format.
            Defaults to False.
        cpu_offload_input (bool, optional): Whether to mark saved activation inputs for CPU
            offloading. Defaults to False.
        clamp_value (float, optional): Maximum gate value and absolute linear value. When None,
            preserve the legacy unclamped SwiGLU behavior. Mutually exclusive with
            ``gate_clamp_scale``.
        gate_clamp_scale (Optional[float]): If set, compute SiTU-GLU instead of SwiGLU: the gate
            becomes ``s_g * tanh(x / s_g) * sigmoid(x)``, bounded by ``s_g``.
        linear_clamp_scale (Optional[float]): If set, also soft-clamp the linear half, bounding
            the output by ``gate_clamp_scale * linear_clamp_scale``. Requires ``gate_clamp_scale``.

    Returns:
        torch.Tensor: Result of biased SwiGLU activation.

    Raises:
        AssertionError: If input tensor does not have 2 or 3 dimensions.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    assert gate_clamp_scale is not None or linear_clamp_scale is None
    assert gate_clamp_scale is None or clamp_value is None
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        output = BiasSwiGLUFunction.apply(
            input,
            bias,
            fp8_input_store,
            cpu_offload_input,
            clamp_value,
            gate_clamp_scale,
            linear_clamp_scale,
        )
    else:
        output = SwiGLUFunction.apply(
            input,
            fp8_input_store,
            cpu_offload_input,
            clamp_value,
            gate_clamp_scale,
            linear_clamp_scale,
        )

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


def weighted_bias_swiglu_impl(
    input,
    bias,
    weights,
    fp8_input_store=False,
    clamp_value=None,
    gate_clamp_scale: Optional[float] = None,
    linear_clamp_scale: Optional[float] = None,
):
    """
    Token-wise-weighted bias swiglu fusion.

    Args:
        clamp_value (float, optional): Maximum gate value and absolute linear value. When None,
            preserve the legacy unclamped SwiGLU behavior. Mutually exclusive with
            ``gate_clamp_scale``.
        gate_clamp_scale (Optional[float]): If set, compute SiTU-GLU instead of SwiGLU: the gate
            becomes ``s_g * tanh(x / s_g) * sigmoid(x)``, bounded by ``s_g``.
        linear_clamp_scale (Optional[float]): If set, also soft-clamp the linear half. Requires
            ``gate_clamp_scale``.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    assert gate_clamp_scale is not None or linear_clamp_scale is None
    assert gate_clamp_scale is None or clamp_value is None
    input = propagate_te_grouped_tensor_marker(input, input.view(-1, ori_shape[-1]))
    if bias is not None:
        raise NotImplementedError("Bias is not supported for weighted swiglu fusion")
    else:
        output = WeightedSwiGLUFunction.apply(
            input, weights, fp8_input_store, clamp_value, gate_clamp_scale, linear_clamp_scale
        )

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


# bias_swiglu_impl = BiasSwiGLUFunction.apply
# swiglu_impl = SwiGLUFunction.apply
# fmt: on
