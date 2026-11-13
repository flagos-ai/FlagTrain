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

The baseline is Megatron-LM's ``bias_swiglu_impl`` when megatron-core >= 0.20
is installed. Older versions lack the clamp / SiTU arguments, so the
main-branch implementation vendored at the end of this file is used instead and
every variant stays compared against a compiled Megatron baseline. The
``Baseline:`` line printed by the test records which one ran.

Timing, warmup and synchronization come from ``benchmark/base.py``
(``triton.testing.do_bench`` in kernel mode). ``--level core`` runs the shapes
below. One table covers the three activations (plain, clamped, SiTU-GLU);
``--dtypes`` chooses the dtypes. The operator is forward-only.
"""

import importlib.metadata
import os
import re
import warnings
from typing import Optional

import pytest
import torch

import flag_train

from .. import base, consts

# (tokens, 2 * hidden): the last dim is the concatenated [gate | linear].
_CORE_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (16384, 4096),
    (2048, 11008),
    (16, 128, 8192),
    (256, 32768),
]


# Enough variants for 7 shapes x 3 activations x 3 dtypes of forward cases
# without Dynamo falling back to eager halfway through the suite.
_DYNAMO_RECOMPILE_LIMIT = 128

# megatron-core 0.20 is the first release with the clamp / SiTU arguments.
_MCORE_MIN_VERSION = (0, 20)


def _megatron_version():
    """Installed megatron-core version and ``(major, minor)``, or ``(None, None)``."""
    try:
        version = importlib.metadata.version("megatron-core")
    except Exception:
        return None, None
    match = re.match(r"\s*(\d+)(?:\.(\d+))?", version)
    parsed = None if match is None else (int(match.group(1)), int(match.group(2) or 0))
    return version, parsed


_MCORE_VERSION, _MCORE_VERSION_TUPLE = _megatron_version()


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
    saved_suppress = dynamo.suppress_errors
    # Hygon's installed torch and Triton disagree on Inductor's config format
    # (``instance_descriptor`` vs dict), so the compiled baseline cannot build.
    # Fall back to eager while the vendored baseline is in use; once
    # megatron-core >= 0.20 provides it, keep the normal compiled path.
    if flag_train.vendor_name == "hygon" and (
        _MCORE_VERSION_TUPLE is None or _MCORE_VERSION_TUPLE < _MCORE_MIN_VERSION
    ):
        dynamo.suppress_errors = True

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
        dynamo.suppress_errors = saved_suppress
        if limit_name is not None:
            setattr(dynamo, limit_name, saved_limit)


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


class BiasSwigluImplBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _CORE_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_CORE_SHAPES)


# Variant label -> operator kwargs. The label travels with each input row, so
# one benchmark table covers all three activations.
_VARIANT_KWARGS = {
    "plain": {},
    "clamped": _CLAMPED_KWARGS,
    "situ": _SITU_KWARGS,
}


def bias_swiglu_input_fn(shape, dtype, device):
    x = torch.randn(shape, dtype=dtype, device=device)
    bias = torch.randn((shape[-1],), dtype=dtype, device=device)
    for variant in _VARIANT_KWARGS:
        yield (x, bias, variant)


def torch_op(x, bias, variant):
    """Installed Megatron >= 0.20, or the vendored main-branch copy."""
    return _baseline_op(x, bias, **_VARIANT_KWARGS[variant])


def train_op(x, bias, variant):
    """The operator under test."""
    return flag_train.megatron.bias_swiglu_impl(x, bias, **_VARIANT_KWARGS[variant])


@pytest.mark.bias_swiglu_impl
def test_bias_swiglu_impl_perf():
    """Plain, clamped, and SiTU-GLU rows for every shape in one table."""
    print(f"\nBaseline: {_BASELINE}")
    # On Ascend, enable parallel scheduling of all blocks to avoid excessive
    # logical cores and reduce scheduling overhead when the logical core count
    # exceeds 65,535.
    # See https://github.com/flagos-ai/FlagGems/pull/7304
    if flag_train.vendor_name == "ascend":
        os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"

    bench = BiasSwigluImplBenchmark(
        input_fn=bias_swiglu_input_fn,
        op_name="bias_swiglu_impl",
        torch_op=torch_op,
        dtypes=_dtypes(),
        is_backward=False,
    )
    bench.set_train(train_op)
    bench.run()


# ===========================================================================
# Appendix: vendored Megatron-LM baseline
#
# Source: NVIDIA Megatron-LM main branch, fetched 2026-10-28:
#   https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/fusions/fused_bias_swiglu.py
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. Apache-2.0.
#
# ``_load_baseline`` below picks this copy whenever megatron-core < 0.20 or is
# not installed. The forward math is verbatim under ``fmt: off`` so it can be
# re-synced with upstream; ``jit_fuser`` is replaced by ``torch.compile``,
# ``F.`` is expanded to ``torch.nn.functional.``, and backward-only helpers are
# omitted because this benchmark measures forward only. Benchmark baseline
# only, never a correctness oracle. Delete this appendix once every supported
# megatron-core is >= 0.20.
# ===========================================================================

jit_fuser = torch.compile


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
def bias_situ_glu(
    y, bias, gate_clamp_scale: float, linear_clamp_scale: Optional[float]
):
    """Bias addition followed by SiTU-GLU. The clamps act on the full pre-activation."""
    return situ_glu(y + bias, gate_clamp_scale, linear_clamp_scale)


def _vendored_baseline(
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


def _load_baseline():
    """Installed Megatron when it has the clamp/SiTU arguments, else vendored."""
    if _MCORE_VERSION_TUPLE is not None and _MCORE_VERSION_TUPLE >= _MCORE_MIN_VERSION:
        try:
            from megatron.core.fusions.fused_bias_swiglu import (
                bias_swiglu_impl as installed,
            )
        except ImportError as exc:
            reason = f"megatron-core {_MCORE_VERSION} cannot import it: {exc}"
        else:
            return installed, f"megatron-core {_MCORE_VERSION}"
    elif _MCORE_VERSION is None:
        reason = "Megatron-LM is not installed"
    else:
        reason = (
            f"megatron-core {_MCORE_VERSION} predates the clamp/SiTU arguments "
            f"(need >= {_MCORE_MIN_VERSION[0]}.{_MCORE_MIN_VERSION[1]})"
        )

    warnings.warn(
        f"{reason}; using the vendored main-branch implementation as the "
        "benchmark baseline",
        RuntimeWarning,
        stacklevel=2,
    )
    return _vendored_baseline, "vendored main-branch implementation"


_baseline_op, _BASELINE = _load_baseline()
