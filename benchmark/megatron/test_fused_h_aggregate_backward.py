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
"""Performance benchmark for the fused n-stream weighted aggregation (backward).

The baseline is Megatron-LM's cuTile ``_cutile_h_aggregate_bwd`` when the
installed megatron-core provides one that can run here. When it cannot, the
baseline is the autograd derivative of Megatron's own ``native_h_aggregate``
(a ``@torch.compile``d torch composition), built on the same ``(x, h_pre)`` the
operator is given; with an older megatron-core that predates it, the
transcription below is compiled the way Megatron compiles it. ``_BASELINE``
records which case was measured.

Timing range: only the backward. The operator under test takes ``grad_output``,
``x`` and ``h_pre`` and does no forward work; Megatron's cuTile backward takes
the same three operands, so there is nothing to build outside the timed region
and nothing is charged to the baseline that the operator does not do.

The harness's own ``is_backward`` path is not used: it calls ``randn_like`` on
the op's output, and both this operator and the baseline return a tuple.

The gradient is worth its own benchmark because it is a *different* shape of
work from the forward: ``grad_x`` is a broadcast product with no reduction, while
``grad_h`` reduces over ``C``. Latency is reported with GB/s (the traffic the
three inputs and two outputs imply) and TFLOP/s (one multiply per element for
``grad_x`` and a multiply-add for ``grad_h``, i.e. ``3 * s * b * n * C``).
"""

import warnings

import pytest
import torch
import torch._functorch.config

import flag_train
import flag_train.megatron

from .. import base

# ``(s, b, n, C)``: ``x`` is ``[s, b, n, C]``, ``h_pre`` is ``[s, b, n]``,
# ``grad_output`` is ``[s, b, C]``, and the outputs are ``grad_x`` of
# ``[s, b, n, C]`` and ``grad_h`` of ``[s, b, n]`` -- the same sweep the
# correctness suite checks every entry of.
#
# ``n`` is the mHC stream count (2-8) and ``C`` the hidden size; together they
# set the traffic, which is what this stage is. The row count walks a single row
# (where the launch, not the kernel, is the measurement), the tiles the
# correctness suite masks, and a full micro-batch. 2 and 8 bracket the stream
# axis, 7168 is a real hidden size that is not a power of two, and the
# ``(256, 32, ...)`` rows are the largest that fit comfortably beside the
# reference's own allocations.
# !! KEEP IN SYNC with tests/megatron/test_fused_h_aggregate_backward.py !!
_BENCHMARK_SHAPES = [
    (1, 1, 2, 256),
    (1, 1, 4, 4096),
    (8, 16, 4, 4096),
    (64, 8, 4, 4096),
    (32, 32, 4, 4096),
    (128, 32, 4, 4096),
    (256, 32, 4, 4096),
    (256, 32, 2, 4096),
    (256, 32, 8, 4096),
    (256, 32, 4, 7168),
]

# The upstream stage runs on FP16/BF16 activations; FP32 is measured by the
# correctness suite (it is the exact-arithmetic path) but is not what a training
# step pays for here.
_DTYPES = [torch.float16]
if flag_train.runtime.device.support_bf16:
    _DTYPES.append(torch.bfloat16)


def native_h_aggregate_ref(x, h_pre):
    """Megatron-LM's ``native_h_aggregate`` body, transcribed."""
    return (x * h_pre.unsqueeze(-1)).sum(dim=2)


def _load_native_h_aggregate():
    """Megatron's ``native_h_aggregate``, or its transcribed body compiled."""
    try:
        from megatron.core.transformer.hyper_connection import native_h_aggregate
    except Exception:
        return torch.compile(native_h_aggregate_ref), "transcribed"
    return native_h_aggregate, "megatron-core"


# The compiled baseline needs one graph per (shape, dtype); with dynamo's
# default budget of 8 recompiles the tail of the sweep would be served eager.
_COMPILE_BUDGET = 256


@pytest.fixture(scope="module", autouse=True)
def _full_compile_budget():
    """Keep ``@torch.compile`` compiling across the whole sweep."""
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


@pytest.fixture(scope="module", autouse=True)
def _no_buffer_donation():
    """Compile the baseline's backward without buffer donation.

    A donating backward may only run once (``retain_graph=False``); this
    benchmark re-runs the same graph for every timed repetition.
    """
    config = torch._functorch.config
    saved = config.donated_buffer
    config.donated_buffer = False
    try:
        yield
    finally:
        config.donated_buffer = saved


class _MegatronBackwardBaseline:
    """Megatron's backward for one input set, with the graph built up front.

    Only the backward is timed; the forward that builds the graph is done once
    per input set, outside the timed region.
    """

    def __init__(self, forward):
        self._forward = forward
        self._source_x = None
        self._source_h = None
        self._x = None
        self._h = None
        self._out = None

    def _build(self, x, h_pre):
        # Leaves on a copy: the harness's tensors are reused by the operator
        # under test, and ``requires_grad_`` on them would make that call build a
        # graph it never needs.
        self._x = x.detach().clone().requires_grad_(True)
        self._h = h_pre.detach().clone().requires_grad_(True)
        self._out = self._forward(self._x, self._h)
        self._source_x = x
        self._source_h = h_pre

    def __call__(self, grad_output, x, h_pre):
        if self._source_x is not x or self._source_h is not h_pre:
            self._build(x, h_pre)
        return torch.autograd.grad(
            self._out, (self._x, self._h), grad_output, retain_graph=True
        )


def _load_baseline():
    """Megatron's cuTile backward, else the autograd of ``native_h_aggregate``."""
    try:
        from megatron.core.fusions.fused_mhc_kernels import _cutile_h_aggregate_bwd

        x = torch.randn((1, 1, 2, 8), dtype=torch.bfloat16, device=flag_train.device)
        h_pre = torch.randn((1, 1, 2), dtype=torch.bfloat16, device=flag_train.device)
        grad_output = torch.randn(
            (1, 1, 8), dtype=torch.bfloat16, device=flag_train.device
        )
        with base.torch_device_fn.device(x.device):
            _cutile_h_aggregate_bwd(grad_output, x, h_pre)
            base.torch_device_fn.synchronize()
        return _cutile_h_aggregate_bwd, "megatron _cutile_h_aggregate_bwd"
    except Exception as exc:
        warnings.warn(
            "Megatron-LM's cuTile fused_h_aggregate_backward kernel is "
            f"unavailable ({exc!r}); timing the autograd derivative of its "
            "native_h_aggregate torch composition instead.",
            stacklevel=2,
        )

    op, source = _load_native_h_aggregate()
    return _MegatronBackwardBaseline(op), (
        f"megatron native_h_aggregate autograd ({source}, @torch.compile, "
        "donated buffers off)"
    )


torch_op, _BASELINE = _load_baseline()


class FusedHAggregateBackwardBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _BENCHMARK_SHAPES
    DEFAULT_SHAPE_DESC = "s, b, n, C"
    # The pass reads ``x`` once, ``grad_output`` once and writes ``grad_x`` --
    # three times the forward's traffic -- so GB/s is the reading that moves.
    DEFAULT_METRICS = base.consts.DEFAULT_METRICS[:] + ["gbps", "tflops"]

    def set_shapes(self, shape_file=None):
        # Each row is an ``(s, b, n, C)`` tuple with a fixed meaning, so the
        # shared core_shapes.yaml -- and the generic 1D/2D/3D extra shapes -- do
        # not apply.
        self.shapes = list(_BENCHMARK_SHAPES)

    def set_more_shapes(self):
        return []

    def get_gbps(self, args, latency):
        """Boundary traffic of one call: the three inputs in, both gradients out."""
        grad_output, x, h_pre = args
        s, b, n, C = x.shape
        element = x.element_size()
        io_bytes = x.numel() * element
        io_bytes += h_pre.numel() * element
        io_bytes += s * b * C * grad_output.element_size()
        io_bytes += x.numel() * element  # grad_x
        io_bytes += h_pre.numel() * element  # grad_h
        return io_bytes * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        """``3 * s * b * n * C``: a multiply for ``grad_x``, a multiply-add for ``grad_h``.

        The ``grad_h`` accumulation is counted as two FLOPs and the ``grad_x``
        product as one, which is what the kernel issues. The harness divides the
        count by the *operator's* latency, so the reading describes the operator
        rather than the baseline.
        """
        s, b, n, C = args[1].shape
        return 3 * s * b * n * C


def fused_h_aggregate_backward_input_fn(shape, dtype, device):
    """``(grad_output, x, h_pre)`` for one ``(s, b, n, C)`` case."""
    s, b, n, C = shape
    # ``x`` is what the forward computed from and ``grad_output`` what came back
    # through it; the operator reads both, so both are generated here rather than
    # derived from one another.
    grad_output = torch.randn((s, b, C), dtype=dtype, device=device)
    x = torch.randn((s, b, n, C), dtype=dtype, device=device)
    h_pre = torch.randn((s, b, n), dtype=dtype, device=device)
    yield grad_output, x, h_pre


def train_op(grad_output, x, h_pre):
    """The operator under test."""
    return flag_train.megatron.fused_h_aggregate_backward(grad_output, x, h_pre)


@pytest.mark.fused_h_aggregate_backward
def test_fused_h_aggregate_backward_perf():
    print(f"\nBaseline: {_BASELINE}")

    bench = FusedHAggregateBackwardBenchmark(
        input_fn=fused_h_aggregate_backward_input_fn,
        op_name="fused_h_aggregate_backward",
        # Megatron's cuTile kernel, or the autograd of its native_h_aggregate
        # composition when that kernel cannot run here.
        torch_op=torch_op,
        dtypes=_DTYPES,
    )
    bench.set_train(train_op)
    bench.run()
