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
"""FP8 GEMM performance against the explicitly selected platform baseline.

Ascend, MetaX and Hygon use Torch dequantization + BF16 matmul, including the complete
dequantization in each timed invocation. Other vendors retain original
DeepSpeed matmul_fp8 with its real quantizer. There is no automatic fallback.

Each ``(M, dtype)`` pair is an individual pytest node instead of one opaque
benchmark entry. On a stack for which the original DeepSpeed kernel is known
not to work, only that selected node is explicitly skipped. This keeps the
eleven usable original-BF16 measurements separate from unsupported original
cases; no repaired DeepSpeed kernel is used by this module.

Both timed calls consume the same quantized weights, scales, and activations;
both allocate their output in the timed call. Quantization, the mathematical
oracle, and three-way correctness checks happen before timing. NVIDIA kernel
mode in ``GenericBenchmark`` uses ``triton.testing.do_bench``. Every measured
acceptance case must exceed 0.8x speedup. Explicit ``--metrics latency`` is a
portable candidate-only diagnostic, never a baseline speedup acceptance run.
"""

import json
import math

import pytest
import torch

import flag_train
from flag_train.deepspeed import matmul_fp8_triton
from flag_train.runtime import torch_device_fn
from tests.deepspeed._fp8_gemm_utils import (
    CORE_GROUP_SIZE,
    CORE_K,
    CORE_M,
    CORE_N,
    LARGE_M,
    SUPPLEMENTAL_M,
    assert_fp8_close,
    baseline_matmul,
    baseline_metadata,
    make_baseline_inputs,
    make_portable_inputs,
    original_deepspeed_skip_reason,
    strict_matmul_context,
    torch_reference,
)

from .. import base
from ..consts import BenchmarkCasePlan

_M_VALUES = CORE_M
_K = CORE_K
_N = CORE_N
_GROUP_SIZE = CORE_GROUP_SIZE
_MIN_SPEEDUP = 0.8
_REQUIRED_METRICS = {"latency_base", "latency", "speedup"}
_DTYPES = (torch.bfloat16,)

_PYTEST_CASES = tuple(
    pytest.param(
        m,
        dtype,
        id=f"M{m}-{str(dtype).removeprefix('torch.')}",
    )
    for dtype in _DTYPES
    for m in _M_VALUES
)

pytestmark = pytest.mark.skipif(
    not flag_train.runtime.device.support_bf16,
    reason="The active backend does not support BF16",
)


class Fp8GemmBenchmark(base.GenericBenchmark):
    """One shape and one dtype per pytest node.

    ``GenericBenchmark`` owns case-ID registration and ``--case-id`` filtering.
    Keeping a benchmark instance this narrow ensures unselected parameter nodes
    only register their tensor-free case and never load DeepSpeed or allocate a
    GPU input.
    """

    DEFAULT_SHAPE_DESC = "M, K, N (quantization_group_size=128)"

    def __init__(self, *, selected_shape, **kwargs):
        self._selected_shape = tuple(selected_shape)
        self.candidate_only = False
        super().__init__(**kwargs)

    def set_shapes(self, shape_file=None):
        # FP8 acceptance shapes are fixed; a global shape file must not make a
        # parameterized test node benchmark a different shape.
        self.shapes = [self._selected_shape]

    def set_more_shapes(self):
        return []

    def init_user_config(self):
        super().init_user_config()
        if base.Config.query or base.Config.list_cases:
            return
        if base.Config.mode != base.consts.BenchMode.KERNEL:
            raise ValueError("FP8 GEMM requires --mode kernel (backend timer)")
        self.candidate_only = set(self.to_bench_metrics) == {"latency"}
        if self.candidate_only:
            return
        if base.Config.skip_native:
            raise ValueError("FP8 GEMM acceptance requires the selected baseline")
        if not _REQUIRED_METRICS.issubset(self.to_bench_metrics):
            raise ValueError(
                "FP8 GEMM acceptance requires latency_base, latency, and speedup metrics"
            )

    def build_inputs(self, case):
        """Materialize only a selected case, and skip only a known bad original one."""
        plan = case.builder_args[0]
        m, k, n = plan.builder_args
        group_size = plan.params["group_size"]
        if self.candidate_only:
            x, weight, scale = make_portable_inputs(
                m, k, n, case.dtype, group_size, self.device
            )
            actual = matmul_fp8_triton(x, weight, scale, group_size)
            with strict_matmul_context():
                assert_fp8_close(actual, torch_reference(x, weight, scale, group_size))
            torch_device_fn.synchronize()
            return (
                x,
                weight,
                scale,
                group_size,
                {
                    "fpq": None,
                    "baseline_label": json.dumps(
                        {
                            "mode": "candidate-only",
                            "acceptance": "not-evaluated",
                            "baseline": None,
                            "vendor": flag_train.vendor_name,
                            "shape_group": "large-m" if m in LARGE_M else "base",
                        },
                        sort_keys=True,
                    ),
                },
            )
        metadata = baseline_metadata()
        metadata["shape_group"] = "large-m" if m in LARGE_M else "base"
        reason = original_deepspeed_skip_reason(
            m,
            k,
            n,
            case.dtype,
            group_size,
            self.device,
            metadata=metadata,
        )
        if reason:
            # ``Benchmark._run_cases`` has registered this ID as available, but
            # it cannot append an explicit pytest skip to ``executed_case_ids``.
            # The local benchmark hook consumes this set when validating
            # --case-id selection, leaving unknown IDs as real errors.
            skipped = getattr(base.Config, "fp8_skipped_case_ids", None)
            if skipped is None:
                skipped = set()
                base.Config.fp8_skipped_case_ids = skipped
            skipped.add(case.case_id)
            pytest.skip(reason)
        return fp8_gemm_build_inputs(plan, case.dtype, self.device, metadata)

    def _measure_input(self, inputs, case_id=None):
        metric = super()._measure_input(inputs, case_id=case_id)
        label = case_id or str(metric.shape_detail)
        assert metric.error_msg is None, f"{label}: {metric.error_msg}"
        required = {"latency"} if self.candidate_only else _REQUIRED_METRICS
        for name in required:
            value = getattr(metric, name)
            assert (
                value is not None and math.isfinite(value) and value > 0
            ), f"{label}: invalid {name}={value!r}"
        if self.candidate_only:
            assert metric.latency_base is None and metric.speedup is None
            return metric
        # Verify the actual ratio as well as the framework's EPS-adjusted ratio.
        ratio = metric.latency_base / metric.latency
        assert (
            ratio > _MIN_SPEEDUP and metric.speedup > _MIN_SPEEDUP
        ), f"{label}: speedup {ratio:.6f}x must be > {_MIN_SPEEDUP}x"
        return metric

    def _emit_result(self, dtype, metrics):
        if not self.candidate_only:
            return super()._emit_result(dtype, metrics)
        previous = base.Config.native_baseline_skip_reason
        base.Config.native_baseline_skip_reason = (
            "Explicit --metrics latency: candidate-only diagnostic; "
            "Baseline speedup acceptance NOT evaluated"
        )
        try:
            return super()._emit_result(dtype, metrics)
        finally:
            base.Config.native_baseline_skip_reason = previous


def fp8_gemm_case_fn(shape, dtype):
    """Describe one case without importing DeepSpeed or allocating tensors.

    Listing/query modes deliberately report all 15 BF16 shapes. Whether
    a particular original DeepSpeed kernel can execute is checked only when
    that concrete case is selected for measurement.
    """
    m, k, n = shape
    yield BenchmarkCasePlan(
        shape={"activation": (m, k), "weight": (k, n)},
        params={
            "group_size": _GROUP_SIZE,
            "baseline": (
                "none-candidate-only"
                if set(base.Config.user_desired_metrics or ()) == {"latency"}
                else (
                    "torch-dequantize-matmul"
                    if flag_train.vendor_name in ("ascend", "metax", "hygon")
                    else "deepspeed-original"
                )
            ),
            "shape_origin": "supplemental" if m in SUPPLEMENTAL_M else "upstream",
            "shape_group": "large-m" if m in LARGE_M else "base",
            "original_deepspeed_check": (
                "not-applicable"
                if flag_train.vendor_name in ("ascend", "metax", "hygon")
                else "known-check-runtime"
            ),
        },
        builder_args=(m, k, n),
    )


def fp8_gemm_build_inputs(plan, dtype, device, metadata):
    """Validate both implementations outside timing; do not cache dequantization."""
    m, k, n = plan.builder_args
    group_size = plan.params["group_size"]
    x, weight, scale, fpq = make_baseline_inputs(m, k, n, dtype, group_size, device)
    reference = (
        torch_reference(x, weight, scale, group_size)
        if fpq is None
        else torch.matmul(x, fpq.dequantize(weight, scale=fpq.scale))
    )
    candidate = matmul_fp8_triton(x, weight, scale, group_size)
    baseline = baseline_matmul(x, weight, scale, group_size, fpq)
    assert_fp8_close(candidate, reference)
    assert_fp8_close(baseline, reference)
    assert_fp8_close(candidate, baseline, mean_relative=False)
    torch_device_fn.synchronize()

    # The framework drops unknown positional object types. Keep the quantizer
    # in kwargs, and persist the selected baseline identity in recorded inputs.
    return (
        x,
        weight,
        scale,
        group_size,
        {"fpq": fpq, "baseline_label": json.dumps(metadata, sort_keys=True)},
    )


def torch_op(x, weight, scale, group_size, *, fpq, baseline_label):
    """Full allocating baseline; Torch paths decode/scale on every timed call."""
    return baseline_matmul(x, weight, scale, group_size, fpq)


def train_op(x, weight, scale, group_size, *, fpq, baseline_label):
    """FlagTrain's allocating API; no quantization or dequantization cache."""
    return matmul_fp8_triton(x, weight, scale, group_size)


@pytest.mark.fp8_gemm
@pytest.mark.parametrize("m,dtype", _PYTEST_CASES)
def test_fp8_gemm_perf(m, dtype):
    """Benchmark exactly one visible core case against the selected baseline."""
    bench = Fp8GemmBenchmark(
        selected_shape=(m, _K, _N),
        case_fn=fp8_gemm_case_fn,
        build_inputs_fn=fp8_gemm_build_inputs,
        op_name="fp8_gemm",
        torch_op=torch_op,
        dtypes=[dtype],
    )
    bench.set_train(train_op)

    # ``run`` handles query/list-cases before it ever asks build_inputs for
    # DeepSpeed metadata. With --case-id, all parameter nodes register their
    # lightweight IDs, but only the chosen node reaches build_inputs/skip/GPU.
    if base.Config.query or base.Config.list_cases:
        bench.run()
        return

    with torch.no_grad(), strict_matmul_context():
        bench.run()
