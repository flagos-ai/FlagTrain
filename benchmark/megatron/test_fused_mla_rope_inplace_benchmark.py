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


"""Time fused_mla_rope_inplace using the same workload and reference as correctness."""

import math
import statistics
import time

import flag_train
import pytest
import torch
from tests.megatron.test_fused_mla_rope_inplace import (
    CASES,
    DTYPES,
    OP_NAME,
    make_inputs,
    reference,
)

from benchmark.base import GenericBenchmark
from benchmark.cases import BenchmarkCasePlan
from benchmark.conftest import Config


class OperatorBenchmark(GenericBenchmark):
    def get_latency(self, op, *args, _reference_candidate=False, **kwargs):
        t = args[0]
        original = t.clone()
        device = torch.get_device_module(t.device)
        if Config.mode.value == "cudagraph":
            pytest.fail(
                "In-place RoPE requires timing with restoration outside each sample"
            )
        wall_time = Config.mode.value in ("operator", "wrapper")

        def call_on_original():
            t.copy_(original)
            op(*args, **kwargs)

        def measure(count):
            if wall_time:
                samples = []
                for _ in range(count):
                    t.copy_(original)
                    device.synchronize()
                    start = time.perf_counter()
                    op(*args, **kwargs)
                    device.synchronize()
                    samples.append((time.perf_counter() - start) * 1000)
                return samples

            samples = []
            # Bound the number of outstanding device events for long budgets.
            pairs = [
                (device.Event(enable_timing=True), device.Event(enable_timing=True))
                for _ in range(min(count, 64))
            ]
            while len(samples) < count:
                batch = pairs[: count - len(samples)]
                for start, end in batch:
                    t.copy_(original)
                    start.record()
                    op(*args, **kwargs)
                    end.record()
                device.synchronize()
                samples.extend(start.elapsed_time(end) for start, end in batch)
            return samples

        try:
            # Exclude lazy compilation and estimate the cost of one call in ms.
            call_on_original()
            device.synchronize()
            estimate = statistics.median(measure(5))
            assert math.isfinite(estimate) and estimate > 0
            for _ in range(max(0, int(Config.warm_up / estimate))):
                call_on_original()
            device.synchronize()
            samples = measure(max(1, int(Config.repetition / estimate)))
            latency = statistics.median(samples)
            assert math.isfinite(latency) and latency > 0
            return latency
        finally:
            t.copy_(original)

    def set_more_shapes(self):
        return []

    def set_shapes(self, shape_file=None):
        self.shapes = [()]


def _shapes(case):
    n = case.rows
    h = case.width
    mode = case.variant
    return {
        "t": [n, 2, 3, h + 8] if mode == "dense" else [n, 3, h + 8],
        "cos": [2 * n, 1, 1, h],
        "sin": [2 * n, 1, 1, h],
        "cu_seqlens_q": None if mode == "dense" else [3],
    }


def _params(case):
    h = case.width
    mode = case.variant
    return dict(
        nope_dim=8,
        emb_dim=h,
        cp_rank=0,
        cp_size=1,
        rotary_interleaved=False,
        inverse=False,
        remove_interleaving=mode == "packed",
        position_ids=None,
    )


def _plans(_shape, _dtype):
    for case in CASES:
        if not case.correctness_only:
            yield BenchmarkCasePlan(
                shape=_shapes(case),
                params={**_params(case), "variant": case.variant, "stage": "call"},
                builder_args=(case,),
            )


def _build(plan, dtype, device):
    args = make_inputs(plan.builder_args[0], dtype, device)
    return args


@pytest.mark.fused_mla_rope_inplace
def test_fused_mla_rope_inplace_benchmark():
    bench = OperatorBenchmark(
        op_name=OP_NAME,
        torch_op=reference,
        dtypes=list(DTYPES),
        is_backward=False,
        case_fn=_plans,
        build_inputs_fn=_build,
    )
    bench.op_name = OP_NAME
    bench.set_train(flag_train.megatron.fused_mla_rope_inplace)
    bench.run()
