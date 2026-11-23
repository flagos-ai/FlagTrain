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


"""Time moe_permute_mask_map using the same workload and reference as correctness."""

import pytest

import flag_train
from benchmark.base import GenericBenchmark
from benchmark.cases import BenchmarkCasePlan
from tests.megatron.test_moe_permute_mask_map import (
    CASES,
    DTYPES,
    OP_NAME,
    make_inputs,
    reference,
)


class OperatorBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        return []

    def set_shapes(self, shape_file=None):
        self.shapes = [()]


def _shapes(case):
    n = case.rows
    h = case.width
    mode = case.variant
    return {
        "inp": [n, h],
        "routing_map": [n, 4],
        "probs": [n, 4],
        "pad_offsets": [4] if mode == "padded" else None,
    }


def _params(case):
    n = case.rows
    mode = case.variant
    layout = [0, 1, 3, 6] if mode == "padded" else None
    return dict(
        num_out_tokens=2 * (n - (n + 6) // 7) + (8 if layout else 0), pad_offsets=layout
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


@pytest.mark.moe_permute_mask_map
def test_moe_permute_mask_map_benchmark():
    bench = OperatorBenchmark(
        op_name=OP_NAME,
        torch_op=reference,
        dtypes=list(DTYPES),
        is_backward=False,
        case_fn=_plans,
        build_inputs_fn=_build,
    )
    bench.op_name = OP_NAME
    bench.set_train(flag_train.megatron.moe_permute_mask_map)
    bench.run()
