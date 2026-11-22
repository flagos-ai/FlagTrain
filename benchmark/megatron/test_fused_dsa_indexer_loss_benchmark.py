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


"""Time fused_dsa_indexer_loss using the same workload and reference as correctness."""

import flag_train
import pytest
from tests.megatron.test_fused_dsa_indexer_loss import (
    CASES,
    DTYPES,
    OP_NAME,
    make_inputs,
    reference,
)

from benchmark.base import GenericBenchmark
from benchmark.cases import BenchmarkCasePlan


class OperatorBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        return []

    def set_shapes(self, shape_file=None):
        self.shapes = [()]


def _shapes(case):
    n = case.rows
    h = case.width
    return {
        "q": [n, 2, 2, h],
        "weights": [n, 2, 2],
        "k": [n + 5, 2, h],
        "query": [n, 2, 3, h],
        "key": [n + 5, 2, 3, h],
        "mask": [n, n + 5],
    }


def _params(case):
    h = case.width
    mode = case.variant
    return dict(
        softmax_scale=h ** (-0.5),
        topk=4,
        loss_coeff=0.7,
        sparse_loss=mode == "sparse",
        calculate_per_token_loss=False,
        use_relu=True,
        mask="prefix_q_plus_two",
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


@pytest.mark.fused_dsa_indexer_loss
def test_fused_dsa_indexer_loss_benchmark():
    bench = OperatorBenchmark(
        op_name=OP_NAME,
        torch_op=reference,
        dtypes=list(DTYPES),
        is_backward=False,
        case_fn=_plans,
        build_inputs_fn=_build,
    )
    bench.op_name = OP_NAME
    bench.set_train(flag_train.megatron.fused_dsa_indexer_loss)
    bench.run()
