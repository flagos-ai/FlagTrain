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


"""List and time backward of the explicit MTP loss scaler."""

import pytest

import flag_train
from benchmark.base import GenericBenchmark
from benchmark.cases import BenchmarkCasePlan
from tests.megatron.test_mtp_loss_auto_scaler import (
    DTYPES,
    OP_NAME,
    WORKLOADS,
    make_inputs,
    reference,
)


class MTPLossAutoScalerBenchmark(GenericBenchmark):
    def set_shapes(self, shape_file=None):
        self.shapes = list(WORKLOADS)

    def set_more_shapes(self):
        return []


def _cases(workload, dtype):
    shape, scale = workload
    yield BenchmarkCasePlan(
        shape={"output": list(shape), "mtp_loss": [], "loss_scale": []},
        params={"loss_scale": scale, "stage": "backward"},
        builder_args=(shape, scale),
    )


def _build_inputs(plan, dtype, device):
    shape, scale = plan.builder_args
    output, mtp_loss, scale = make_inputs(shape, dtype, device, scale)
    return output, mtp_loss, {"loss_scale": scale}


@pytest.mark.mtp_loss_auto_scaler
def test_mtp_loss_auto_scaler_perf():
    bench = MTPLossAutoScalerBenchmark(
        op_name=OP_NAME,
        torch_op=reference,
        dtypes=list(DTYPES),
        is_backward=True,
        case_fn=_cases,
        build_inputs_fn=_build_inputs,
    )
    # Keep the callable's Definition name in case and timing records.
    bench.op_name = OP_NAME
    bench.set_train(flag_train.megatron.mtp_loss_auto_scaler)
    bench.run()
