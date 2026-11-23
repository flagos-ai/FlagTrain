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


"""List and measure Megatron mRoPE with explicit position IDs."""

import pytest
import torch

import flag_train
from benchmark.base import GenericBenchmark
from benchmark.cases import BenchmarkCasePlan
from tests.megatron.test_multi_modal_rotary_embedding import (
    OP_NAME,
    TIMING_WORKLOADS,
    make_inputs,
    reference,
)


class MultimodalRotaryEmbeddingBenchmark(GenericBenchmark):
    def set_shapes(self, shape_file=None):
        self.shapes = list(TIMING_WORKLOADS)

    def set_more_shapes(self):
        return []


def _cases(workload, dtype):
    yield BenchmarkCasePlan(
        shape={
            "position_ids": [3, workload.batch, workload.length],
            "inv_freq": [sum(workload.sections)],
        },
        params={
            "position_ids_dtype": "int64",
            "mrope_section": list(workload.sections),
            "seq_len_interpolation_factor": workload.interpolation,
        },
        builder_args=(workload,),
    )


def _build_inputs(plan, dtype, device):
    position_ids, inv_freq, sections, interpolation = make_inputs(
        plan.builder_args[0], device
    )
    return (
        position_ids,
        inv_freq,
        {
            "mrope_section": sections,
            "seq_len_interpolation_factor": interpolation,
        },
    )


@pytest.mark.multi_modal_rotary_embedding
def test_multi_modal_rotary_embedding_perf():
    bench = MultimodalRotaryEmbeddingBenchmark(
        op_name=OP_NAME,
        torch_op=reference,
        dtypes=[torch.float32],
        case_fn=_cases,
        build_inputs_fn=_build_inputs,
    )
    bench.set_train(flag_train.megatron.multi_modal_rotary_embedding)
    bench.run()
