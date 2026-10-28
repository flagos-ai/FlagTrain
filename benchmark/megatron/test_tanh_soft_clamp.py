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

import pytest
import torch

from flag_train.megatron import tanh_soft_clamp

from .. import base

_SCALE = 0.5

_SHAPES = [
    (1024,),
    (65536,),
    (1048576,),
    (4096, 4096),
    (8192, 4096),
]


class TanhSoftClampBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_SHAPES)

    def set_more_shapes(self):
        return []


def tanh_soft_clamp_input_fn(shape, dtype, device):
    yield torch.randn(shape, dtype=dtype, device=device), _SCALE


def torch_op(x, scale):
    return (scale * torch.tanh(x.float() / scale)).to(x.dtype)


def train_op(x, scale):
    return tanh_soft_clamp(x, scale)


@pytest.mark.tanh_soft_clamp
def test_tanh_soft_clamp_perf():
    bench = TanhSoftClampBenchmark(
        input_fn=tanh_soft_clamp_input_fn,
        op_name="tanh_soft_clamp",
        torch_op=torch_op,
    )
    bench.set_train(train_op)
    bench.run()
