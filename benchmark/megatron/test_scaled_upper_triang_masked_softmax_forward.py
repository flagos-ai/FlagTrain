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
import megatron.core.fusions.fused_softmax
import pytest
import torch

import flag_train

from .. import base

# Scales are not comparable across runs if they differ, so the 1/sqrt(d) used
# for attention scores is fixed here.
_SCALE = 0.125

_SHAPES = [
    (64, 64, 64),
    (16, 128, 128),
    (64, 128, 128),
    (16, 256, 256),
    (32, 256, 256),
    (32, 512, 512),
    (16, 1024, 1024),
    (8, 2048, 2048),
    (16, 2048, 2048),
    (8, 4096, 4096),
    (32, 4096, 4096),
    (8, 8192, 8192),
]

# Megatron's CUDA fused kernel rejects fp32.
_DTYPES = [torch.float16, torch.bfloat16]

# On Ascend the Megatron reference calls torch_npu.npu_scaled_masked_softmax,
# which supports fp32.
if flag_train.vendor_name == "ascend":
    _DTYPES.append(torch.float32)


class _ScaledUpperTriangMaskedSoftmaxBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_SHAPES)


def scaled_upper_triang_masked_softmax_input_fn(shape, dtype, device):
    yield (torch.randn(shape, dtype=dtype, device=device), _SCALE)


@pytest.mark.scaled_upper_triang_masked_softmax_forward
def test_scaled_upper_triang_masked_softmax_forward():
    bench = _ScaledUpperTriangMaskedSoftmaxBenchmark(
        input_fn=scaled_upper_triang_masked_softmax_input_fn,
        op_name="scaled_upper_triang_masked_softmax_forward",
        torch_op=megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax.apply,
        dtypes=_DTYPES,
    )
    bench.set_train(flag_train.megatron.scaled_upper_triang_masked_softmax_forward)
    bench.run()
