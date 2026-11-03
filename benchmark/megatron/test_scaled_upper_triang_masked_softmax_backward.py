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

# !! KEEP IN SYNC with benchmark/megatron/test_scaled_upper_triang_masked_softmax_forward.py !!
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


class _MegatronBackward:
    """Megatron's backward over a graph built once per case.

    See the module docstring for why the graph is built here and why its input is
    a fresh tensor. Holding ``_source`` keeps the ``output`` it keys on alive, so a
    freed tensor's address cannot be handed to the next case's allocation and
    mistaken for the same one.
    """

    def __init__(self):
        self._source = None

    def __call__(self, grad_output, output, scale):
        if output is not self._source:
            self._source = output
            self._leaf = torch.randn(
                output.shape, dtype=output.dtype, device=output.device
            ).requires_grad_(True)
            self._output = megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax.apply(
                self._leaf, scale
            )
        return torch.autograd.grad(
            self._output, self._leaf, grad_output, retain_graph=True
        )[0]


class _ScaledUpperTriangMaskedSoftmaxBackwardBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _SHAPES
    DEFAULT_SHAPE_DESC = "attn_batches, sq, sk"

    def set_shapes(self, shape_file=None):
        self.shapes = list(_SHAPES)


def scaled_upper_triang_masked_softmax_backward_input_fn(shape, dtype, device):
    """``(grad_output, output, scale)`` for one case.

    ``output`` is the reference's own forward, so the inputs do not depend on the
    operator under test.
    """
    x = torch.randn(shape, dtype=dtype, device=device)
    output = megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax.apply(
        x, _SCALE
    )
    yield torch.randn(shape, dtype=dtype, device=device), output, _SCALE


@pytest.mark.scaled_upper_triang_masked_softmax_backward
def test_scaled_upper_triang_masked_softmax_backward():
    bench = _ScaledUpperTriangMaskedSoftmaxBackwardBenchmark(
        input_fn=scaled_upper_triang_masked_softmax_backward_input_fn,
        op_name="scaled_upper_triang_masked_softmax_backward",
        torch_op=_MegatronBackward(),
        dtypes=_DTYPES,
    )
    bench.set_train(flag_train.megatron.scaled_upper_triang_masked_softmax_backward)
    bench.run()
