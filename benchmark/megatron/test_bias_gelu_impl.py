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
"""Benchmark bias_gelu_impl against the installed Megatron-Core operator."""

import pytest
import torch

import flag_train

from .. import base

_CORE_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (16384, 4096),
    (2048, 11008),
    (16, 128, 8192),
    (256, 32768),
]


def _dtypes():
    dtypes = [torch.float16, torch.float32]
    if flag_train.runtime.device.support_bf16:
        dtypes.append(torch.bfloat16)
    return dtypes


class BiasGeluBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _CORE_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_CORE_SHAPES)


def bias_gelu_input_fn(shape, dtype, device):
    yield (
        torch.randn(shape, dtype=dtype, device=device),
        torch.randn(shape[-1], dtype=dtype, device=device),
    )


@pytest.mark.bias_gelu_impl
def test_bias_gelu_impl_perf():
    try:
        from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
    except ImportError as exc:
        raise RuntimeError(
            "bias_gelu_impl benchmark requires installed megatron-core"
        ) from exc

    bench = BiasGeluBenchmark(
        input_fn=bias_gelu_input_fn,
        op_name="bias_gelu_impl",
        torch_op=bias_gelu_impl,
        dtypes=_dtypes(),
    )
    bench.set_train(flag_train.megatron.bias_gelu_impl)
    bench.run()
