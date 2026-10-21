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
"""Performance benchmark for Megatron Quick GELU.

The baseline is the same PyTorch composition the correctness tests use,
``x * sigmoid(1.702 * x)``, on the same device, shape and dtype. Timing,
warmup and synchronization come from ``benchmark/base.py`` (``triton.testing.do_bench``
in kernel mode). ``--level core`` runs the small, medium and large shapes below;
``--level comprehensive`` adds the extra shapes.
"""

import pytest
import torch

import flag_train
from flag_train.megatron import quick_gelu

from .. import base

# Megatron-LM coefficient. See tests/megatron/test_quick_gelu.py.
_ALPHA = 1.702

# Small, medium, and a Megatron-MLP-sized activation tensor (tokens, hidden).
_CORE_SHAPES = [
    (1,),
    (1024,),
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (4, 8192, 4096),
    (8, 4096, 11008),
]


def quick_gelu_ref(x):
    return x * torch.sigmoid(_ALPHA * x)


def _dtypes():
    dtypes = [torch.float16, torch.float32]
    if flag_train.runtime.device.support_bf16:
        dtypes.append(torch.bfloat16)
    return dtypes


class QuickGeluBenchmark(base.GenericBenchmark):
    """Shapes are fixed here because ``benchmark/core_shapes.yaml`` does not exist yet.

    ``Benchmark.set_shapes`` opens that file, so an override is required for an
    operator that is not listed in one. This matches ``LambBenchmark``.
    """

    DEFAULT_SHAPES = _CORE_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_CORE_SHAPES)


def quick_gelu_input_fn(shape, dtype, device):
    yield (torch.randn(shape, dtype=dtype, device=device),)


def _run(is_backward):
    bench = QuickGeluBenchmark(
        input_fn=quick_gelu_input_fn,
        op_name="quick_gelu_backward" if is_backward else "quick_gelu",
        torch_op=quick_gelu_ref,
        dtypes=_dtypes(),
        is_backward=is_backward,
    )
    bench.set_train(quick_gelu)
    bench.run()


@pytest.mark.quick_gelu
def test_quick_gelu_perf():
    _run(is_backward=False)


@pytest.mark.quick_gelu
def test_quick_gelu_backward_perf():
    _run(is_backward=True)
