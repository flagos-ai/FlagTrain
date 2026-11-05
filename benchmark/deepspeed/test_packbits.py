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
"""Performance benchmark for packbits.

There is no on-machine baseline: DeepSpeed's ``pack_bits`` was XPU-only and was
deleted upstream (8ae0509ef), and torch has no packbits. A composition of torch
primitives is not a competitor, so it is not timed: ``torch_op`` is ``None``,
the benchmark reports Train latency only, and ``--base-data`` is the only
comparison.
"""

import pytest
import torch

import flag_train

from .. import base, consts

_PACKBITS_SHAPES = [
    (4096,),
    (16384,),
    (65536,),
    (262144,),
    (1048576,),
    (4194304,),
    (16777216,),
    (1000003,),
    (1024, 1024),
    (4096, 4096),
    (11008, 1024),
    (128256, 128),
    (512, 4097),
    (32000, 8),
]

_BASELINE = "none (no on-machine baseline; --base-data is the only comparison)"


class _PackbitsBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _PACKBITS_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_PACKBITS_SHAPES)


def packbits_input_fn(shape, dtype, device):
    yield (torch.randn(shape, dtype=dtype, device=device),)


@pytest.mark.packbits
def test_packbits_perf():
    print(f"\nBaseline: {_BASELINE}")
    bench = _PackbitsBenchmark(
        input_fn=packbits_input_fn,
        op_name="packbits",
        # None: DeepSpeed's operator is gone and a torch composition is not
        # timed as a baseline, so --base-data supplies the comparison.
        torch_op=None,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_train(flag_train.deepspeed.packbits)
    bench.run()
