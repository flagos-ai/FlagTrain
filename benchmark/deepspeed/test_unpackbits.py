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
"""Performance benchmark for unpackbits.

There is no on-machine baseline: DeepSpeed's ``unpack_bits`` was XPU-only and
was deleted upstream (8ae0509ef), and torch has no unpackbits. A composition of
torch primitives is not a competitor, so it is not timed: ``torch_op`` is
``None``, the benchmark reports Train latency only, and ``--base-data`` is the
only comparison.
"""

import pytest
import torch

import flag_train

from .. import base, consts

_UNPACKBITS_SHAPES = [
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


class _UnpackbitsBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _UNPACKBITS_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_UNPACKBITS_SHAPES)


def unpackbits_input_fn(shape, dtype, device):
    n_bytes = (shape[-1] + 7) // 8
    packed = torch.randint(
        0, 256, (*shape[:-1], n_bytes), dtype=torch.uint8, device=device
    )
    yield packed, {"dtype": dtype}


@pytest.mark.unpackbits
def test_unpackbits_perf():
    print(f"\nBaseline: {_BASELINE}")
    bench = _UnpackbitsBenchmark(
        input_fn=unpackbits_input_fn,
        op_name="unpackbits",
        # None: DeepSpeed's operator is gone and a torch composition is not
        # timed as a baseline, so --base-data supplies the comparison.
        torch_op=None,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_train(flag_train.deepspeed.unpackbits)
    bench.run()
