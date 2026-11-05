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
import importlib.util
import warnings

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

_EXTENSION = "scaled_upper_triang_masked_softmax_cuda"


# ``megatron-core``'s pip wheel ships the class but not the Apex extension its
# forward imports, so probe the extension. A missing kernel leaves the run with
# no on-machine baseline rather than a torch composition.
def _load_megatron_apply():
    try:
        from megatron.core.fusions.fused_softmax import ScaledUpperTriangMaskedSoftmax
    except Exception as exc:
        warnings.warn(
            "Megatron-LM's ScaledUpperTriangMaskedSoftmax could not be imported "
            f"({exc!r}). This run has no on-machine baseline; a torch "
            "composition is not timed, so --base-data is the only comparison.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if (
        flag_train.vendor_name != "ascend"
        and importlib.util.find_spec(_EXTENSION) is None
    ):
        warnings.warn(
            "Megatron-LM's ScaledUpperTriangMaskedSoftmax needs the Apex CUDA "
            f"extension {_EXTENSION!r}, which is not installed. This run has no "
            "on-machine baseline; a torch composition is not timed, so "
            "--base-data is the only comparison.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return ScaledUpperTriangMaskedSoftmax.apply


# Resolved once, so the first-use import is not counted in the measurement.
# ``None`` is the benchmark harness's "no on-machine baseline" sentinel; the
# harness checks ``torch_op is not None`` before timing anything.
torch_op = _load_megatron_apply()
_BASELINE = (
    "megatron-core ScaledUpperTriangMaskedSoftmax"
    if torch_op is not None
    else "none (no on-machine baseline; --base-data is the only comparison)"
)


class _ScaledUpperTriangMaskedSoftmaxBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_SHAPES)


def scaled_upper_triang_masked_softmax_input_fn(shape, dtype, device):
    yield (torch.randn(shape, dtype=dtype, device=device), _SCALE)


@pytest.mark.scaled_upper_triang_masked_softmax_forward
def test_scaled_upper_triang_masked_softmax_forward():
    print(f"\nBaseline: {_BASELINE}")
    bench = _ScaledUpperTriangMaskedSoftmaxBenchmark(
        input_fn=scaled_upper_triang_masked_softmax_input_fn,
        op_name="scaled_upper_triang_masked_softmax_forward",
        # None when the Apex extension is unavailable: a torch composition is
        # not timed as a baseline, so --base-data supplies the comparison.
        torch_op=torch_op,
        dtypes=_DTYPES,
    )
    bench.set_train(flag_train.megatron.scaled_upper_triang_masked_softmax_forward)
    bench.run()
