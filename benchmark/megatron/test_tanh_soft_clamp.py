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


import warnings

import pytest
import torch

import flag_train

from .. import base

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

_SCALE = 0.5


def tanh_soft_clamp_ref(x, scale):
    return (scale * torch.tanh(x.float() / scale)).to(x.dtype)


def _load_mcore_tanh_soft_clamp():
    try:
        import megatron.core.activations
    except ImportError:
        warnings.warn(
            "Megatron-LM is unavailable; using the PyTorch Tanh Soft Clamp "
            "composition as the benchmark baseline",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    op = getattr(megatron.core.activations, "tanh_soft_clamp", None)
    if op is None:
        warnings.warn(
            "Megatron-LM predates tanh_soft_clamp (needs megatron-core >= "
            "0.20.0); using the PyTorch Tanh Soft Clamp composition as the "
            "benchmark baseline",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return op


# Resolved once, so the baseline is the same one ``native_baseline`` describes.
_mcore_tanh_soft_clamp = _load_mcore_tanh_soft_clamp()


def base_op():
    return _mcore_tanh_soft_clamp or tanh_soft_clamp_ref


def _dtypes():
    dtypes = [torch.float16, torch.float32]
    if flag_train.runtime.device.support_bf16:
        dtypes.append(torch.bfloat16)
    return dtypes


class TanhSoftClampBenchmark(base.GenericBenchmark):

    DEFAULT_SHAPES = _CORE_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_CORE_SHAPES)


def tanh_soft_clamp_input_fn(shape, dtype, device):
    yield (torch.randn(shape, dtype=dtype, device=device), _SCALE)


@pytest.mark.tanh_soft_clamp
def test_tanh_soft_clamp_perf():
    bench = TanhSoftClampBenchmark(
        input_fn=tanh_soft_clamp_input_fn,
        op_name="tanh_soft_clamp",
        torch_op=base_op(),
        dtypes=_dtypes(),
        # Megatron's own activation is a real local baseline; the torch
        # composition is not, and --base-data still applies to it.
        native_baseline=_mcore_tanh_soft_clamp is not None,
    )
    bench.set_train(flag_train.megatron.tanh_soft_clamp)
    bench.run()
