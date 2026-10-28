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

import flag_train
from flag_train.megatron import tanh_soft_clamp

from .. import accuracy_utils as utils


def tanh_soft_clamp_ref(x, scale):
    return (scale * torch.tanh(x.float() / scale)).to(x.dtype)


def _dtype_id(dtype):
    return str(dtype).removeprefix("torch.")


_DTYPES = list(utils.FLOAT_DTYPES)
if utils.fp64_is_supported:
    _DTYPES.append(torch.float64)


def _assert_matches(out, x, scale, dtype):
    ref = tanh_soft_clamp_ref(utils.to_reference(x), scale)
    assert out.shape == x.shape
    assert out.dtype == dtype
    utils.train_assert_close(utils.to_reference(out), ref, dtype)


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize(
    "shape",
    [(1,), (1024,), (7, 13), (20, 320, 15), (16, 128, 64, 60)],
)
@pytest.mark.parametrize("dtype", _DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("scale", [1.0, 0.5, 4.0, -2.5, 1e-3, 1e3])
def test_matches_torch(shape, dtype, scale):
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    before = x.clone()

    out = tanh_soft_clamp(x, scale)

    _assert_matches(out, x, scale, dtype)
    assert torch.equal(x, before)
    assert out.data_ptr() != x.data_ptr()


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize("scale", [1.0, -2.0, 0.0])
def test_special_values(scale):
    values = [
        0.0,
        1.0,
        -1.0,
        1e30,
        -1e30,
        float("inf"),
        float("-inf"),
        float("nan"),
    ]
    x = torch.tensor(values, dtype=torch.float32, device=flag_train.device)
    out = tanh_soft_clamp(x, scale)
    ref = tanh_soft_clamp_ref(utils.to_reference(x), scale)

    utils.train_assert_close(
        utils.to_reference(out), ref, torch.float32, equal_nan=True
    )


_NON_CONTIGUOUS_LAYOUTS = [
    ("transposed", lambda t: t.transpose(0, 1)),
    ("permuted", lambda t: t.permute(2, 0, 1)),
    ("strided", lambda t: t[:, ::2, :]),
    ("sliced", lambda t: t[::2]),
]


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize("layout", _NON_CONTIGUOUS_LAYOUTS, ids=lambda case: case[0])
def test_non_contiguous(layout):
    _, make = layout
    base = torch.randn((32, 64, 128), dtype=torch.float32, device=flag_train.device)
    x = make(base)
    assert not x.is_contiguous()

    out = tanh_soft_clamp(x, 0.5)

    _assert_matches(out, x, 0.5, torch.float32)


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize("shape", [(), (0,), (0, 4), (1,)])
def test_shape_edges(shape):
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)

    out = tanh_soft_clamp(x, 1.0)

    _assert_matches(out, x, 1.0, dtype)
    assert out.numel() == x.numel()


@pytest.mark.tanh_soft_clamp
@pytest.mark.skipif(not utils.fp64_is_supported, reason="backend lacks float64")
def test_float64_follows_torch_contract():
    x = torch.linspace(-8.0, 8.0, 4096, dtype=torch.float64, device=flag_train.device)

    out = tanh_soft_clamp(x, 1.0)

    _assert_matches(out, x, 1.0, torch.float64)


try:
    from megatron.core.activations import tanh_soft_clamp as _megatron_tanh_soft_clamp
except Exception:
    _megatron_tanh_soft_clamp = None

requires_megatron = pytest.mark.skipif(
    _megatron_tanh_soft_clamp is None,
    reason="megatron-core is not importable; the upstream source is unavailable",
)


@requires_megatron
@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize("dtype", _DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_matches_megatron(dtype, scale):
    x = torch.randn((4096,), dtype=dtype, device=flag_train.device)

    out = tanh_soft_clamp(x, scale)
    ref = _megatron_tanh_soft_clamp(utils.to_reference(x), scale)

    utils.train_assert_close(utils.to_reference(out), ref, dtype)
