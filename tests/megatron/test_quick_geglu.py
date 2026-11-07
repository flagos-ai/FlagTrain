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
"""Correctness tests for Megatron QUICK GEGLU.

The preferred reference is Megatron-LM's official
``megatron.core.fusions.fused_bias_geglu.quick_geglu`` interface. If
Megatron-LM is not installed, the tests log a warning and fall back to the
equivalent PyTorch composition. Under ``--ref cpu`` the reference operands
are moved to the CPU
and upcast, so the comparison is against a higher-precision CPU result rather
than a second run of the device kernel. Tolerances are
``train_assert_close``'s defaults (``atol=1e-4``, dtype rtol from
``flag_train.testing.RESOLUTION``).
"""

import warnings

import pytest
import torch

import flag_train

from .. import accuracy_utils as utils

# Empty inputs, the three n/BLOCK relations, and the last-dimension values
# (d = 1, d < BLOCK dividing or not, d == BLOCK, d > BLOCK) that drive the
# row/column mapping.
_SHAPES = [
    (0,),
    (0, 8),
    (8, 0),
    (1, 2),
    (1025, 2),
    (1024, 6),
    (2, 2048),
    (2, 2052),
    (1024,),
    (4, 8, 32, 16),
]

_COMPILE_BUDGET = 256


@pytest.fixture(scope="module", autouse=True)
def _full_compile_budget():
    config = torch._dynamo.config
    names = [
        name
        for name in ("recompile_limit", "cache_size_limit")
        if hasattr(config, name)
    ]
    saved = {name: getattr(config, name) for name in names}
    for name in names:
        setattr(config, name, max(saved[name], _COMPILE_BUDGET))
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(config, name, value)


def quick_geglu_ref(y, linear_offset=0.0):
    y_1, y_2 = torch.chunk(y, 2, dim=-1)
    return y_1 * torch.sigmoid(1.702 * y_1) * (y_2 + linear_offset)


def base_op():
    try:
        from megatron.core.fusions.fused_bias_geglu import quick_geglu
    except ImportError:
        warnings.warn(
            "Megatron-LM is unavailable; using the PyTorch Quick GEGLU "
            "composition as the correctness reference",
            RuntimeWarning,
            stacklevel=2,
        )
        return quick_geglu_ref

    chosen = quick_geglu

    def reference(*args, **kwargs):
        nonlocal chosen
        if chosen is quick_geglu_ref:
            return chosen(*args, **kwargs)
        try:
            return chosen(*args, **kwargs)
        except Exception:
            warnings.warn(
                "Megatron quick_geglu failed to compile on this platform; "
                "falling back to the PyTorch Quick GEGLU composition",
                RuntimeWarning,
                stacklevel=2,
            )
            chosen = quick_geglu_ref
            return chosen(*args, **kwargs)

    return reference


reference_quick_geglu = base_op()


def _reference(tensor):
    return utils.to_reference(tensor.detach(), upcast=True)


def _assert_close(actual, expected, dtype):
    utils.train_assert_close(utils.to_reference(actual), expected, dtype)


def _check_forward(y, linear_offset=0.0):
    out = flag_train.megatron.quick_geglu(y, linear_offset)
    assert out.shape == y.shape[:-1] + (y.shape[-1] // 2,)
    assert out.dtype == y.dtype

    y_ref = _reference(y)
    out_ref = reference_quick_geglu(y_ref, linear_offset)
    _assert_close(out, out_ref, y.dtype)


@pytest.mark.quick_geglu
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("linear_offset", [0.0, 0.5], ids=["default", "offset"])
def test_quick_geglu(shape, dtype, linear_offset):
    utils.init_seed(0)
    y = torch.randn(shape, dtype=dtype, device=flag_train.device)
    _check_forward(y, linear_offset)


@pytest.mark.quick_geglu
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_quick_geglu_values(dtype):
    """Saturated, near-zero, and landmark gate values against positive,
    negative, and zero info values."""
    utils.init_seed(0)
    values = [
        (-50.0, 1.0),
        (-8.0, -1.0),
        (-4.0, 0.5),
        (-2.0, 0.0),
        (-1.0, -0.5),
        (-1e-3, 1.0),
        (-1e-6, -1.0),
        (0.0, 0.5),
        (1e-6, 0.0),
        (1e-3, -0.5),
        (1.0, 1.0),
        (2.0, -1.0),
        (4.0, 0.5),
        (8.0, 0.0),
        (50.0, 1.0),
    ]
    y = torch.tensor(values, dtype=torch.float32, device=flag_train.device).to(dtype)
    _check_forward(y)


@pytest.mark.quick_geglu
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_quick_geglu_noncontiguous(dtype):
    utils.init_seed(1)
    base = torch.randn((32, 64), dtype=dtype, device=flag_train.device)
    y = base.t()
    assert not y.is_contiguous()
    _check_forward(y)
