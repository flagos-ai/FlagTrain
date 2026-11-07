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

from .. import accuracy_utils as utils

_SHAPES = [
    (16, 16, 16),
    (16, 32, 32),
    (8, 33, 33),
    (8, 64, 64),
    (16, 96, 96),
    (32, 128, 128),
    (8, 256, 256),
    (16, 512, 512),
    (8, 1024, 1024),
    (8, 2048, 2048),
    (8, 4096, 4096),
    (8, 4160, 4160),
    (8, 8192, 8192),
]

# Megatron's CUDA fused kernel rejects fp32.
_DTYPES = [torch.float16, torch.bfloat16]

# On Ascend the Megatron reference calls torch_npu.npu_scaled_masked_softmax,
# which supports fp32.
if flag_train.vendor_name == "ascend":
    _DTYPES.append(torch.float32)

_SCALES = [0.0, 0.125, 1.0, -0.5, 2]

_EXTENSION = "scaled_upper_triang_masked_softmax_cuda"


# ``megatron-core``'s pip wheel ships the class but not the Apex extension its
# forward imports, so probe the extension and fall back to the torch composition.
def _load_megatron_apply():
    try:
        from megatron.core.fusions.fused_softmax import ScaledUpperTriangMaskedSoftmax
    except Exception as exc:
        warnings.warn(
            "Megatron-LM's ScaledUpperTriangMaskedSoftmax could not be imported "
            f"({exc!r}); using scaled_upper_triang_masked_softmax_ref.",
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
            f"extension {_EXTENSION!r}, which is not installed; using "
            "scaled_upper_triang_masked_softmax_ref.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return ScaledUpperTriangMaskedSoftmax.apply


_MEGATRON_APPLY = _load_megatron_apply()
_BASELINE = (
    "megatron-core ScaledUpperTriangMaskedSoftmax"
    if _MEGATRON_APPLY is not None
    else "scaled_upper_triang_masked_softmax_ref (torch)"
)


def scaled_upper_triang_masked_softmax_ref(x, scale):
    scores = x.to(torch.float32) * scale
    rows = torch.arange(x.shape[-2], device=x.device).unsqueeze(-1)
    cols = torch.arange(x.shape[-1], device=x.device).unsqueeze(0)
    masked = cols > rows
    scores = scores.masked_fill(masked, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return probs.masked_fill(masked, 0.0).to(x.dtype)


def _reference(x, scale):
    if _MEGATRON_APPLY is None:
        return scaled_upper_triang_masked_softmax_ref(x, scale)
    # The fused kernel reads dense rows; the composition does not need the copy.
    return _MEGATRON_APPLY(x.contiguous(), scale)


@pytest.mark.scaled_upper_triang_masked_softmax_forward
def test_baseline_source():
    assert _BASELINE
    print(f"\nBaseline: {_BASELINE}")


@pytest.mark.scaled_upper_triang_masked_softmax_forward
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("scale", _SCALES)
def test_scaled_upper_triang_masked_softmax_forward(shape, dtype, scale):
    utils.init_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)

    ref = _reference(x, scale)
    res = flag_train.megatron.scaled_upper_triang_masked_softmax_forward(x, scale)

    # Megatron needs device inputs and returns device tensors; only the
    # reference follows ``--ref cpu``.
    if utils.TO_CPU:
        ref = utils.to_reference(ref)

    utils.train_assert_close(res, ref, dtype)


@pytest.mark.scaled_upper_triang_masked_softmax_forward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_scaled_upper_triang_masked_softmax_forward_values(dtype):
    if flag_train.vendor_name == "ascend":
        pytest.skip(
            "Megatron's Ascend reference is the legacy ScaledMaskedSoftmax, which "
            "fills masked positions with -10000 (not dtype min) and turns a masked "
            "-inf into NaN; the -1e4/-inf inputs here hit both."
        )

    n = 96
    wide = torch.linspace(
        -1e4, 1e4, 16 * n * n, dtype=torch.float32, device=flag_train.device
    )
    wide = wide.reshape(16, n, n).to(dtype)

    equal = torch.full((16, 32, 32), 1e4, dtype=dtype, device=flag_train.device)

    neg_inf = torch.zeros((16, 32, 32), dtype=dtype, device=flag_train.device)
    neg_inf[0, 5, 3] = float("-inf")
    neg_inf[1, 20, 25] = float("-inf")

    for x in (wide, equal, neg_inf):
        ref = _reference(x, 1.0)
        res = flag_train.megatron.scaled_upper_triang_masked_softmax_forward(x, 1.0)

        if utils.TO_CPU:
            ref = utils.to_reference(ref)

        utils.train_assert_close(res, ref, dtype)


@pytest.mark.scaled_upper_triang_masked_softmax_forward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_scaled_upper_triang_masked_softmax_forward_non_contiguous(dtype):
    utils.init_seed(2)

    base = torch.randn((64, 8, 64), dtype=dtype, device=flag_train.device)
    transposed = base.transpose(0, 1)

    wide = torch.randn((16, 32, 64), dtype=dtype, device=flag_train.device)
    strided = wide[..., ::2]

    for x in (transposed, strided):
        assert not x.is_contiguous()

        ref = _reference(x, 0.125)
        res = flag_train.megatron.scaled_upper_triang_masked_softmax_forward(x, 0.125)

        if utils.TO_CPU:
            ref = utils.to_reference(ref)

        utils.train_assert_close(res, ref, dtype)
