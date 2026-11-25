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


"""Test the grouped_linear operator against its Torch reference."""

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

import flag_train
from flag_train.reference_candidate import get_candidate
from flag_train.runtime import torch_device_fn
from flag_train.testing import assert_close, assert_equal


def _is_ppu(tensor):
    return flag_train.vendor_name == "thead" and tensor.device.type == flag_train.device


def _tf32_rne(a):
    # The native FP32 cuBLASLt call rounds both multiplicands to TF32.
    # Emulate that rounding on every backend without changing global matmul flags.
    bits = a.contiguous().view(torch.int32)
    rounded = ((bits + 0xFFF + ((bits >> 13) & 1)) & ~0x1FFF).view(torch.float32)
    return torch.where(torch.isfinite(a), rounded, a)


def _ppu_linear(inp, weight, bias):
    # Use separate FP32 matmul and bias addition. Cast the result once so
    # FP16/BF16 GEMM and bias still round together.
    result = torch.mm(inp.float(), weight.float().T)
    if bias is not None:
        result = result + bias.float()
    return result.to(inp.dtype)


def reference(inp, m_splits, weights, biases=None):
    if inp.dtype == torch.float32:
        inp, weights = _tf32_rne(inp), _tf32_rne(weights)
    if isinstance(m_splits, torch.Tensor):
        m_splits = m_splits.tolist()
    chunks = inp.split(m_splits)
    linear = _ppu_linear if _is_ppu(inp) else F.linear
    return torch.cat(
        [
            linear(chunk, weights[i], None if biases is None else biases[i])
            for i, chunk in enumerate(chunks)
        ]
    )


OP_NAME = "grouped_linear"
DTYPES = (torch.float32, torch.bfloat16, torch.float16)


@dataclass(frozen=True)
class Case:
    rows: int
    width: int
    variant: str
    correctness_only: bool = False

    @property
    def id(self):
        return f"{self.rows}x{self.width}-{self.variant}"


CASES = (
    Case(64, 128, "no-bias"),
    Case(1024, 256, "biased"),
    Case(32, 64, "empty-group", True),
    Case(64, 64, "biased"),
    Case(128, 128, "no-bias"),
    Case(256, 64, "biased"),
    Case(256, 256, "no-bias"),
    Case(512, 128, "biased"),
    Case(512, 256, "no-bias"),
)


def make_inputs(case, dtype, device):
    n, h, mode = (case.rows, case.width, case.variant)
    gen = torch.Generator().manual_seed(4234321)

    def rand(*shape, as_type=dtype):
        return torch.randn(shape, generator=gen).to(device, as_type)

    splits = (
        [n // 4, 0, n * 3 // 4] if mode == "empty-group" else [n // 4, n // 4, n // 2]
    )
    return (
        rand(n, h),
        splits,
        rand(3, h // 2, h),
        None if mode == "no-bias" else rand(3, h // 2),
    )


def _review_inputs(case, device):

    def tensor(values, dtype=torch.float32):
        return torch.tensor(values, device=device, dtype=dtype)

    return (
        tensor([[1, 2], [3, 4]]).T,
        tensor([1, 99, 1, 99], torch.int32)[::2],
        tensor([[[3, 2], [5, 1]], [[1, 3], [0, 2]]]).transpose(1, 2),
        tensor([[2, 3], [4, 5]]).T,
    )


def _tensor_outputs(result):
    return result if isinstance(result, tuple) else (result,)


def _assert_alias(output, input, aliases):
    original = input.untyped_storage()
    actual = output.untyped_storage()
    same_storage = (
        actual.data_ptr() == original.data_ptr()
        if output.numel() and input.numel()
        else torch._C._is_alias_of(output, input)
    )
    assert same_storage is aliases
    if aliases:
        assert (output.data_ptr(), output.stride(), output.storage_offset()) == (
            input.data_ptr(),
            input.stride(),
            input.storage_offset(),
        )


def _layout(tensor):
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.data_ptr(),
        tensor.shape,
        tensor.stride(),
        tensor.dtype,
        tensor.device,
    )


def _observe(fn, args):
    originals = tuple(
        (
            (
                x.detach().clone()
                if torch.is_tensor(x)
                else x.copy() if isinstance(x, list) else x
            )
            for x in args
        )
    )
    layouts = tuple((_layout(x) if torch.is_tensor(x) else None for x in args))
    outputs = _tensor_outputs(fn(*args))
    alias_first = False
    for i, result in enumerate(outputs):
        assert isinstance(result, torch.Tensor)
        for j, value in enumerate(args):
            if torch.is_tensor(value):
                _assert_alias(result, value, alias_first and i == j == 0)
    for i, (arg, snapshot) in enumerate(zip(args, originals)):
        if not torch.is_tensor(arg):
            assert arg == snapshot
            continue
        assert _layout(arg) == layouts[i], "Input was rebound or resized"
        assert_equal(arg, snapshot)
    return outputs


def _compare(actual, expected):
    assert len(actual) == len(expected)
    for i, (got, wanted) in enumerate(zip(actual, expected)):
        assert (got.shape, got.dtype, got.device) == (
            wanted.shape,
            wanted.dtype,
            wanted.device,
        )
        assert_close(got, wanted, wanted.dtype)


def _candidate():
    return get_candidate(f"megatron.{OP_NAME}", reference=reference)


@pytest.mark.grouped_linear
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("dtype", DTYPES)
def test_grouped_linear(case, dtype):
    _compare(
        _observe(_candidate(), make_inputs(case, dtype, flag_train.device)),
        _observe(reference, make_inputs(case, dtype, flag_train.device)),
    )


@pytest.mark.grouped_linear
@pytest.mark.parametrize("case", ("strided-splits-and-activation",))
def test_grouped_linear_review_regressions(case):
    _compare(
        _observe(_candidate(), _review_inputs(case, flag_train.device)),
        _observe(reference, _review_inputs(case, flag_train.device)),
    )


@pytest.mark.grouped_linear
def test_grouped_linear_large_row_offset():
    candidate = _candidate()
    device = torch.device(flag_train.device)
    try:
        free_memory, _ = torch_device_fn.mem_get_info(device)
    except (AttributeError, RuntimeError):
        pytest.skip("Cannot query free memory on the test device")
    if free_memory < 8 * 1024**3:
        pytest.skip("Requires at least 8 GiB free on the test device")
    inputs = torch.zeros((524289, 4096), device=device, dtype=torch.float16)
    inputs[-1, 0] = 1
    weights = inputs.new_ones((1, 1, 4096))
    actual = candidate(inputs, [len(inputs)], weights)
    expected = inputs.new_zeros((len(inputs), 1))
    expected[-1, 0] = 1
    assert_equal(actual, expected)


@pytest.mark.grouped_linear
@pytest.mark.parametrize(
    "first_dtype,second_dtype", [(torch.int64, torch.int32), (torch.int32, torch.int64)]
)
@pytest.mark.parametrize("host_splits", [False, True], ids=["device", "cpu"])
def test_grouped_linear_switch_split_dtype(first_dtype, second_dtype, host_splits):
    device = flag_train.device
    inp = torch.arange(128, dtype=torch.float32, device=device).reshape(8, 16)
    weights = torch.ones((2, 4, 16), device=device)
    fn = _candidate()
    splits_device = "cpu" if host_splits else device
    for dtype in (first_dtype, second_dtype):
        args = (inp, torch.tensor([3, 5], dtype=dtype, device=splits_device), weights)
        _compare(_observe(fn, args), _observe(reference, args))
