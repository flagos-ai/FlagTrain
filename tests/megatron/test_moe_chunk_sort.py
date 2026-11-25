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


"""Test the moe_chunk_sort operator against its Torch reference."""

from dataclasses import dataclass
from itertools import accumulate

import pytest
import torch

import flag_train
from flag_train.reference_candidate import get_candidate
from flag_train.testing import assert_equal


def reference(inp, split_sizes, sorted_idxs, probs):
    if not inp.numel():
        reordered = (
            inp.new_empty(0, dtype=torch.float32) if probs is None else probs.clone()
        )
        return inp.clone(), reordered, inp.new_empty(0, dtype=torch.int32)
    sizes, order = split_sizes.tolist(), sorted_idxs.tolist()
    starts = [0, *list(accumulate(sizes))]
    permutation = torch.tensor(
        [i for group in order for i in range(starts[group], starts[group + 1])]
        + list(range(starts[-1], inp.size(0))),
        device=inp.device,
        dtype=torch.long,
    )
    return (
        inp[permutation],
        (
            probs[permutation]
            if probs is not None
            else inp.new_empty(0, dtype=torch.float32)
        ),
        permutation.argsort().int(),
    )


OP_NAME = "moe_chunk_sort"
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
    Case(19, 33, "weighted"),
    Case(1024, 128, "unweighted"),
    Case(17, 7, "tail", True),
    Case(0, 7, "empty", True),
    Case(8, 16, "weighted"),
    Case(33, 64, "unweighted"),
    Case(64, 128, "weighted"),
    Case(129, 96, "unweighted"),
    Case(256, 256, "weighted"),
    Case(512, 256, "unweighted"),
)


def make_inputs(case, dtype, device):
    n, h, mode = (case.rows, case.width, case.variant)
    gen = torch.Generator().manual_seed(4234321)

    def rand(*shape, as_type=dtype):
        return torch.randn(shape, generator=gen).to(device, as_type)

    def ints(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    sizes = [n // 3, 0, n // 4, n - n // 3 - n // 4]
    if mode == "tail":
        sizes[-1] -= 2
    return (
        rand(n, h),
        ints(sizes),
        ints([3, 2, 1, 0]),
        None if mode == "unweighted" else rand(n),
    )


def _review_inputs(case, device):

    def tensor(values, dtype=torch.float32):
        return torch.tensor(values, device=device, dtype=dtype)

    return (
        tensor([[1, 2], [3, 4], [5, 6]]),
        tensor([1, 2], torch.int32),
        tensor([1, 0], torch.int32),
        tensor([0.1, 99, 0.2, 99, 0.3])[::2],
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
        assert_equal(got, wanted)


def _candidate():
    return get_candidate(f"megatron.{OP_NAME}", reference=reference)


@pytest.mark.moe_chunk_sort
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("dtype", DTYPES)
def test_moe_chunk_sort(case, dtype):
    _compare(
        _observe(_candidate(), make_inputs(case, dtype, flag_train.device)),
        _observe(reference, make_inputs(case, dtype, flag_train.device)),
    )


@pytest.mark.moe_chunk_sort
@pytest.mark.parametrize("case", ("strided-probabilities",))
def test_moe_chunk_sort_review_regressions(case):
    _compare(
        _observe(_candidate(), _review_inputs(case, flag_train.device)),
        _observe(reference, _review_inputs(case, flag_train.device)),
    )


@pytest.mark.moe_chunk_sort
def test_moe_chunk_sort_offset_and_fresh_empty():
    device = flag_train.device
    fn = _candidate()
    splits = torch.tensor([1, 2], device=device, dtype=torch.int32)
    order = torch.tensor([1, 0], device=device, dtype=torch.int32)
    base = torch.arange(3 * 128 + 1, device=device, dtype=torch.float32)
    first = fn(base[:-1].view(3, 128), splits, order, None)
    first[1].resize_(1)
    args = (base[1:].view(3, 128), splits, order, None)
    _compare(fn(*args), reference(*args))


@pytest.mark.moe_chunk_sort
def test_moe_chunk_sort_zero_hidden_width():
    device = flag_train.device
    args = (
        torch.empty((2, 0), device=device),
        torch.tensor([1, 1], dtype=torch.int32, device=device),
        torch.tensor([1, 0], dtype=torch.int32, device=device),
        torch.tensor([3.0, 4.0], device=device),
    )
    _compare(_observe(_candidate(), args), _observe(reference, args))
