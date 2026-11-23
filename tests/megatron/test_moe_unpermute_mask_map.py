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


"""Test the moe_unpermute_mask_map operator against its Torch reference."""

from dataclasses import dataclass

import pytest
import torch

import flag_train
from flag_train.reference_candidate import get_candidate


def reference(
    inp, row_id_map, merging_probs, num_tokens, num_experts, hidden_size, pad_offsets
):
    if not inp.numel():
        return inp.clone()
    dst = row_id_map[:, :num_experts].long()
    exp = row_id_map[:, num_experts : 2 * num_experts].long()
    active = torch.arange(num_experts, device=inp.device)[None] < row_id_map[:, -1:]
    exp = exp.masked_fill(~active, 0)
    if pad_offsets is not None:
        dst = dst + pad_offsets[exp]
    dst = dst.masked_fill(~active, 0)
    values = inp[dst].float()
    if merging_probs is not None:
        values = values * merging_probs.gather(1, exp).float().unsqueeze(-1)
    return values.masked_fill(~active.unsqueeze(-1), 0).sum(dim=1).to(inp.dtype)


def permute_mask_map(inp, routing_map, num_out_tokens, probs, pad_offsets):
    tokens, experts = routing_map.shape
    row_map = torch.zeros(
        (tokens, 2 * experts + 1), dtype=torch.int32, device=inp.device
    )
    if not inp.numel():
        return (
            inp.clone(),
            row_map,
            inp.new_empty(0, dtype=probs.dtype if probs is not None else torch.float32),
        )
    dest = (routing_map.T.flatten().cumsum(0) - 1).reshape(experts, tokens).T
    dest = dest.masked_fill((routing_map == 0) | (dest >= num_out_tokens), -1)
    ordered, expert = dest.sort(dim=1, descending=True)
    row_map[:, :experts] = ordered
    row_map[:, experts : 2 * experts] = expert
    row_map[:, -1] = (ordered >= 0).sum(dim=1)
    src, exp = (dest >= 0).nonzero(as_tuple=True)
    dst = dest[src, exp]
    if pad_offsets is not None:
        dst = dst + pad_offsets[exp]
    output = inp.new_zeros((num_out_tokens, inp.size(1)))
    values = inp[src]
    moved_probs = inp.new_empty(0, dtype=torch.float32)
    if probs is not None:
        gathered = probs[src, exp]
        values = torch.where(gathered[:, None] != 0, values, 0)
        moved_probs = probs.new_zeros(num_out_tokens)
        moved_probs[dst] = gathered
    output[dst] = values
    return output, row_map, moved_probs


OP_NAME = "moe_unpermute_mask_map"
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
    Case(19, 33, "unweighted"),
    Case(1024, 128, "padded"),
    Case(17, 7, "weighted", True),
    Case(8, 16, "unweighted"),
    Case(33, 64, "padded"),
    Case(64, 128, "unweighted"),
    Case(129, 96, "padded"),
    Case(256, 256, "unweighted"),
    Case(512, 256, "padded"),
)


def make_inputs(case, dtype, device):
    n, h, mode = (case.rows, case.width, case.variant)
    gen = torch.Generator().manual_seed(4234321)

    def rand(*shape, as_type=dtype):
        return torch.randn(shape, generator=gen).to(device, as_type)

    def ints(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    e = 4
    inp = rand(n, h)
    token = torch.arange(n, device=device)[:, None]
    expert = torch.arange(e, device=device)[None]
    routing = (((token + expert) % e < 2) & (token % 7 != 0)).int()
    probs = rand(n, e) if mode != "unweighted" else None
    if probs is not None and n:
        probs[1] = 0
    count = 2 * (n - (n + 6) // 7)
    pads = ints([0, 1, 3, 6]) if mode == "padded" else None
    size = count + 8 if pads is not None else count
    if mode == "truncated":
        size -= 3
    permuted, row_map, _ = permute_mask_map(inp, routing, size, probs, pads)
    return (permuted, row_map, probs, n, e, h, pads)


def _review_inputs(case, device):

    def tensor(values, dtype=torch.float32):
        return torch.tensor(values, device=device, dtype=dtype)

    return (
        tensor([[1, 2], [3, 4]]).T,
        tensor([[0, 0, 1], [1, 0, 1]], torch.int32),
        None,
        2,
        1,
        2,
        None,
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
        torch.testing.assert_close(arg, snapshot, rtol=0, atol=0)
    return outputs


def _compare(actual, expected):
    assert len(actual) == len(expected)
    for i, (got, wanted) in enumerate(zip(actual, expected)):
        assert (got.shape, got.dtype, got.device) == (
            wanted.shape,
            wanted.dtype,
            wanted.device,
        )
        torch.testing.assert_close(got, wanted)


def _candidate():
    return get_candidate(f"megatron.{OP_NAME}", reference=reference)


@pytest.mark.moe_unpermute_mask_map
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("dtype", DTYPES)
def test_moe_unpermute_mask_map(case, dtype):
    _compare(
        _observe(_candidate(), make_inputs(case, dtype, flag_train.device)),
        _observe(reference, make_inputs(case, dtype, flag_train.device)),
    )


@pytest.mark.moe_unpermute_mask_map
@pytest.mark.parametrize("case", ("strided-activation",))
def test_moe_unpermute_mask_map_review_regressions(case):
    _compare(
        _observe(_candidate(), _review_inputs(case, flag_train.device)),
        _observe(reference, _review_inputs(case, flag_train.device)),
    )


@pytest.mark.moe_unpermute_mask_map
def test_unpermute_rechecks_layout_after_transpose():
    fn = _candidate()
    args = _review_inputs("strided-activation", flag_train.device)
    fn(*args)
    args[0].transpose_(0, 1)
    expected = reference(*args)
    actual = fn(*args)
    torch.testing.assert_close(actual, expected)


@pytest.mark.moe_unpermute_mask_map
def test_moe_unpermute_no_routed_input():
    device = flag_train.device
    args = (
        torch.empty((0, 5), device=device),
        torch.zeros((3, 3), device=device, dtype=torch.int32),
        None,
        3,
        1,
        5,
        None,
    )
    _compare(_observe(_candidate(), args), _observe(reference, args))
