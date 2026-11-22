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


"""Test the fused_mla_rope_inplace operator against its Torch reference."""

from dataclasses import dataclass

import flag_train
import pytest
import torch
from flag_train.reference_candidate import get_candidate


def reference(
    t,
    cos,
    sin,
    nope_dim,
    emb_dim,
    cu_seqlens_q=None,
    cp_rank=0,
    cp_size=1,
    rotary_interleaved=False,
    inverse=False,
    remove_interleaving=False,
    position_ids=None,
):
    if rotary_interleaved:
        raise ValueError("The native entry point requires rotary_interleaved=False")
    if position_ids is not None:
        c, s = cos[position_ids, 0], sin[position_ids, 0]
    elif cu_seqlens_q is None:
        c, s = cos[: t.size(0)], sin[: t.size(0)]
    else:
        positions = torch.arange(t.size(0), device=t.device)
        ends = cu_seqlens_q.long() // cp_size
        seq = torch.searchsorted(ends[1:], positions, right=True).clamp(
            max=len(ends) - 2
        )
        local = positions - ends[seq]
        length = ends[seq + 1] - ends[seq]
        half = length // 2
        index = (
            local
            if cp_size == 1
            else torch.where(
                local < half,
                local + cp_rank * length // 2,
                local - half + (2 * cp_size - cp_rank - 1) * length // 2,
            )
        )
        index = index.masked_fill(positions >= ends[-1], 0)
        c, s = cos[index, 0], sin[index, 0]
    s = -s if inverse else s
    x, y = t[..., nope_dim::2].float(), t[..., nope_dim + 1 :: 2].float()
    cl, cr = c.float().chunk(2, dim=-1)
    sl, sr = s.float().chunk(2, dim=-1)
    left, right = x * cl - y * sl, y * cr + x * sr
    value = (
        torch.stack((left, right), dim=-1).flatten(-2)
        if remove_interleaving
        else torch.cat((left, right), dim=-1)
    )
    t[..., nope_dim : nope_dim + emb_dim].copy_(value)
    return t


OP_NAME = "fused_mla_rope_inplace"
pytestmark = pytest.mark.skipif(
    flag_train.vendor_name != "nvidia", reason="NVIDIA kernel validation"
)
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
    Case(19, 64, "dense"),
    Case(128, 64, "packed"),
    Case(18, 32, "inverse-cp", True),
    Case(18, 32, "positions", True),
    Case(8, 16, "dense"),
    Case(17, 32, "packed"),
    Case(33, 64, "dense"),
    Case(64, 16, "packed"),
    Case(128, 32, "dense"),
    Case(256, 64, "packed"),
)


def make_inputs(case, dtype, device):
    n, h, mode = (case.rows, case.width, case.variant)
    gen = torch.Generator().manual_seed(4234321)

    def rand(*shape, as_type=dtype):
        return torch.randn(shape, generator=gen).to(device, as_type)

    def ints(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    dense = mode == "dense"
    shape = (n, 2, 3, h + 8) if dense else (n, 3, h + 8)
    lengths = [0, n // 2, n]
    if mode == "inverse-cp":
        lengths = [0, 16, 32]
    position_ids = (
        (2 * torch.arange(n, device=device)).int() if mode == "positions" else None
    )
    angles = rand(2 * n, 1, 1, h, as_type=torch.float32)
    return (
        rand(*shape),
        angles.cos(),
        angles.sin(),
        8,
        h,
        None if dense else ints(lengths),
        1 if mode == "inverse-cp" else 0,
        2 if mode == "inverse-cp" else 1,
        False,
        mode == "inverse-cp",
        mode in ("packed", "positions"),
        position_ids,
    )


def _review_inputs(case, device):

    def tensor(values, dtype=torch.float32):
        return torch.tensor(values, device=device, dtype=dtype)

    shape = (2, 1, 1, 4)
    values = torch.arange(8, device=device, dtype=torch.float32).view(shape)
    return (
        values,
        torch.zeros(shape, device=device),
        torch.ones(shape, device=device),
        0,
        4,
        None,
        0,
        1,
        False,
        False,
        False,
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
    alias_first = True
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
        if i == 0:
            cutoff = args[3]
            torch.testing.assert_close(
                arg[..., :cutoff], snapshot[..., :cutoff], rtol=0, atol=0
            )
            assert not torch.equal(arg[..., cutoff:], snapshot[..., cutoff:])
        else:
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


@pytest.mark.fused_mla_rope_inplace
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_mla_rope_inplace(case, dtype):
    _compare(
        _observe(_candidate(), make_inputs(case, dtype, flag_train.device)),
        _observe(reference, make_inputs(case, dtype, flag_train.device)),
    )


@pytest.mark.fused_mla_rope_inplace
@pytest.mark.parametrize("case", ("dense-rotation",))
def test_fused_mla_rope_inplace_review_regressions(case):
    _compare(
        _observe(_candidate(), _review_inputs(case, flag_train.device)),
        _observe(reference, _review_inputs(case, flag_train.device)),
    )


@pytest.mark.fused_mla_rope_inplace
@pytest.mark.parametrize("remove_interleaving", [False, True])
def test_mla_backward(remove_interleaving):
    args = list(_review_inputs("dense-rotation", flag_train.device))
    args[10] = remove_interleaving
    x = args[0].requires_grad_()
    args[0] = x * 1
    out = _candidate()(*args)
    (grad,) = torch.autograd.grad(out[..., 0].sum(), x)
    expected = torch.zeros_like(x)
    expected[..., 1] = -1
    torch.testing.assert_close(grad, expected, rtol=0, atol=0)


@pytest.mark.fused_mla_rope_inplace
def test_fused_mla_rope_inplace_on_noncurrent_cuda_device():
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    with torch.cuda.device(0):
        got = _observe(_candidate(), _review_inputs("dense-rotation", "cuda:1"))
        expected = _observe(reference, _review_inputs("dense-rotation", "cuda:1"))
        assert torch.cuda.current_device() == 0
    _compare(got, expected)
