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


"""Test the moe_aux_loss_auto_scaler_backward operator against its Torch reference."""

from dataclasses import dataclass

import pytest
import torch

import flag_train
from flag_train.testing import assert_equal


def reference(grad_output, aux_loss, loss_scale=None):
    if loss_scale is None:
        loss_scale = torch.tensor(1.0, device=aux_loss.device)
    return grad_output, torch.ones_like(aux_loss) * loss_scale


OP_NAME = "moe_aux_loss_auto_scaler_backward"
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
    Case(17, 64, "default"),
    Case(1024, 256, "scale"),
    Case(3, 7, "broadcast", True),
    Case(1, 32, "default"),
    Case(32, 128, "scale"),
    Case(128, 128, "default"),
    Case(256, 256, "scale"),
    Case(512, 512, "default"),
    Case(1024, 1024, "scale"),
)


def make_inputs(case, dtype, device):
    n, h, mode = (case.rows, case.width, case.variant)
    gen = torch.Generator().manual_seed(4234321)

    def rand(*shape, as_type=dtype):
        return torch.randn(shape, generator=gen).to(device, as_type)

    aux = rand(2, 1) if mode == "broadcast" else rand()
    scale = (
        rand(1, 3, as_type=torch.float32)
        if mode == "broadcast"
        else torch.tensor(1.0 if mode == "default" else 2.5, device=device)
    )
    return (rand(n, h), aux, scale)


def _review_inputs(case, device):

    def tensor(values, dtype=torch.float32):
        return torch.tensor(values, device=device, dtype=dtype)

    if case == "scale-rank":
        return (tensor([0]), tensor(0, torch.float16), tensor([2]))
    return (tensor([0]), tensor([0, 0], torch.float16), tensor(2))


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


@pytest.mark.moe_aux_loss_auto_scaler_backward
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("dtype", DTYPES)
def test_moe_aux_loss_auto_scaler_backward(case, dtype):
    _compare(
        _observe(
            flag_train.megatron.moe_aux_loss_auto_scaler_backward,
            make_inputs(case, dtype, flag_train.device),
        ),
        _observe(reference, make_inputs(case, dtype, flag_train.device)),
    )


@pytest.mark.moe_aux_loss_auto_scaler_backward
@pytest.mark.parametrize("case", ("scale-rank", "vector-promotion"))
def test_moe_aux_loss_auto_scaler_backward_review_regressions(case):
    _compare(
        _observe(
            flag_train.megatron.moe_aux_loss_auto_scaler_backward,
            _review_inputs(case, flag_train.device),
        ),
        _observe(reference, _review_inputs(case, flag_train.device)),
    )


@pytest.mark.moe_aux_loss_auto_scaler_backward
def test_aux_scale_changes_between_calls():
    fn = flag_train.megatron.moe_aux_loss_auto_scaler_backward
    grad, aux, _ = _review_inputs("vector-promotion", flag_train.device)
    for scale in (2.5, 1.0, 3.0, 1.0):
        _compare(fn(grad, aux, scale), reference(grad, aux, scale))


@pytest.mark.moe_aux_loss_auto_scaler_backward
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moe_aux_loss_unset_scale(dtype):
    grad = torch.ones(2, device=flag_train.device, dtype=dtype)
    aux = torch.zeros((), device=grad.device, dtype=dtype)
    _compare(
        flag_train.megatron.moe_aux_loss_auto_scaler_backward(grad, aux),
        reference(grad, aux),
    )
