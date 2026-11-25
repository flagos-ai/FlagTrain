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


"""Test the dsa_indexer_loss_auto_scaler operator against its Torch reference."""

from dataclasses import dataclass

import pytest
import torch

import flag_train
from flag_train.reference_candidate import get_candidate
from flag_train.testing import assert_close, assert_equal


class _DSAScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, indexer_loss, loss_scale):
        ctx.save_for_backward(indexer_loss)
        # Saving this buffer with save_for_backward would prohibit a later update.
        ctx.scale = loss_scale.detach()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (loss,) = ctx.saved_tensors
        return grad_output, torch.ones_like(loss) * ctx.scale.to(loss.device), None


def reference(output, indexer_loss, loss_scale):
    return _DSAScaler.apply(output, indexer_loss, loss_scale)


OP_NAME = "dsa_indexer_loss_auto_scaler"
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
    Case(1, 32, "default"),
    Case(32, 128, "scale"),
    Case(128, 128, "default"),
    Case(256, 256, "scale"),
    Case(512, 512, "default"),
    Case(1024, 1024, "scale"),
    Case(1, 1, "scale", True),
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

    shape = (1025,) if case == "multi-block" else (2, 3)
    if case == "column-scale":
        scale = tensor([[2], [3]])
    elif case == "cpu-scale":
        scale = torch.tensor(2.0, device="cpu")
    else:
        scale = tensor(2.0)
    return (tensor([1, 2]), torch.ones(shape, device=device), scale)


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
    args[0].requires_grad_()
    args[1].requires_grad_()
    args[2].requires_grad_()
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
    grad = (
        torch.linspace(-1, 2, args[0].numel(), device=args[0].device)
        .reshape_as(args[0])
        .to(args[0])
    )
    grad_output, grad_loss, grad_scale = torch.autograd.grad(
        outputs[0], args, grad, allow_unused=True
    )
    assert grad_scale is None
    assert_equal(grad_output, grad)
    for arg, snapshot, layout in zip(args, originals, layouts):
        assert _layout(arg) == layout
        assert_equal(arg, snapshot)
    outputs += (grad_output, grad_loss)
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


@pytest.mark.dsa_indexer_loss_auto_scaler
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("dtype", DTYPES)
def test_dsa_indexer_loss_auto_scaler(case, dtype):
    _compare(
        _observe(_candidate(), make_inputs(case, dtype, flag_train.device)),
        _observe(reference, make_inputs(case, dtype, flag_train.device)),
    )


@pytest.mark.dsa_indexer_loss_auto_scaler
@pytest.mark.parametrize("case", ("multi-block", "column-scale", "cpu-scale"))
def test_dsa_indexer_loss_auto_scaler_review_regressions(case):
    _compare(
        _observe(_candidate(), _review_inputs(case, flag_train.device)),
        _observe(reference, _review_inputs(case, flag_train.device)),
    )


@pytest.mark.dsa_indexer_loss_auto_scaler
def test_dsa_retained_gradients():
    for fn in (reference, _candidate()):
        x = torch.ones(2, device=flag_train.device, requires_grad=True)
        loss = torch.ones((), device=x.device, requires_grad=True)
        scale = torch.tensor(2.0, device=x.device)
        out = fn(x, loss, scale)
        (first,) = torch.autograd.grad(out.sum(), loss, retain_graph=True)
        scale.fill_(3.0)
        (second,) = torch.autograd.grad(out.sum(), loss)
        assert_close(first, loss.new_tensor(2), loss.dtype)
        assert_close(second, loss.new_tensor(3), loss.dtype)
        assert first.data_ptr() != second.data_ptr()


def check_live_scales(order):
    for fn in (_candidate(), reference):
        scale = torch.tensor(2.0, device=flag_train.device, requires_grad=True)
        graphs = []
        for i in (1, 2):
            out = torch.full((2, 3), float(i), device=scale.device, requires_grad=True)
            loss = torch.full((), float(i), device=scale.device, requires_grad=True)
            result = fn(out, loss, scale)
            _assert_alias(result, out, True)
            assert_equal(result, out)
            graphs.append((out, loss, result))
        for index, current_scale in zip(order, (3.25, 7.5)):
            with torch.no_grad():
                scale.fill_(current_scale)
            out, loss, result = graphs[index]
            grad = torch.full_like(out, float(index + 2))
            dx, dl, ds = torch.autograd.grad(
                result, (out, loss, scale), grad, allow_unused=True
            )
            assert_equal(dx, grad)
            assert_equal(dl, loss.new_tensor(current_scale))
            assert ds is None
            assert_close(out, torch.full_like(out, float(index + 1)), out.dtype)
            assert_close(loss, loss.new_tensor(float(index + 1)), loss.dtype)
            assert_close(scale, scale.new_tensor(current_scale), scale.dtype)


@pytest.mark.dsa_indexer_loss_auto_scaler
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_live_scales(order):
    check_live_scales(order)
