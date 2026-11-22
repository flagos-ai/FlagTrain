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


"""Test the fused_topk_with_score_function operator against its Torch reference."""

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

import flag_train
from flag_train.reference_candidate import get_candidate


def _router_topk(scores, k, use_radix):
    # The naive CUDA reduction keeps ties in the bit-reversed warp lane
    # order, then by position within a lane. Radix fills ties by expert ID.
    if use_radix:
        indices = scores.argsort(dim=-1, descending=True, stable=True)[..., :k]
    else:
        size = scores.size(-1)
        lane_order = [int(f"{lane:05b}"[::-1], 2) for lane in range(32)]
        order = scores.new_tensor(
            [i for lane in lane_order for i in range(lane, size, 32)],
            dtype=torch.long,
        )
        ranked = scores.index_select(-1, order).argsort(
            dim=-1, descending=True, stable=True
        )
        indices = order[ranked[..., :k]]
    return scores.gather(-1, indices), indices


def reference(
    logits,
    topk,
    use_pre_softmax,
    num_groups,
    group_topk,
    scaling_factor,
    score_function,
    expert_bias,
    routing_map_format="bytemap",
):
    leading = logits.shape[:-1]
    raw = logits.reshape(-1, logits.size(-1)).float()
    if score_function == "softmax":
        raw = raw.softmax(-1) if use_pre_softmax else raw
    elif score_function == "sigmoid":
        raw = raw.sigmoid()
    elif score_function == "sqrtsoftplus":
        raw = F.softplus(raw).sqrt()
    else:
        raise ValueError(f"Unknown score function: {score_function}")
    scores = raw
    if expert_bias is not None and score_function != "softmax":
        scores = scores + expert_bias.float()
    use_radix = topk >= 16 and logits.size(-1) <= 8160
    if num_groups and group_topk:
        groups = scores.reshape(scores.shape[0], num_groups, -1)
        best = _router_topk(groups, topk // group_topk, use_radix)[0].sum(-1)
        group_map = torch.zeros_like(best, dtype=torch.bool).scatter_(
            -1, _router_topk(best, group_topk, use_radix)[1], True
        )
        scores = groups.masked_fill(~group_map[:, :, None], -torch.inf).flatten(1)
    selected_scores, chosen = _router_topk(scores, topk, use_radix)
    selected = raw.gather(-1, chosen)
    if score_function == "sqrtsoftplus" and expert_bias is not None:
        selected = selected_scores - expert_bias[chosen].float()
    if score_function == "softmax" and not use_pre_softmax:
        selected = selected.softmax(-1)
    elif score_function != "softmax" and topk > 1:
        selected = selected / (selected.sum(-1, keepdim=True) + 1e-20)
    probs = torch.zeros_like(raw).scatter_(
        -1, chosen, selected * (1.0 if scaling_factor is None else scaling_factor)
    )
    routes = torch.zeros_like(raw, dtype=torch.bool).scatter_(-1, chosen, True)
    if getattr(routing_map_format, "value", routing_map_format) in (1, "bitmap_u8"):
        bits = 2 ** torch.arange(8, device=logits.device)
        n = routes.size(-1)
        routes = (
            (F.pad(routes, (0, (-n) % 8)).reshape(routes.size(0), -1, 8) * bits)
            .sum(-1)
            .to(torch.uint8)
        )
    return probs.reshape(logits.shape).to(logits.dtype), routes.reshape(
        *leading, routes.size(-1)
    )


OP_NAME = "fused_topk_with_score_function"
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
    Case(19, 64, "softmax"),
    Case(512, 64, "sigmoid-grouped"),
    Case(19, 64, "pre-softmax", True),
    Case(19, 64, "sqrtsoftplus", True),
    Case(19, 64, "top1", True),
    Case(12, 65, "batched-bitmap", True),
    Case(5, 64, "tie-warp", True),
    Case(5, 64, "tie-radix", True),
    Case(8, 16, "softmax"),
    Case(32, 32, "sigmoid-grouped"),
    Case(64, 128, "softmax"),
    Case(128, 32, "softmax"),
    Case(256, 128, "sigmoid-grouped"),
    Case(512, 128, "softmax"),
)


def make_inputs(case, dtype, device):
    n, h, mode = (case.rows, case.width, case.variant)
    gen = torch.Generator().manual_seed(4234321)

    def rand(*shape, as_type=dtype):
        return torch.randn(shape, generator=gen).to(device, as_type)

    grouped = mode == "sigmoid-grouped"
    function = (
        "sigmoid"
        if grouped or mode == "top1"
        else "sqrtsoftplus" if mode in ("sqrtsoftplus", "batched-bitmap") else "softmax"
    )
    logits = (
        rand(n, h, as_type=torch.float32) * 0.07 + torch.arange(h, device=device) * 0.13
    ).to(dtype)
    if mode.startswith("tie-"):
        logits.zero_()
    if mode == "batched-bitmap":
        logits = logits.reshape(3, n // 3, h)
    return (
        logits,
        1 if mode == "top1" else 16 if mode == "tie-radix" else 4,
        mode == "pre-softmax",
        4 if grouped else None,
        2 if grouped else None,
        None if mode == "pre-softmax" else 1.75,
        function,
        rand(h, as_type=torch.float32) if function != "softmax" else None,
        (
            "bitmap_u8"
            if mode in ("sigmoid-grouped", "sqrtsoftplus", "batched-bitmap")
            else "bytemap"
        ),
    )


def _review_inputs(case, device):

    def tensor(values, dtype=torch.float32):
        return torch.tensor(values, device=device, dtype=dtype)

    k, groups, group_k, score = (4, None, None, "softmax")
    if case == "group-ties":
        logits, groups, group_k, score = ([0] * 64, 4, 2, "sigmoid")
    elif case == "warp-ties-k9":
        logits, k = ([0] * 32, 9)
    elif case == "post-softmax-underflow":
        logits, k = ([1000, 0] + [-1] * 30, 2)
    elif case == "sqrtsoftplus-tail":
        logits, k, score = ([-24, -22] + [-24] * 30, 1, "sqrtsoftplus")
    elif case == "sixteen-groups":
        logits, groups, group_k = (list(range(256)), 16, 2)
    else:
        logits, k = (list(range(256)), 128)
    return (tensor([logits]), k, False, groups, group_k, None, score, None, "bytemap")


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


@pytest.mark.fused_topk_with_score_function
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_topk_with_score_function(case, dtype):
    _compare(
        _observe(_candidate(), make_inputs(case, dtype, flag_train.device)),
        _observe(reference, make_inputs(case, dtype, flag_train.device)),
    )


@pytest.mark.fused_topk_with_score_function
@pytest.mark.parametrize(
    "case",
    (
        "group-ties",
        "warp-ties-k9",
        "post-softmax-underflow",
        "sqrtsoftplus-tail",
        "sixteen-groups",
        "k128",
    ),
)
def test_fused_topk_with_score_function_review_regressions(case):
    _compare(
        _observe(_candidate(), _review_inputs(case, flag_train.device)),
        _observe(reference, _review_inputs(case, flag_train.device)),
    )


@pytest.mark.fused_topk_with_score_function
@pytest.mark.parametrize("format", [0, 1])
def test_fused_topk_numeric_routing_format(format):
    args = _review_inputs("warp-ties-k9", flag_train.device)
    _compare(
        _observe(_candidate(), (*args[:-1], format)),
        _observe(reference, (*args[:-1], format)),
    )
