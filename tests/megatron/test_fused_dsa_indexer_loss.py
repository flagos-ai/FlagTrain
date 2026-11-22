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


"""Test the fused_dsa_indexer_loss operator against its Torch reference."""

import math
from dataclasses import dataclass

import pytest
import torch

import flag_train
from flag_train.reference_candidate import get_candidate


def is_ppu(tensor):
    if tensor.device.type != "cuda":
        return False
    from flag_train import vendor_name

    return vendor_name == "thead"


def _ppu_teacher_scores(query, key):
    # PPU aborts in the batched contraction used by this einsum. Each batch
    # and teacher head is an independent matrix product.
    sq, batch, heads, hidden = query.shape
    sk = key.size(0)
    queries = query.permute(1, 2, 0, 3).contiguous().reshape(batch * heads, sq, hidden)
    keys = key.permute(1, 2, 0, 3).contiguous().reshape(batch * heads, sk, hidden)
    scores = [torch.mm(q, k.T) for q, k in zip(queries.unbind(), keys.unbind())]
    return torch.stack(scores).reshape(batch, heads, sq, sk)


def reference(
    q,
    weights,
    k,
    query,
    key,
    softmax_scale,
    topk,
    loss_coeff,
    mask=None,
    sparse_loss=False,
    varlen_starts=None,
    varlen_ends=None,
    key_positions=None,
    query_valid_rows=None,
    calculate_per_token_loss=False,
    use_relu=True,
    non_compressed_lse=None,
):
    sq, batch = q.shape[:2]
    sk = k.size(0)
    loss_mask = mask
    if varlen_starts is not None:
        if mask is not None:
            raise ValueError("mask and varlen_starts are mutually exclusive")
        if varlen_ends is None:
            raise ValueError("varlen_ends is required when varlen_starts is provided")
        varlen_starts = varlen_starts.to(device=q.device, dtype=torch.int64)
        varlen_ends = varlen_ends.to(device=q.device, dtype=torch.int64)
        if key_positions is not None:
            key_positions = key_positions.to(device=q.device, dtype=torch.int64)
    elif mask is None:
        loss_mask = torch.full(
            (sq, sk), -torch.inf, dtype=torch.float32, device=q.device
        ).triu(diagonal=1)
    else:
        assert mask.dtype == torch.float32, "mask dtype must be float32"
        assert mask.device == q.device, "mask device mismatch"
        assert mask.shape in ((sq, sk), (batch, sq, sk)), "mask shape mismatch"
    index = torch.einsum("sbhd,tbd->bsht", q.float(), k.float())
    if use_relu:
        index = index.relu()
    index = (index * weights.transpose(0, 1).float().unsqueeze(-1)).sum(2)
    if query.ndim == 3:
        query, key = query.unsqueeze(1), key.unsqueeze(1)
    query, key = query.detach().float(), key.detach().float()
    if is_ppu(query):
        teacher = _ppu_teacher_scores(query, key)
    else:
        teacher = torch.einsum("sbhd,tbhd->bhst", query, key)
    teacher = teacher * softmax_scale
    if varlen_starts is not None:
        if key_positions is None:
            key_positions = torch.arange(k.size(0), device=k.device)
        valid = (
            (key_positions >= varlen_starts[:, None])
            & (key_positions < varlen_ends[:, None])
        ).expand_as(index)
        index = index.masked_fill(~valid, -torch.inf)
    else:
        teacher = teacher + loss_mask.unsqueeze(-3)
        if mask is not None:
            index = index + mask
        valid = torch.isfinite(loss_mask).expand_as(index)
    top_scores, ids = index.topk(min(topk, k.size(0)))
    ids = ids.masked_fill(top_scores == -torch.inf, -1)
    if loss_mask is not None:
        # The native loss always applies a mask. Only an explicit additive
        # mask was already applied by the top-k path.
        index = index + loss_mask
    if sparse_loss:
        selected = torch.zeros_like(index, dtype=torch.int32)
        selected.scatter_add_(-1, ids.clamp_min(0), (ids >= 0).int())
        valid = valid & (selected > 0)
    live = valid.any(-1, keepdim=True)
    safe = lambda x, keep, row: x.masked_fill(~keep, -torch.inf).masked_fill(~row, 0)
    index_logp = safe(index, valid, live).log_softmax(-1).masked_fill(~valid, 0)
    teacher = safe(teacher, valid[:, None], live[:, None])
    p = teacher.softmax(-1).masked_fill(~valid[:, None], 0)
    if non_compressed_lse is not None:
        compressed_lse = teacher.logsumexp(-1)
        log_mass = (
            compressed_lse - torch.logaddexp(compressed_lse, non_compressed_lse)
        ).masked_fill(~live.squeeze(-1)[:, None], -torch.inf)
        row_max = log_mass.amax(dim=1, keepdim=True)
        shift = torch.where(
            torch.isfinite(row_max)
            & (row_max < math.log(torch.finfo(torch.float32).tiny)),
            row_max,
            0,
        )
        p = p * (log_mass - shift).exp().unsqueeze(-1)
    p = p.sum(dim=1)
    mass = p.sum(-1, keepdim=True)
    p = p / (
        mass.clamp_min(1e-10)
        if non_compressed_lse is None
        else torch.where(mass > 0, mass, 1)
    )
    rows = (p * (p.clamp_min(1e-10).log() - index_logp)).sum(-1)
    if query_valid_rows is not None:
        query_valid_rows = query_valid_rows.to(
            device=rows.device, dtype=torch.bool
        ).expand_as(rows)
        rows = rows * query_valid_rows
    denominator = (
        1
        if calculate_per_token_loss
        else (
            rows.numel()
            if query_valid_rows is None
            else query_valid_rows.sum().clamp_min(1)
        )
    )
    return ids, rows.sum() * loss_coeff / denominator


OP_NAME = "fused_dsa_indexer_loss"
pytestmark = pytest.mark.skipif(
    flag_train.vendor_name != "nvidia", reason="NVIDIA kernel validation"
)
DTYPES = (torch.float32, torch.bfloat16)


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
    Case(19, 16, "dense"),
    Case(128, 32, "sparse"),
    Case(17, 16, "packed", True),
    Case(17, 16, "external-lse", True),
    Case(17, 16, "all-masked", True),
    Case(17, 16, "sparse"),
    Case(33, 16, "dense"),
    Case(64, 16, "sparse"),
    Case(96, 16, "dense"),
    Case(128, 16, "sparse"),
    Case(32, 32, "dense"),
    Case(64, 32, "sparse"),
    Case(96, 32, "dense"),
    Case(192, 32, "sparse"),
)


def make_inputs(case, dtype, device):
    n, h, mode = (case.rows, case.width, case.variant)
    gen = torch.Generator().manual_seed(4234321)

    def rand(*shape, as_type=dtype):
        return torch.randn(shape, generator=gen).to(device, as_type)

    packed = mode in ("packed", "external-lse")
    b, nq, nk = (1 if packed else 2, 2, 3)
    q, w, k = (rand(n, b, nq, h), rand(n, b, nq), rand(n + 5, b, h))
    t_q, t_k = (rand(n, b, nk, h), rand(n + 5, b, nk, h))
    starts = torch.arange(n, device=device) // 5 * 5
    ends = torch.arange(n, device=device) + 2
    keys = torch.arange(n + 5, device=device)
    mask = torch.zeros(n, n + 5, device=device).masked_fill(
        keys >= ends[:, None], -torch.inf
    )
    if mode == "all-masked":
        mask[0] = -torch.inf
    return (
        q,
        w,
        k,
        t_q,
        t_k,
        h ** (-0.5),
        n + 10 if mode == "all-masked" else 4,
        0.7,
        None if packed else mask,
        mode != "dense",
        starts if packed else None,
        ends if packed else None,
        keys if packed else None,
        (
            torch.arange(n, device=device) != 0
            if mode in ("all-masked", "external-lse")
            else None
        ),
        mode == "external-lse",
        mode != "packed",
        rand(b, nk, n, as_type=torch.float32) if mode == "external-lse" else None,
    )


def _review_inputs(case, device):

    def tensor(values, dtype=torch.float32):
        return torch.tensor(values, device=device, dtype=dtype)

    q = torch.zeros((2, 1, 1, 16), device=device)
    q[..., 0] = 1
    keys = torch.zeros((2, 1, 16), device=device)
    keys[:, 0, 0] = tensor([0, 0.02])
    teacher_key = keys.unsqueeze(2).clone()
    teacher_key[1, 0, 0, 0] = 2.0
    use_lse = case == "underflow-lse"
    mask = torch.zeros((2, 2), device=device) if use_lse else None
    lse = torch.full((1, 1, 2), 1000.0, device=device) if use_lse else None
    return (
        q,
        q[..., 0].clone(),
        keys,
        q.clone(),
        teacher_key,
        1.0,
        2,
        1.0,
        mask,
        False,
        None,
        None,
        None,
        None,
        False,
        False,
        lse,
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


@pytest.mark.fused_dsa_indexer_loss
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_dsa_indexer_loss(case, dtype):
    _compare(
        _observe(_candidate(), make_inputs(case, dtype, flag_train.device)),
        _observe(reference, make_inputs(case, dtype, flag_train.device)),
    )


@pytest.mark.fused_dsa_indexer_loss
@pytest.mark.parametrize("case", ("implicit-causal", "underflow-lse"))
def test_fused_dsa_indexer_loss_review_regressions(case):
    _compare(
        _observe(_candidate(), _review_inputs(case, flag_train.device)),
        _observe(reference, _review_inputs(case, flag_train.device)),
    )


@pytest.mark.fused_dsa_indexer_loss
def test_fused_dsa_indexer_loss_on_noncurrent_cuda_device():
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    with torch.cuda.device(0):
        got = _observe(_candidate(), _review_inputs("implicit-causal", "cuda:1"))
        expected = _observe(reference, _review_inputs("implicit-causal", "cuda:1"))
        assert torch.cuda.current_device() == 0
    _compare(got, expected)


@pytest.mark.fused_dsa_indexer_loss
def test_fused_dsa_indexer_loss_packed_cpu_metadata():
    args = list(make_inputs(Case(17, 16, "packed"), torch.float32, flag_train.device))
    for index in (10, 11, 12):
        args[index] = args[index].cpu()
    _compare(_observe(_candidate(), args), _observe(reference, args))


@pytest.mark.fused_dsa_indexer_loss
def test_fused_dsa_indexer_loss_wide_context():
    device = flag_train.device
    q = torch.ones((1, 1, 1, 16), device=device)
    k = torch.zeros((65537, 1, 16), device=device)
    k[-1, 0, 0] = 2
    indices, loss = _candidate()(
        q, q[..., 0], k, q.clone(), k.unsqueeze(2).clone(), 1.0, 1, 1.0
    )
    torch.testing.assert_close(indices, torch.tensor([[[65536]]], device=device))
    torch.testing.assert_close(loss, torch.zeros((), device=device), rtol=0, atol=0)


@pytest.mark.fused_dsa_indexer_loss
def test_fused_dsa_indexer_loss_boolean_cpu_valid_rows():
    args = list(
        make_inputs(Case(17, 16, "external-lse"), torch.float32, flag_train.device)
    )
    args[13] = torch.ones(17, dtype=torch.int32)
    args[13][:2] = torch.tensor([0, 2], dtype=torch.int32)
    _compare(_observe(_candidate(), args), _observe(reference, args))
