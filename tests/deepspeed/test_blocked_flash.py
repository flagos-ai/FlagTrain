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
"""Functional tests for the DeepSpeed-compatible ``blocked_flash`` operator.

The public operator consumes DeepSpeed's pointer-layout atoms. Three oracles
are used, because they fail differently:

* ``blocked_flash_ref`` -- plain torch over the same paged cache; runs on any
  device and is the oracle for every case.
* ``flash_attn_varlen_func`` -- an independent dense FA2 implementation over the
  gathered KV, checked whenever ``flash_attn`` is installed.
* DeepSpeed's ``BlockedFlashAttn`` -- the kernel this port replaces, checked on
  the same paged cache and atoms when the compiled op is available.

The DeepSpeed oracle is CUDA-only and is skipped on other backends; the torch
reference and the operator's own output checks run on every backend.
"""

import importlib
import warnings

import pytest
import torch

import flag_train
from flag_train.deepspeed import blocked_flash
from flag_train.deepspeed.blocked_flash import get_kv_block_size, get_q_block_size

from .. import accuracy_utils as utils
from ..conftest import TO_CPU

# The external reference's atol. The local testing helper supplies the
# dtype-wise rtol and the CPU fallback.
_ATOL = {
    torch.float16: 2e-3,
    torch.bfloat16: 3.2e-2,
}

_HEAD_SIZE = 64


def _assert_close(res, ref, dtype, atol):
    res = utils.to_reference(res)
    ref = utils.to_reference(ref)
    utils.train_assert_close(res, ref, dtype, atol=atol)


def _assert_equal(res, ref):
    utils.train_assert_equal(utils.to_reference(res), utils.to_reference(ref))


# ---------------------------------------------------------------------------
# DeepSpeed oracle
# ---------------------------------------------------------------------------


def _load_deepspeed_blocked_flash():
    if flag_train.vendor_name != "nvidia":
        return None

    try:
        from deepspeed.inference.v2.inference_utils import DtypeEnum
        from deepspeed.inference.v2.kernels.ragged_ops import BlockedFlashAttn
        from deepspeed.ops.op_builder import RaggedOpsBuilder

        RaggedOpsBuilder().load()
        return BlockedFlashAttn, DtypeEnum
    except Exception as exc:
        warnings.warn(
            "DeepSpeed's BlockedFlashAttn could not be loaded, so this run "
            f"skips the differential check: {exc!r}",
            stacklevel=2,
        )
        return None


_deepspeed_blocked_flash = _load_deepspeed_blocked_flash()
_deepspeed_kernels = {}


def _deepspeed_kernel(head_size, dtype):
    key = (head_size, dtype)
    if key not in _deepspeed_kernels:
        BlockedFlashAttn, DtypeEnum = _deepspeed_blocked_flash
        _deepspeed_kernels[key] = BlockedFlashAttn(head_size, DtypeEnum(dtype))
    return _deepspeed_kernels[key]


try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False


# ---------------------------------------------------------------------------
# Atom construction
#
# The operator needs upstream's pointer layout: slots [0:2] are the 64-bit
# address of the sequence's flat block-id list. The offset atom row is kept as
# the torch reference's input, since the reference can read the logical block
# offsets directly.
# ---------------------------------------------------------------------------


def build_atoms(seq_params, q_block_size, kv_block_size, device, block_ids=None):
    """Atoms for ``(q_len, history_len)`` sequences, in both layouts.

    Returns ``(offset_atoms, pointer_atoms, block_ids)``. ``block_ids`` maps a
    logical block index to the physical block to read; pass a permuted mapping
    to exercise the indirection.
    """
    block_offsets = []
    total_blocks = 0
    for q_len, history in seq_params:
        n_blocks = (q_len + history + kv_block_size - 1) // kv_block_size
        block_offsets.append(total_blocks)
        total_blocks += n_blocks

    if block_ids is None:
        block_ids = torch.arange(total_blocks, dtype=torch.int32, device=device)

    rows = []
    q_start = 0
    for seq, (q_len, history) in enumerate(seq_params):
        n_atoms = (q_len + q_block_size - 1) // q_block_size
        for atom in range(n_atoms):
            atom_q_len = min(q_block_size, q_len - atom * q_block_size)
            local_q_start = atom * q_block_size
            total_extent = history + local_q_start + atom_q_len
            kv_blocks = (total_extent + kv_block_size - 1) // kv_block_size
            rows.append(
                [
                    block_offsets[seq],
                    0,
                    q_start + local_q_start,
                    atom_q_len,
                    kv_blocks,
                    total_extent,
                    history + local_q_start,
                    0,
                ]
            )
        q_start += q_len

    offset_atoms = torch.tensor(rows, dtype=torch.int32, device=device)
    pointer_atoms = offset_atoms.clone()
    addresses = torch.tensor(
        [block_ids.data_ptr() + 4 * int(row[0]) for row in rows],
        dtype=torch.int64,
        device=device,
    )
    pointer_atoms[:, 0:2] = addresses.view(torch.int32).reshape(-1, 2)
    return offset_atoms, pointer_atoms, block_ids


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def blocked_flash_ref(out, q, k, v, attention_atoms, kv_block_idx, softmax_scale):
    """Plain-torch reference over the same atoms and paged cache."""
    head_size = k.size(-1)
    n_heads_q = q.size(-1) // head_size
    n_heads_kv = k.size(-2)
    group = n_heads_q // n_heads_kv
    device = q.device

    for atom in attention_atoms.tolist():
        block_offset, _, q_start, q_len, kv_blocks, total_extent, global_q_idx, _ = atom
        if q_len == 0 or kv_blocks == 0:
            continue

        blocks = kv_block_idx[block_offset : block_offset + kv_blocks].long()
        k_atom = k[blocks].reshape(-1, n_heads_kv, head_size)[:total_extent]
        v_atom = v[blocks].reshape(-1, n_heads_kv, head_size)[:total_extent]
        k_atom = k_atom.repeat_interleave(group, dim=1).transpose(0, 1).float()
        v_atom = v_atom.repeat_interleave(group, dim=1).transpose(0, 1).float()

        q_atom = q[q_start : q_start + q_len].reshape(q_len, n_heads_q, head_size)
        q_atom = q_atom.transpose(0, 1).float()

        scores = torch.matmul(q_atom, k_atom.transpose(-1, -2)) * softmax_scale
        q_pos = torch.arange(q_len, device=device) + global_q_idx
        k_pos = torch.arange(total_extent, device=device)
        scores = scores.masked_fill(
            k_pos[None, None, :] > q_pos[None, :, None], float("-inf")
        )

        probs = torch.softmax(scores, dim=-1)
        atom_out = torch.matmul(probs, v_atom).transpose(0, 1)
        out[q_start : q_start + q_len] = atom_out.reshape(
            q_len, n_heads_q * head_size
        ).to(out.dtype)

    return out


def _dense_reference(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_kv,
    seq_params,
    head_size,
    n_heads_q,
    n_heads_kv,
    softmax_scale,
):
    return flash_attn_varlen_func(
        q.reshape(-1, n_heads_q, head_size),
        k.reshape(-1, n_heads_kv, head_size),
        v.reshape(-1, n_heads_kv, head_size),
        cu_seqlens_q,
        cu_seqlens_kv,
        max(s[0] for s in seq_params),
        max(s[0] + s[1] for s in seq_params),
        softmax_scale=softmax_scale,
        causal=True,
    ).reshape(-1, n_heads_q * head_size)


# ---------------------------------------------------------------------------
# Case driver
# ---------------------------------------------------------------------------


def _run_case(
    seq_params,
    head_size=_HEAD_SIZE,
    n_heads_q=16,
    n_heads_kv=16,
    softmax_scale=1.0,
    dtype=torch.float16,
    permute_blocks=False,
    out_pad_columns=0,
    v_row_padding=0,
):
    device = flag_train.device
    torch.manual_seed(0)
    q_block_size = get_q_block_size(head_size)
    kv_block_size = get_kv_block_size(head_size)

    total_q = sum(q_len for q_len, _ in seq_params)
    n_blocks = sum(
        (history + q_len + kv_block_size - 1) // kv_block_size
        for q_len, history in seq_params
    )

    qkv = torch.randn(
        (total_q, (n_heads_q + 2 * n_heads_kv) * head_size),
        dtype=dtype,
        device=device,
    )
    q = qkv[:, : n_heads_q * head_size]
    inflight_kv = qkv[:, n_heads_q * head_size :]

    history = [
        (
            torch.randn((h, 2 * n_heads_kv * head_size), dtype=dtype, device=device)
            if h > 0
            else None
        )
        for _, h in seq_params
    ]

    k_cache = torch.zeros(
        (n_blocks, kv_block_size, n_heads_kv, head_size), dtype=dtype, device=device
    )
    v_cache = torch.zeros_like(k_cache)

    cu_seqlens_q = [0]
    cu_seqlens_kv = [0]
    full_kvs = []
    first_block = 0
    for i, (q_len, history_len) in enumerate(seq_params):
        cur = inflight_kv[cu_seqlens_q[i] : cu_seqlens_q[i] + q_len]
        full = cur if history[i] is None else torch.cat([history[i], cur], dim=0)
        full_kvs.append(full)

        n_seq_blocks = (full.shape[0] + kv_block_size - 1) // kv_block_size
        padded = torch.zeros(
            (n_seq_blocks * kv_block_size, full.shape[1]), dtype=dtype, device=device
        )
        padded[: full.shape[0]] = full
        paged = padded.reshape(n_seq_blocks, kv_block_size, 2 * n_heads_kv * head_size)
        k_cache[first_block : first_block + n_seq_blocks] = paged[
            :, :, : n_heads_kv * head_size
        ].reshape(n_seq_blocks, kv_block_size, n_heads_kv, head_size)
        v_cache[first_block : first_block + n_seq_blocks] = paged[
            :, :, n_heads_kv * head_size :
        ].reshape(n_seq_blocks, kv_block_size, n_heads_kv, head_size)
        first_block += n_seq_blocks

        cu_seqlens_q.append(cu_seqlens_q[-1] + q_len)
        cu_seqlens_kv.append(cu_seqlens_kv[-1] + full.shape[0])

    block_ids = torch.arange(n_blocks, dtype=torch.int32, device=device)
    if permute_blocks:
        permutation = torch.randperm(n_blocks, device=device)
        k_cache = k_cache[permutation].contiguous()
        v_cache = v_cache[permutation].contiguous()
        block_ids = torch.argsort(permutation).to(torch.int32)

    if v_row_padding:
        row_stride = n_heads_kv * head_size + v_row_padding
        storage = torch.zeros(
            n_blocks * kv_block_size * row_stride, dtype=dtype, device=device
        )
        padded_v = storage.as_strided(
            v_cache.shape,
            (kv_block_size * row_stride, row_stride, head_size, 1),
        )
        padded_v.copy_(v_cache)
        v_cache = padded_v

    offset_atoms, pointer_atoms, block_ids = build_atoms(
        seq_params, q_block_size, kv_block_size, device, block_ids=block_ids
    )

    if out_pad_columns:
        out_buffer = torch.zeros(
            (total_q, n_heads_q * head_size + out_pad_columns),
            dtype=dtype,
            device=device,
        )
        out = out_buffer[:, : n_heads_q * head_size]
    else:
        out = torch.zeros((total_q, n_heads_q * head_size), dtype=dtype, device=device)

    blocked_flash(out, q, k_cache, v_cache, pointer_atoms, softmax_scale)

    if TO_CPU:
        ref = torch.zeros_like(out).cpu()
        blocked_flash_ref(
            ref,
            q.cpu(),
            k_cache.cpu(),
            v_cache.cpu(),
            offset_atoms.cpu(),
            block_ids.cpu(),
            softmax_scale,
        )
    else:
        ref = torch.zeros_like(out)
        blocked_flash_ref(
            ref, q, k_cache, v_cache, offset_atoms, block_ids, softmax_scale
        )

    atol = _ATOL[dtype]
    _assert_close(out, ref, dtype, atol)

    if _HAS_FLASH_ATTN:
        run_kvs = torch.cat(full_kvs, dim=0)
        dense = _dense_reference(
            q,
            run_kvs[:, : n_heads_kv * head_size],
            run_kvs[:, n_heads_kv * head_size :],
            torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device),
            torch.tensor(cu_seqlens_kv, dtype=torch.int32, device=device),
            seq_params,
            head_size,
            n_heads_q,
            n_heads_kv,
            softmax_scale,
        )
        _assert_close(out, dense, dtype, atol)

    if _deepspeed_blocked_flash is not None:
        reference_out = torch.zeros_like(out)
        _deepspeed_kernel(head_size, dtype)(
            reference_out, q, k_cache, v_cache, pointer_atoms, softmax_scale
        )
        _assert_close(reference_out, ref, dtype, atol)


# ---------------------------------------------------------------------------
# Core semantic matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_tokens", [1, 2, 33, 65, 128, 256, 2037])
@pytest.mark.blocked_flash
def test_single_prompt(n_tokens):
    _run_case([(n_tokens, 0)])


@pytest.mark.parametrize(
    "prompt_lengths", [(128, 128), (192, 38), (514, 713), (83, 312, 610)]
)
@pytest.mark.blocked_flash
def test_multiple_prompts(prompt_lengths):
    _run_case([(length, 0) for length in prompt_lengths])


@pytest.mark.parametrize(
    "seq_params", [(1, 34), (43, 40), (1, 144), (64, 128), (332, 628)]
)
@pytest.mark.blocked_flash
def test_continuation(seq_params):
    _run_case([seq_params], n_heads_q=32, n_heads_kv=32)


@pytest.mark.parametrize("head_size", [64, 128, 256], ids=["64", "128", "256"])
@pytest.mark.blocked_flash
def test_head_size(head_size):
    _run_case([(128, 128), (192, 38), (1, 814)], head_size=head_size)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.blocked_flash
def test_dtype(dtype):
    _run_case([(128, 128), (192, 38), (1, 814)], dtype=dtype)


@pytest.mark.parametrize("head_config", [(32, 8), (64, 16), (40, 8), (16, 1)])
@pytest.mark.blocked_flash
def test_gqa(head_config):
    n_heads_q, n_heads_kv = head_config
    _run_case(
        [(128, 128), (192, 38), (1, 814)],
        head_size=128,
        n_heads_q=n_heads_q,
        n_heads_kv=n_heads_kv,
    )


@pytest.mark.blocked_flash
def test_fully_composed():
    _run_case([(332, 628), (1, 718), (1, 323), (180, 5), (224, 0)])


# ---------------------------------------------------------------------------
# Paged-cache / call-state scenarios
# ---------------------------------------------------------------------------


@pytest.mark.blocked_flash
def test_kv_blocks_need_not_be_contiguous():
    _run_case([(300, 0), (1, 128), (128, 256)], permute_blocks=True)


@pytest.mark.blocked_flash
def test_out_may_be_a_view_with_a_wider_row_stride():
    _run_case([(300, 0), (1, 128)], out_pad_columns=256)


@pytest.mark.blocked_flash
def test_value_row_stride_is_honored():
    _run_case([(257, 31), (128, 128)], v_row_padding=64)


@pytest.mark.parametrize("scale", [1.0, 0.125], ids=["unit", "1/sqrt(d)"])
@pytest.mark.blocked_flash
def test_softmax_scale_is_applied(scale):
    _run_case([(128, 128), (192, 38)], softmax_scale=scale)


@pytest.mark.blocked_flash
def test_many_uneven_sequences():
    lengths = [1, 2, 63, 64, 65, 127, 128, 129, 200, 257, 300, 511, 512, 700, 1000, 3]
    _run_case([(length, 0) for length in lengths])


def _single_inputs(n_tokens=128, head_size=_HEAD_SIZE, dtype=torch.float16):
    q_block_size = get_q_block_size(head_size)
    kv_block_size = get_kv_block_size(head_size)
    n_blocks = (n_tokens + kv_block_size - 1) // kv_block_size
    torch.manual_seed(0)

    q = torch.randn((n_tokens, 16 * head_size), dtype=dtype, device=flag_train.device)
    kv = torch.randn((n_tokens, 32 * head_size), dtype=dtype, device=flag_train.device)
    padded = torch.zeros(
        (n_blocks * kv_block_size, 32 * head_size),
        dtype=dtype,
        device=flag_train.device,
    )
    padded[:n_tokens] = kv
    paged = padded.reshape(n_blocks, kv_block_size, 32 * head_size)
    k = (
        paged[:, :, : 16 * head_size]
        .reshape(n_blocks, kv_block_size, 16, head_size)
        .contiguous()
    )
    v = (
        paged[:, :, 16 * head_size :]
        .reshape(n_blocks, kv_block_size, 16, head_size)
        .contiguous()
    )
    offset_atoms, pointer_atoms, block_ids = build_atoms(
        [(n_tokens, 0)], q_block_size, kv_block_size, flag_train.device
    )
    return q, k, v, offset_atoms, pointer_atoms, block_ids


@pytest.mark.blocked_flash
def test_output_is_overwritten_not_accumulated():
    q, k, v, _, atoms, _ = _single_inputs()
    out = torch.zeros_like(q)

    blocked_flash(out, q, k, v, atoms, 1.0)
    first = out.clone()
    assert first.abs().sum() > 0

    blocked_flash(out, q, k, v, atoms, 1.0)
    _assert_equal(out, first)


@pytest.mark.blocked_flash
def test_atom_order_does_not_matter():
    q, k, v, _, atoms, _ = _single_inputs()
    out = torch.zeros_like(q)
    blocked_flash(out, q, k, v, atoms, 1.0)
    reference = out.clone()

    order = torch.randperm(atoms.size(0), device=atoms.device)
    shuffled = torch.zeros_like(q)
    blocked_flash(shuffled, q, k, v, atoms[order], 1.0)
    _assert_equal(shuffled, reference)


@pytest.mark.blocked_flash
def test_large_logits_stay_finite():
    q, k, v, _, atoms, _ = _single_inputs(n_tokens=256)
    q = q * 30.0
    k = k * 30.0
    out = torch.zeros_like(q)

    blocked_flash(out, q, k, v, atoms, 1.0)
    assert torch.isfinite(out).all()
    assert out.abs().max() <= v.abs().max() + 1e-3


# ---------------------------------------------------------------------------
# Pointer-source and interface checks
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    flag_train.vendor_name != "nvidia",
    reason="pinned host atom pointers are a CUDA/UVA path",
)
@pytest.mark.blocked_flash
def test_pinned_host_block_list():
    head_size = 128  # two KV blocks, so the pinned list is walked past its first entry
    q, k, v, _, device_atoms, block_ids = _single_inputs(head_size=head_size)
    host_ids = block_ids.cpu().pin_memory()
    q_block_size = get_q_block_size(head_size)
    kv_block_size = get_kv_block_size(head_size)
    _, pinned_atoms, _ = build_atoms(
        [(q.size(0), 0)],
        q_block_size,
        kv_block_size,
        flag_train.device,
        block_ids=host_ids,
    )

    device_out = torch.zeros_like(q)
    blocked_flash(device_out, q, k, v, device_atoms, 1.0)
    pinned_out = torch.zeros_like(q)
    blocked_flash(pinned_out, q, k, v, pinned_atoms, 1.0)
    _assert_equal(device_out, pinned_out)


@pytest.mark.blocked_flash
def test_infers_head_size_and_dtype():
    q, k, v, _, atoms, _ = _single_inputs(head_size=128)
    inferred = torch.zeros_like(q)
    blocked_flash(inferred, q, k, v, atoms, 1.0)

    explicit = torch.zeros_like(q)
    blocked_flash(
        explicit,
        q,
        k,
        v,
        atoms,
        1.0,
        head_size=128,
        dtype=torch.float16,
    )
    _assert_equal(inferred, explicit)


@pytest.mark.skipif(
    flag_train.vendor_name != "nvidia",
    reason="DeepSpeed's geometry table is CUDA-specific",
)
@pytest.mark.blocked_flash
def test_geometry_matches_deepspeed():
    module = pytest.importorskip(
        "deepspeed.inference.v2.kernels.ragged_ops.blocked_flash.blocked_flash"
    )
    for head_size in range(8, 257, 8):
        assert get_q_block_size(head_size) == module.get_q_block_size(head_size)
        assert get_kv_block_size(head_size) == module.get_kv_block_size(head_size)


@pytest.mark.parametrize(
    "head_size, tensor_dtype, kwargs, message",
    [
        (24, torch.float16, {}, "divisible by 16"),
        (8, torch.float16, {}, "divisible by 16"),
        (64, torch.float32, {"dtype": torch.float32}, "Unsupported data type"),
        (64, torch.float16, {"head_size": 128}, "does not match"),
        (64, torch.float16, {"dtype": torch.bfloat16}, "does not match"),
    ],
)
@pytest.mark.blocked_flash
def test_argument_validation(head_size, tensor_dtype, kwargs, message):
    q, k, v, _, atoms, _ = _single_inputs(head_size=head_size, dtype=tensor_dtype)
    with pytest.raises(ValueError, match=message):
        blocked_flash(torch.zeros_like(q), q, k, v, atoms, 1.0, **kwargs)


@pytest.mark.blocked_flash
def test_rejects_cpu_tensors():
    q, k, v, _, atoms, _ = _single_inputs()
    with pytest.raises((RuntimeError, ValueError), match="device"):
        blocked_flash(
            torch.zeros_like(q).cpu(),
            q.cpu(),
            k.cpu(),
            v.cpu(),
            atoms.cpu(),
            1.0,
        )


@pytest.mark.blocked_flash
def test_only_blocked_flash_is_public():
    from flag_train import deepspeed

    module = importlib.import_module("flag_train.deepspeed.blocked_flash")
    assert deepspeed.blocked_flash is blocked_flash
    assert "blocked_flash" in deepspeed.__all__
    assert module.__all__ == ["blocked_flash"]
    for name in ("flash_attn_by_atoms", "get_q_block_size", "get_kv_block_size"):
        assert name not in deepspeed.__all__
