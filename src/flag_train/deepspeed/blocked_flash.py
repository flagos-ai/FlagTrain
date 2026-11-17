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
"""Flash attention forward over a blocked KV-cache, driven by attention atoms.

DeepSpeed-compatible public interface: pointer-layout atoms, fixed atom
geometry, causal only. The kernel walks an atom's paged KV blocks with an
online softmax and writes the preallocated output in place."""

import logging

import torch
import triton
import triton.language as tl

from flag_train.runtime import device, torch_device_fn
from flag_train.utils import libentry

logger = logging.getLogger(__name__)

# Atom field indices, in int32 slots. See the module docstring for meanings.
_ATOM_BLOCK_OFFSET = tl.constexpr(0)
_ATOM_Q_START = tl.constexpr(2)
_ATOM_Q_LEN = tl.constexpr(3)
_ATOM_KV_BLOCKS = tl.constexpr(4)
_ATOM_TOTAL_EXTENT = tl.constexpr(5)
_ATOM_GLOBAL_Q_IDX = tl.constexpr(6)
_ATOM_STRIDE = tl.constexpr(8)

_ATOM_PTR_HIGH = tl.constexpr(1)
_ATOM_UNUSED = tl.constexpr(7)

# log2(e): the kernel's softmax runs on exp2, so this rides on the qk scale.
_LOG2E = 1.4426950408889634

_BLOCK_N_MAX = 64


@triton.jit
def _kv_tile(
    q,
    k_ptr,
    v_ptr,
    k_block_row_base,
    v_block_row_base,
    offs_n,
    kv_pos,
    total_extent,
    global_q_pos,
    qk_scale,
    m_i,
    l_i,
    acc,
    k_row_stride,
    v_row_stride,
    hk,
    offs_d,
    d_mask,
    MASKED: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TRANSPOSED: tl.constexpr = False,
):
    """One ``BLOCK_N``-wide KV tile, folded into the running softmax state."""
    n_mask = kv_pos < total_extent

    if TRANSPOSED:
        k = tl.load(
            k_ptr
            + k_block_row_base
            + hk * HEAD_SIZE
            + offs_d[:, None]
            + offs_n[None, :] * k_row_stride,
            mask=d_mask[:, None] & n_mask[None, :] if MASKED else d_mask[:, None],
            other=0.0,
        )
    elif MASKED:
        k = tl.load(
            k_ptr
            + k_block_row_base
            + offs_n[:, None] * k_row_stride
            + hk * HEAD_SIZE
            + offs_d[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
    else:
        k = tl.load(
            k_ptr
            + k_block_row_base
            + offs_n[:, None] * k_row_stride
            + hk * HEAD_SIZE
            + offs_d[None, :],
            mask=d_mask[None, :],
            other=0.0,
        )

    if TRANSPOSED:
        qk = tl.dot(q, k) * qk_scale
    else:
        qk = tl.dot(q, tl.trans(k)) * qk_scale

    if MASKED:
        if IS_CAUSAL:
            # A query attends to every KV position up to and including its own.
            qk = tl.where(kv_pos[None, :] <= global_q_pos[:, None], qk, float("-inf"))
        qk = tl.where(n_mask[None, :], qk, float("-inf"))

    m_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(qk - m_new[:, None])

    if TRANSPOSED:
        v = tl.load(
            v_ptr
            + v_block_row_base
            + hk * HEAD_SIZE
            + offs_d[:, None]
            + offs_n[None, :] * v_row_stride,
            mask=d_mask[:, None] & n_mask[None, :] if MASKED else d_mask[:, None],
            other=0.0,
        )
        v = tl.trans(v)
    elif MASKED:
        v = tl.load(
            v_ptr
            + v_block_row_base
            + offs_n[:, None] * v_row_stride
            + hk * HEAD_SIZE
            + offs_d[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
    else:
        v = tl.load(
            v_ptr
            + v_block_row_base
            + offs_n[:, None] * v_row_stride
            + hk * HEAD_SIZE
            + offs_d[None, :],
            mask=d_mask[None, :],
            other=0.0,
        )

    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

    return m_new, l_i, acc


@libentry()
@triton.jit
def _blocked_flash_fwd_kernel(
    out_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    atoms_ptr,
    qk_scale,
    q_row_stride,
    o_row_stride,
    k_row_stride,
    v_row_stride,
    h_h_k_ratio,
    KV_BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    atom = tl.program_id(0)
    hq = tl.program_id(1)

    # --- atom metadata -----------------------------------------------------
    atom_base = atoms_ptr + atom * _ATOM_STRIDE
    # Slots [0:2] are the 64-bit address of the sequence's block-id list.
    # Assemble it from two int32 loads: the tensor only guarantees int32
    # alignment, so an int64 load would fault on a 4-byte-offset view.
    ptr_lo = tl.load(atom_base).to(tl.uint32).to(tl.uint64)
    ptr_hi = tl.load(atom_base + 1).to(tl.uint32).to(tl.uint64)
    block_ptr = (ptr_lo | (ptr_hi << 32)).to(tl.pointer_type(tl.int32))
    q_start = tl.load(atom_base + _ATOM_Q_START)
    q_len = tl.load(atom_base + _ATOM_Q_LEN)
    kv_blocks = tl.load(atom_base + _ATOM_KV_BLOCKS)
    total_extent = tl.load(atom_base + _ATOM_TOTAL_EXTENT)
    global_q_idx = tl.load(atom_base + _ATOM_GLOBAL_Q_IDX)

    hk = hq // h_h_k_ratio

    # --- queries -----------------------------------------------------------
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < q_len
    d_mask = offs_d < HEAD_SIZE

    q_rows = q_start + offs_m
    q_ptrs = q_ptr + q_rows[:, None] * q_row_stride + hq * HEAD_SIZE + offs_d[None, :]
    q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    global_q_pos = global_q_idx + offs_m

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    peeled = tl.minimum(
        total_extent // KV_BLOCK_SIZE, (global_q_idx + 1) // KV_BLOCK_SIZE
    )

    for b in range(0, peeled):
        block = tl.load(block_ptr + b)
        k_block_row_base = block * k_row_stride * KV_BLOCK_SIZE
        v_block_row_base = block * v_row_stride * KV_BLOCK_SIZE
        for t0 in range(0, KV_BLOCK_SIZE, BLOCK_N):
            offs_n = t0 + tl.arange(0, BLOCK_N)
            kv_pos = b * KV_BLOCK_SIZE + offs_n
            m_i, l_i, acc = _kv_tile(
                q,
                k_ptr,
                v_ptr,
                k_block_row_base,
                v_block_row_base,
                offs_n,
                kv_pos,
                total_extent,
                global_q_pos,
                qk_scale,
                m_i,
                l_i,
                acc,
                k_row_stride,
                v_row_stride,
                hk,
                offs_d,
                d_mask,
                MASKED=False,
                IS_CAUSAL=True,
                HEAD_SIZE=HEAD_SIZE,
                BLOCK_N=BLOCK_N,
            )

    for b in range(peeled, kv_blocks):
        block = tl.load(block_ptr + b)
        k_block_row_base = block * k_row_stride * KV_BLOCK_SIZE
        v_block_row_base = block * v_row_stride * KV_BLOCK_SIZE
        for t0 in range(0, KV_BLOCK_SIZE, BLOCK_N):
            offs_n = t0 + tl.arange(0, BLOCK_N)
            kv_pos = b * KV_BLOCK_SIZE + offs_n
            m_i, l_i, acc = _kv_tile(
                q,
                k_ptr,
                v_ptr,
                k_block_row_base,
                v_block_row_base,
                offs_n,
                kv_pos,
                total_extent,
                global_q_pos,
                qk_scale,
                m_i,
                l_i,
                acc,
                k_row_stride,
                v_row_stride,
                hk,
                offs_d,
                d_mask,
                MASKED=True,
                IS_CAUSAL=True,
                HEAD_SIZE=HEAD_SIZE,
                BLOCK_N=BLOCK_N,
            )

    # A row whose keys were all masked out has no contribution to renormalize.
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]

    o_ptrs = out_ptr + q_rows[:, None] * o_row_stride + hq * HEAD_SIZE + offs_d[None, :]
    tl.store(
        o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & d_mask[None, :]
    )


# Vendors whose ``torch_device_fn`` answers with a *CUDA* compute capability.
_CAPABILITY_VENDORS = {"nvidia"}


def _device_capability():
    """``(major, minor)`` of the current device, or ``None`` if it will not say.

    ``None`` means "cannot tell", and the caller then leaves the device alone
    rather than guessing a limit it did not state. A vendor that reports in
    another scale counts as "cannot tell" for the same reason -- the number is a
    true answer to a different question, and the caller's limit is an sm_XX one.
    """
    if device.vendor_name not in _CAPABILITY_VENDORS:
        return None
    try:
        return torch_device_fn.get_device_capability(torch_device_fn.current_device())
    except Exception:
        return None


def _check_launch_arguments(q, k, v, head_size, num_heads_q, num_heads_kv):
    """Reject inputs the kernel cannot serve."""
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.device.type == "cpu":
            raise ValueError(f"{name} must be a device tensor, not on the CPU")

    if head_size > 256:
        raise ValueError(f"head_size must be <= 256, got {head_size}")
    if head_size % 8 != 0:
        raise ValueError(f"head_size must be divisible by 8, got {head_size}")

    capability = _device_capability()
    if capability is not None and capability[0] < 8:
        raise ValueError(
            "blocked flash needs a compute capability of 8.0 or newer; this "
            "device reports %d.%d" % capability
        )

    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("blocked flash attention only supports fp16 and bf16")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("query, key and value must share a dtype")
    if num_heads_q % num_heads_kv != 0:
        raise ValueError("n_heads_q must be divisible by n_heads_kv")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError(
            "query, key and value must be contiguous in the last dimension"
        )


def _blocked_flash_launch(
    out,
    q,
    k,
    v,
    attention_atoms,
    softmax_scale,
    q_block_size,
):
    """Launch the pointer-atom kernel.

    ``attention_atoms`` is DeepSpeed's ``AttentionAtom`` layout: slots [0:2]
    are the 64-bit address of the sequence's block-id list. ``q_block_size`` is
    the fixed geometry from ``get_q_block_size(head_size)``; the atoms are
    expected to have been built with it.
    """
    logger.debug("TRAIN BLOCKED_FLASH")

    head_size = k.size(-1)
    num_heads_q = q.size(-1) // head_size
    num_heads_kv = k.size(-2)
    num_atoms = attention_atoms.size(0)

    _check_launch_arguments(q, k, v, head_size, num_heads_q, num_heads_kv)

    kv_block_size = k.stride(0) // k.stride(1)
    if kv_block_size * k.stride(1) != k.stride(0):
        raise ValueError("key cache must be contiguous across the block dimension")
    if v.stride(1) == 0 or kv_block_size * v.stride(1) != v.stride(0):
        raise ValueError("value cache must be contiguous across the block dimension")

    BLOCK_M = max(triton.next_power_of_2(q_block_size), 16)
    BLOCK_N = min(1 << (int(kv_block_size).bit_length() - 1), _BLOCK_N_MAX)
    BLOCK_D = triton.next_power_of_2(head_size)

    num_warps = max(4, (BLOCK_M * BLOCK_D) // 2048)

    _blocked_flash_fwd_kernel[(num_atoms, num_heads_q)](
        out,
        q,
        k,
        v,
        attention_atoms,
        softmax_scale * _LOG2E,
        q.stride(0),
        out.stride(0),
        k.stride(1),
        v.stride(1),
        num_heads_q // num_heads_kv,
        KV_BLOCK_SIZE=kv_block_size,
        HEAD_SIZE=head_size,
        num_warps=num_warps,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return out


# ---------------------------------------------------------------------------
def _compat_device_capability():
    """``(major, minor)`` of the current CUDA device, in CUDA's scale."""
    return torch_device_fn.get_device_capability(torch_device_fn.current_device())


def _geometry_capability():
    """Capability scale the DeepSpeed geometry table is evaluated in.

    The table is written in CUDA's sm_XX terms. On NVIDIA the device answers in
    that scale; on another vendor its number is a true answer to a different
    question, so the table is evaluated as cc 8.0 -- the baseline DeepSpeed's
    own choices are written against -- rather than translated.
    """
    if device.vendor_name == "nvidia":
        return _compat_device_capability()
    return (8, 0)


def get_q_block_size(head_size: int) -> int:
    """Query rows per attention atom, matching DeepSpeed's geometry table."""
    cc_major, cc_minor = _geometry_capability()

    if cc_major < 8:
        raise RuntimeError("Blocked attention requires CUDA compute capability >= 8.0")

    if head_size <= 64:
        return 128
    elif head_size <= 160:
        if cc_minor != 0:
            return 64
        else:
            return 128
    elif head_size == 192:
        return 128
    elif head_size == 224:
        if cc_minor != 0:
            return 64
        else:
            return 128
    else:
        if cc_major == 8 and cc_minor == 0:
            return 128
        else:
            return 64


def get_kv_block_size(head_size: int) -> int:
    """KV tokens per cache block, matching DeepSpeed's geometry table."""
    cc_major, cc_minor = _geometry_capability()

    if cc_major < 8:
        raise RuntimeError("Blocked attention requires CUDA compute capability >= 8.0")

    if head_size <= 64:
        return 128
    elif head_size != 160 or cc_minor != 0:
        return 64
    else:
        return 32


_DTYPE_ALIASES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "torch.float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "bfloat": torch.bfloat16,
    "torch.bfloat16": torch.bfloat16,
}


def _normalize_dtype(dtype):
    """``torch.dtype`` for a torch dtype, a DtypeEnum-like value, or an alias."""
    if isinstance(dtype, torch.dtype):
        return dtype
    value = getattr(dtype, "value", None)
    if isinstance(value, torch.dtype):
        return value
    if isinstance(dtype, str):
        return _DTYPE_ALIASES.get(dtype.lower())
    return None


def _resolve_compat_call(out, q, k, v, attention_atoms, head_size, dtype):
    """Shared front-end for the public ``blocked_flash`` on every backend.

    Resolves the optional ``head_size``/``dtype``, validates the tensor and atom
    contract, and returns ``(head_size, kv_block_size)``. The caller still runs
    its own ``_check_launch_arguments`` (capability, last-dim stride, GQA) and
    launches its backend kernel.
    """
    resolved_dtype = q.dtype if dtype is None else _normalize_dtype(dtype)
    if resolved_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            "Unsupported data type: {}, supported data types are {}".format(
                dtype if dtype is not None else q.dtype,
                [torch.float16, torch.bfloat16],
            )
        )
    if dtype is not None and resolved_dtype != q.dtype:
        raise ValueError(
            "dtype {} does not match q.dtype {}".format(resolved_dtype, q.dtype)
        )

    if q.dim() != 2:
        raise ValueError("q must be a 2-D [total_q, n_heads_q * head_size] tensor")
    if k.dim() != 4 or v.dim() != 4:
        raise ValueError("k and v must be 4-D [n_blocks, block_size, heads, dim]")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if out.shape != q.shape:
        raise ValueError("out must have the same shape as q")
    if out.dtype != q.dtype:
        raise ValueError("out must have the same dtype as q")
    if not (q.device == k.device == v.device == out.device == attention_atoms.device):
        raise ValueError("out, q, k, v and attention_atoms must share one device")
    if (
        attention_atoms.dtype != torch.int32
        or attention_atoms.dim() != 2
        or attention_atoms.size(1) != int(_ATOM_STRIDE)
    ):
        raise ValueError("attention_atoms must be a [num_atoms, 8] int32 tensor")
    if not attention_atoms.is_contiguous():
        raise ValueError("attention_atoms must be contiguous")

    actual_head_size = k.size(-1)
    if head_size is None:
        resolved_head_size = actual_head_size
    else:
        if not isinstance(head_size, int) or isinstance(head_size, bool):
            raise ValueError("head_size must be an int, got {!r}".format(head_size))
        if head_size != actual_head_size:
            raise ValueError(
                "head_size %d does not match k.size(-1) %d"
                % (head_size, actual_head_size)
            )
        resolved_head_size = head_size

    if resolved_head_size <= 0:
        raise ValueError("head_size must be positive, got %d" % resolved_head_size)
    if resolved_head_size % 16 != 0:
        raise ValueError(
            "Head size must be divisible by 16 (configured with {})".format(
                resolved_head_size
            )
        )

    if k.stride(1) == 0:
        raise ValueError("key cache row stride must be non-zero")
    kv_block_size = k.stride(0) // k.stride(1)
    if kv_block_size * k.stride(1) != k.stride(0):
        raise ValueError("key cache must be contiguous across the block dimension")
    if v.stride(1) == 0:
        raise ValueError("value cache row stride must be non-zero")
    if kv_block_size * v.stride(1) != v.stride(0):
        raise ValueError("value cache must be contiguous across the block dimension")
    expected_kv_block_size = get_kv_block_size(resolved_head_size)
    if kv_block_size != expected_kv_block_size:
        raise ValueError(
            "key cache block size is %d, but get_kv_block_size(%d) is %d; the "
            "DeepSpeed atom geometry requires the latter"
            % (kv_block_size, resolved_head_size, expected_kv_block_size)
        )

    return resolved_head_size, kv_block_size


def flash_attn_by_atoms(
    out,
    q,
    k,
    v,
    attention_atoms,
    softmax_scale,
):
    """Flash attention forward over the blocked KV-cache (DeepSpeed atom layout)."""
    if device.vendor_name == "nvidia":
        capability = _compat_device_capability()
        is_sm8x = capability[0] == 8 and capability[1] >= 0
        is_sm90 = capability[0] == 9 and capability[1] == 0
        if not (is_sm8x or is_sm90):
            raise RuntimeError("FlashAttention only supports Ampere GPUs or newer.")

    q_dtype = q.dtype
    if q_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("FlashAttention only support fp16 and bf16 data type")
    if k.dtype != q_dtype:
        raise RuntimeError("query and key must have the same dtype")
    if v.dtype != q_dtype:
        raise RuntimeError("query and value must have the same dtype")

    for tensor in (q, k, v):
        if tensor.device.type == "cpu":
            raise RuntimeError("Input tensor must be on a device")

    for tensor in (q, k, v):
        if tensor.stride(-1) != 1:
            raise RuntimeError("Input tensor must have contiguous last dimension")

    if q.dim() != 2:
        raise ValueError("q must be a 2-D [total_q, n_heads_q * head_size] tensor")
    if k.dim() != 4 or v.dim() != 4:
        raise ValueError("k and v must be 4-D [n_blocks, block_size, heads, dim]")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if out.shape != q.shape:
        raise ValueError("out must have the same shape as q")
    if out.dtype != q_dtype:
        raise ValueError("out must have the same dtype as q")
    if not (q.device == k.device == v.device == out.device == attention_atoms.device):
        raise ValueError("out, q, k, v and attention_atoms must share one device")
    if (
        attention_atoms.dtype != torch.int32
        or attention_atoms.dim() != 2
        or attention_atoms.size(1) != int(_ATOM_STRIDE)
    ):
        raise ValueError("attention_atoms must be a [num_atoms, 8] int32 tensor")
    if not attention_atoms.is_contiguous():
        raise ValueError("attention_atoms must be contiguous")

    head_size = k.size(-1)
    num_heads_kv = k.size(-2)
    if head_size <= 0:
        raise RuntimeError("head_size must be positive")
    if num_heads_kv <= 0:
        raise RuntimeError("num_heads_kv must be positive")
    num_heads_q = q.size(-1) // head_size

    if head_size > 256:
        raise RuntimeError("head_size must be <= 256")
    if head_size % 8 != 0:
        raise RuntimeError("head_size must be divisible by 8")
    if num_heads_q % num_heads_kv != 0:
        raise RuntimeError("num_heads_q must be divisible by num_heads_kv")

    if k.stride(1) == 0:
        raise ValueError("key cache row stride must be non-zero")
    kv_block_size = k.stride(0) // k.stride(1)
    if kv_block_size * k.stride(1) != k.stride(0):
        raise ValueError("key cache must be contiguous across the block dimension")
    if v.stride(1) == 0:
        raise ValueError("value cache row stride must be non-zero")
    if kv_block_size * v.stride(1) != v.stride(0):
        raise ValueError("value cache must be contiguous across the block dimension")

    expected_kv_block_size = get_kv_block_size(head_size)
    if kv_block_size != expected_kv_block_size:
        raise ValueError(
            "key cache block size is %d, but get_kv_block_size(%d) is %d; the "
            "DeepSpeed atom geometry requires the latter"
            % (kv_block_size, head_size, expected_kv_block_size)
        )

    _blocked_flash_launch(
        out,
        q,
        k,
        v,
        attention_atoms,
        softmax_scale,
        get_q_block_size(head_size),
    )


def blocked_flash(
    out,
    q,
    k,
    v,
    attention_atoms,
    softmax_scale,
    *,
    head_size=None,
    dtype=None,
):
    """Blocked flash attention with DeepSpeed's atom layout.

    ``attention_atoms`` must be built with ``get_q_block_size(head_size)`` --
    the same bound DeepSpeed's ``AtomBuilder`` uses. An atom whose ``q_len``
    exceeds that bound is outside the contract and its extra query rows are not
    covered by the fixed launch tile.
    """
    _resolve_compat_call(out, q, k, v, attention_atoms, head_size, dtype)
    flash_attn_by_atoms(out, q, k, v, attention_atoms, softmax_scale)
    return out


__all__ = ["blocked_flash"]
