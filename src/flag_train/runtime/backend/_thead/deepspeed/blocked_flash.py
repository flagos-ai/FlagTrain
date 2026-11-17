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
"""THead-tuned blocked flash attention.

Same public contract as the generic operator. The launcher picks a 4- or
8-warp kernel symbol by grid size, and the kernel splits the KV walk at the
causal diagonal."""

import functools
import logging

import triton
import triton.language as tl

from flag_train.deepspeed.blocked_flash import (
    _ATOM_GLOBAL_Q_IDX,
    _ATOM_KV_BLOCKS,
    _ATOM_Q_LEN,
    _ATOM_Q_START,
    _ATOM_STRIDE,
    _ATOM_TOTAL_EXTENT,
    _LOG2E,
    _check_launch_arguments,
    _resolve_compat_call,
    get_q_block_size,
)
from flag_train.utils import libentry

logger = logging.getLogger(__name__)

_BLOCK_N_CAP = 64

_MIN_BLOCK_N = 16

_WIDE_CTA_GRID_SMS = 1
_WIDE_HEAD_GRID_SMS = 8
_WIDE_HEAD_D = 64


@triton.jit
def _blocked_flash_fwd_body(
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
    n_heads_q,
    KV_BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    hq = pid % n_heads_q
    atom = tl.num_programs(0) // n_heads_q - 1 - pid // n_heads_q

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

    tiles_per_block = KV_BLOCK_SIZE // BLOCK_N
    n_tiles = kv_blocks * tiles_per_block
    n_full = global_q_idx // BLOCK_N

    for t in range(0, n_full):
        b = t // tiles_per_block
        t0 = (t % tiles_per_block) * BLOCK_N
        block = tl.load(block_ptr + b)
        k_block_row_base = block * k_row_stride * KV_BLOCK_SIZE
        v_block_row_base = block * v_row_stride * KV_BLOCK_SIZE
        offs_n = t0 + tl.arange(0, BLOCK_N)

        k_ptrs = (
            k_ptr
            + k_block_row_base
            + offs_n[:, None] * k_row_stride
            + hk * HEAD_SIZE
            + offs_d[None, :]
        )
        kk = tl.load(k_ptrs, mask=d_mask[None, :], other=0.0)

        qk = tl.dot(q, tl.trans(kk)) * qk_scale

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])

        v_ptrs = (
            v_ptr
            + v_block_row_base
            + offs_n[:, None] * v_row_stride
            + hk * HEAD_SIZE
            + offs_d[None, :]
        )
        vv = tl.load(v_ptrs, mask=d_mask[None, :], other=0.0)

        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(vv.dtype), vv)
        m_i = m_new

    for t in range(n_full, n_tiles):
        b = t // tiles_per_block
        t0 = (t % tiles_per_block) * BLOCK_N
        block = tl.load(block_ptr + b)
        k_block_row_base = block * k_row_stride * KV_BLOCK_SIZE
        v_block_row_base = block * v_row_stride * KV_BLOCK_SIZE
        offs_n = t0 + tl.arange(0, BLOCK_N)
        kv_pos = t * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = kv_pos < total_extent

        k_ptrs = (
            k_ptr
            + k_block_row_base
            + offs_n[:, None] * k_row_stride
            + hk * HEAD_SIZE
            + offs_d[None, :]
        )
        kk = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        qk = tl.dot(q, tl.trans(kk)) * qk_scale
        # A query attends to every KV position up to and including its own.
        qk = tl.where(kv_pos[None, :] <= global_q_pos[:, None], qk, float("-inf"))
        qk = tl.where(n_mask[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])

        v_ptrs = (
            v_ptr
            + v_block_row_base
            + offs_n[:, None] * v_row_stride
            + hk * HEAD_SIZE
            + offs_d[None, :]
        )
        vv = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(vv.dtype), vv)
        m_i = m_new

    # A row whose keys were all masked out has no contribution to renormalize.
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]

    o_ptrs = out_ptr + q_rows[:, None] * o_row_stride + hq * HEAD_SIZE + offs_d[None, :]
    tl.store(
        o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & d_mask[None, :]
    )


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
    n_heads_q,
    KV_BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """The 4-warp program. Its own symbol -- see the module docstring."""
    _blocked_flash_fwd_body(
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
        n_heads_q,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        HEAD_SIZE=HEAD_SIZE,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )


@libentry()
@triton.jit
def _blocked_flash_fwd_kernel_wide(
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
    n_heads_q,
    KV_BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """The 8-warp program: the same body under a symbol of its own."""
    _blocked_flash_fwd_body(
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
        n_heads_q,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        HEAD_SIZE=HEAD_SIZE,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )


@functools.lru_cache(maxsize=None)
def _device_properties(device_index):
    """The device's property dict, or ``None`` if it cannot be read.

    Cached: the read is on the per-call path and device properties do not change
    under a running process, so there is nothing to invalidate. ``None`` means
    "cannot tell", and the warp rule below then keeps the 4 warps that the
    generic launcher's tile arithmetic lands on anyway.
    """
    try:
        from triton.runtime.driver import driver as triton_driver

        return triton_driver.active.utils.get_device_properties(device_index)
    except Exception:
        return None


def _compute_units(device_index):
    """The device's compute-unit count, or ``None`` if unreadable."""
    props = _device_properties(device_index)
    count = props.get("multiprocessor_count") if props else None
    return int(count) if count else None


def _block_n_for(kv_block_size):
    """The KV tile for this cache block size: a power of two that divides it."""
    block_n = min(_BLOCK_N_CAP, 1 << (int(kv_block_size).bit_length() - 1))
    if block_n < _MIN_BLOCK_N:
        raise ValueError(
            "blocked flash walks a cache block in KV tiles of at least %d rows, "
            "so the key cache's block size must be at least that; this one is %d"
            % (_MIN_BLOCK_N, kv_block_size)
        )
    if kv_block_size % block_n != 0:
        raise ValueError(
            "blocked flash tiles a cache block at %d rows, which must divide the "
            "block; this block size is %d" % (block_n, kv_block_size)
        )
    return block_n


def _use_wide_cta(num_programs, block_d, device_index):
    """Whether a program gets 8 warps instead of 4."""
    compute_units = _compute_units(device_index)
    if compute_units is None:
        return False
    if num_programs < _WIDE_CTA_GRID_SMS * compute_units:
        return True
    if block_d > _WIDE_HEAD_D and num_programs < _WIDE_HEAD_GRID_SMS * compute_units:
        return True
    return False


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
    """Flash attention forward over the blocked KV-cache (THead).

    Public interface matches the NVIDIA generic operator: pointer-layout atoms,
    no ``kv_block_idx``, optional ``head_size``/``dtype``, causal only.
    """
    logger.debug("TRAIN_THEAD BLOCKED_FLASH")

    head_size, kv_block_size = _resolve_compat_call(
        out, q, k, v, attention_atoms, head_size, dtype
    )
    num_heads_q = q.size(-1) // head_size
    num_heads_kv = k.size(-2)
    num_atoms = attention_atoms.size(0)

    _check_launch_arguments(q, k, v, head_size, num_heads_q, num_heads_kv)
    q_len_max = get_q_block_size(head_size)

    BLOCK_M = max(triton.next_power_of_2(q_len_max), 16)
    BLOCK_D = triton.next_power_of_2(head_size)
    BLOCK_N = _block_n_for(kv_block_size)

    device_index = q.device.index if q.device.index is not None else 0
    wide = _use_wide_cta(num_atoms * num_heads_q, BLOCK_D, device_index)
    kernel = _blocked_flash_fwd_kernel_wide if wide else _blocked_flash_fwd_kernel

    kernel[(num_atoms * num_heads_q,)](
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
        num_heads_q,
        KV_BLOCK_SIZE=kv_block_size,
        HEAD_SIZE=head_size,
        num_warps=8 if wide else 4,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return out
