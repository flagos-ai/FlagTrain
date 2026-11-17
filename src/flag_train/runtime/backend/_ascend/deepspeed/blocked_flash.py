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
"""Ascend-tuned blocked flash attention.

Same public contract as the generic operator. Adds a query-tile cap and a flat
grid because this backend's Triton overflows the unified buffer with the
generic tiling."""

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

_NARROW_QUERY_TILE = 64
_WIDE_QUERY_TILE = 128

_WIDE_HEAD_D = 256

_WIDE_KV_TILE = 64


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
    n_q_tiles,
    KV_BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    q_tile = pid % n_q_tiles
    hq = (pid // n_q_tiles) % n_heads_q
    atom = pid // (n_q_tiles * n_heads_q)

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

    q_tile_start = q_tile * BLOCK_M
    if q_tile_start >= q_len:
        return

    row = q_tile_start + tl.arange(0, BLOCK_M)
    m_mask = row < q_len

    hk = hq // h_h_k_ratio

    # --- queries -----------------------------------------------------------
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < HEAD_SIZE

    q_rows = q_start + row
    q_ptrs = q_ptr + q_rows[:, None] * q_row_stride + hq * HEAD_SIZE + offs_d[None, :]
    q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    global_q_pos = global_q_idx + row

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for b in range(0, kv_blocks):
        block = tl.load(block_ptr + b)
        k_block_row_base = block * k_row_stride * KV_BLOCK_SIZE
        v_block_row_base = block * v_row_stride * KV_BLOCK_SIZE

        for t0 in range(0, KV_BLOCK_SIZE, BLOCK_N):
            offs_n = t0 + tl.arange(0, BLOCK_N)
            kv_pos = b * KV_BLOCK_SIZE + offs_n
            n_mask = kv_pos < total_extent

            k_ptrs = (
                k_ptr
                + k_block_row_base
                + offs_n[:, None] * k_row_stride
                + hk * HEAD_SIZE
                + offs_d[None, :]
            )
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

            qk = tl.dot(q, tl.trans(k)) * qk_scale

            keep = (kv_pos[None, :] <= global_q_pos[:, None]) & n_mask[None, :]
            qk = tl.where(keep, qk, float("-inf"))

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
            v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

    # A row whose keys were all masked out has no contribution to renormalize.
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]

    o_ptrs = out_ptr + q_rows[:, None] * o_row_stride + hq * HEAD_SIZE + offs_d[None, :]
    tl.store(
        o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & d_mask[None, :]
    )


def _query_tile_cap(block_d, block_n):
    """The widest query tile that compiles, for a given head and KV tile width."""
    if block_d < _WIDE_HEAD_D and block_n <= _WIDE_KV_TILE:
        return _WIDE_QUERY_TILE
    return _NARROW_QUERY_TILE


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
    """Flash attention forward over the blocked KV-cache (Ascend).

    Public interface matches the NVIDIA generic operator: the atom tensor uses
    DeepSpeed's pointer layout (slots [0:2] are the block-list address), there
    is no ``kv_block_idx`` argument, ``head_size``/``dtype`` are optional, and
    the operator is causal-only.
    """
    logger.debug("TRAIN_ASCEND BLOCKED_FLASH")

    head_size, kv_block_size = _resolve_compat_call(
        out, q, k, v, attention_atoms, head_size, dtype
    )
    num_heads_q = q.size(-1) // head_size
    num_heads_kv = k.size(-2)
    num_atoms = attention_atoms.size(0)

    _check_launch_arguments(q, k, v, head_size, num_heads_q, num_heads_kv)
    q_len_max = get_q_block_size(head_size)

    BLOCK_D = triton.next_power_of_2(head_size)
    BLOCK_N = min(triton.next_power_of_2(kv_block_size), 128)

    cap = _query_tile_cap(BLOCK_D, BLOCK_N)
    BLOCK_M = max(min(triton.next_power_of_2(q_len_max), cap), 16)
    n_q_tiles = (q_len_max + BLOCK_M - 1) // BLOCK_M

    num_warps = max(4, (BLOCK_M * BLOCK_D) // 2048)

    _blocked_flash_fwd_kernel[(num_atoms * num_heads_q * n_q_tiles,)](
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
        n_q_tiles,
        KV_BLOCK_SIZE=kv_block_size,
        HEAD_SIZE=head_size,
        num_warps=num_warps,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return out
