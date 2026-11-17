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
"""Hygon-tuned blocked flash attention.

Same public contract as the generic operator. The device-specific part is the
32-row KV tile and 64-thread warp sizing, which fit Hygon's shared-memory
budget; the kernel follows the generic pointer-atom layout."""

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


_NUM_STAGES = 3


def _shared_memory_for(block_m, block_n, block_d, element_size, num_stages=_NUM_STAGES):
    """Shared memory one program needs, in bytes, for a given tiling."""
    kv_bytes = 2 * num_stages * block_n * block_d
    return max(block_m * block_d, kv_bytes + block_m * block_n) * element_size


def _widest_query_tile(block_d, element_size, limit):
    """The largest power-of-two query tile a device can hold, for an advisory.

    Used only to report the widest query tile this head size fits when even the
    narrowest KV tile does not fit; it is the bare query tile at ``block_n = 1``,
    so it is an upper bound on what will actually launch.
    """
    block_m = 1
    while _shared_memory_for(block_m * 2, 1, block_d, element_size) <= limit:
        block_m *= 2
    return block_m


@functools.lru_cache(maxsize=None)
def _device_properties(device_index):
    """The device's property dict, or ``None`` if it cannot be read.

    Cached, and it has to be: the driver call costs about **1 ms**, against
    kernels that run in tens of microseconds, and this host function asks for it
    twice per invocation. Uncached it does not size the tiling, it *is* the
    timing -- 0.92 average speedup down to 0.14, measured on an A100. Device
    properties do not change under a running process, so nothing to invalidate.
    """
    try:
        from triton.runtime.driver import driver as triton_driver

        return triton_driver.active.utils.get_device_properties(device_index)
    except Exception:
        return None


def _max_shared_memory(device_index):
    """The device's shared-memory limit in bytes, or ``None`` if unreadable.

    ``None`` means "cannot tell", and the caller then leaves the tiling alone
    rather than guessing a limit -- the same behaviour as before this was
    queryable at all.
    """
    props = _device_properties(device_index)
    limit = props.get("max_shared_mem") if props else None
    return int(limit) if limit else None


def _warp_size(device_index):
    """Threads per warp, or 32 when the device will not say.

    A warp is not a fixed width across devices -- 32 on the NVIDIA parts this
    was first tuned on, 64 here -- and the per-CTA sizing below is a count of
    *threads*, so it has to know which. 32 is the value that sizing already
    assumed before it could ask, so an unreadable device keeps that behaviour
    rather than getting a new one.
    """
    props = _device_properties(device_index)
    size = props.get("warpSize") if props else None
    return int(size) if size else 32


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
                + hk * HEAD_SIZE
                + offs_d[:, None]
                + offs_n[None, :] * k_row_stride
            )
            kT = tl.load(k_ptrs, mask=d_mask[:, None] & n_mask[None, :], other=0.0)

            qk = tl.dot(q, kT) * qk_scale
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
    """Flash attention forward over the blocked KV-cache (Hygon).

    Public interface matches the NVIDIA generic operator: the atom tensor uses
    DeepSpeed's pointer layout (slots [0:2] are the block-list address), there
    is no ``kv_block_idx`` argument, ``head_size``/``dtype`` are optional, and
    the operator is causal-only.
    """
    logger.debug("TRAIN_HYGON BLOCKED_FLASH")

    head_size, kv_block_size = _resolve_compat_call(
        out, q, k, v, attention_atoms, head_size, dtype
    )
    num_heads_q = q.size(-1) // head_size
    num_heads_kv = k.size(-2)
    num_atoms = attention_atoms.size(0)

    _check_launch_arguments(q, k, v, head_size, num_heads_q, num_heads_kv)
    q_len_max = get_q_block_size(head_size)

    BLOCK_M = max(triton.next_power_of_2(q_len_max), 16)
    BLOCK_N = min(triton.next_power_of_2(kv_block_size), 32)
    BLOCK_D = triton.next_power_of_2(head_size)

    device_index = q.device.index if q.device.index is not None else 0
    limit = _max_shared_memory(device_index)
    element_size = q.element_size()
    if limit is not None:
        while BLOCK_N > 1 and (
            _shared_memory_for(BLOCK_M, BLOCK_N, BLOCK_D, element_size) > limit
        ):
            BLOCK_N //= 2
        needed = _shared_memory_for(BLOCK_M, BLOCK_N, BLOCK_D, element_size)
        if needed > limit:
            # No KV tile is narrow enough, so the query tile itself is too wide.
            raise RuntimeError(
                "blocked flash needs %d bytes of shared memory for a %d-row query "
                "tile at head_size %d, but the device has %d. The widest query "
                "tile this head size fits here is %d rows."
                % (
                    needed,
                    BLOCK_M,
                    head_size,
                    limit,
                    _widest_query_tile(BLOCK_D, element_size, limit),
                )
            )

    num_warps = max(4, (BLOCK_M * BLOCK_D) // (64 * _warp_size(device_index)))

    _blocked_flash_fwd_kernel[(num_atoms * num_heads_q,)](
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
        num_warps=num_warps,
        num_stages=_NUM_STAGES,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return out
