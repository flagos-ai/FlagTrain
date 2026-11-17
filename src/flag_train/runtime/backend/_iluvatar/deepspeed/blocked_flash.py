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
"""Iluvatar-tuned blocked flash attention.

Same public contract as the generic operator. The launcher probes the
compiler's shared-memory allocation and narrows the KV tile below the device's
limit; the kernel uses a single masked loop, the shape this backend's Triton
compiles reliably."""

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

_BLOCK_N_MAX = 64

_BLOCK_N_MIN = 16

_NUM_STAGES = 1

_MMA_M = 16
_NUM_WARPS_MIN = 4
_NUM_WARPS_MAX = 8


@functools.lru_cache(maxsize=None)
def _device_properties(device_index):
    """The device's property dict, or ``None`` if it cannot be read."""
    try:
        from triton.runtime.driver import driver as triton_driver

        return triton_driver.active.utils.get_device_properties(device_index)
    except Exception:
        return None


def _max_shared_memory(device_index):
    """The device's shared-memory limit in bytes, or ``None`` if unreadable."""
    props = _device_properties(device_index)
    limit = props.get("max_shared_mem") if props else None
    return int(limit) if limit else None


def _num_warps_for(block_m):
    """Warps in the CTA: one MMA row-tile along M, floored at 4 and capped at 8."""
    return min(_NUM_WARPS_MAX, max(_NUM_WARPS_MIN, block_m // _MMA_M))


def _block_n_candidates(kv_block_size):
    """The KV tiles this cache's block size can be walked in, widest first."""
    if kv_block_size < _BLOCK_N_MIN:
        raise ValueError(
            "blocked flash walks a cache block in KV tiles of at least %d rows, "
            "so the key cache's block size must be at least that; this one is %d"
            % (_BLOCK_N_MIN, kv_block_size)
        )
    widest = min(1 << (int(kv_block_size).bit_length() - 1), _BLOCK_N_MAX)
    candidates = []
    while widest >= _BLOCK_N_MIN:
        candidates.append(widest)
        widest //= 2
    return candidates


def _allocated_shared_memory(
    out,
    q,
    k,
    v,
    attention_atoms,
    softmax_scale,
    kv_block_size,
    head_size,
    h_h_k_ratio,
    num_heads_q,
    num_warps,
    block_m,
    block_n,
    block_d,
):
    """Shared memory Triton allocates for one tiling, in bytes, without launching."""
    compiled = _blocked_flash_fwd_kernel.jit_function.warmup(
        out,
        q,
        k,
        v,
        attention_atoms,
        softmax_scale,
        q.stride(0),
        out.stride(0),
        k.stride(1),
        v.stride(1),
        h_h_k_ratio,
        num_heads_q,
        grid=(1,),
        KV_BLOCK_SIZE=kv_block_size,
        HEAD_SIZE=head_size,
        num_warps=num_warps,
        num_stages=_NUM_STAGES,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )
    # ``warmup`` returns the same ``(kernel, hooks)`` pair a launch does.
    compiled = compiled[0] if isinstance(compiled, tuple) else compiled
    return compiled.metadata.shared


def _widest_tile_that_fits(
    needed_for_tile,
    limit,
    candidates,
    block_m,
    head_size,
):
    """The widest KV tile whose tiling the device can hold, or ``None``."""
    last = None
    narrowest = None
    compile_error = None
    for candidate in candidates:
        try:
            size = needed_for_tile(candidate)
        except Exception as exc:
            if compile_error is None:
                compile_error = exc
            continue
        last = size
        narrowest = candidate
        if size <= limit:
            return candidate
    if last is None:
        raise compile_error
    raise RuntimeError(
        "blocked flash needs %d bytes of shared memory even at the narrowest KV "
        "tile this kernel compiles with (%d rows), for a %d-row query tile at "
        "head_size %d, and the device has %d. The query tile is fixed by the "
        "DeepSpeed geometry for this head size."
        % (last, narrowest, block_m, head_size, limit)
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

    TILES_PER_BLOCK: tl.constexpr = (KV_BLOCK_SIZE + BLOCK_N - 1) // BLOCK_N
    for i in range(0, kv_blocks * TILES_PER_BLOCK):
        b = i // TILES_PER_BLOCK
        t0 = (i % TILES_PER_BLOCK) * BLOCK_N
        block = tl.load(block_ptr + b)
        k_block_row_base = block * k_row_stride * KV_BLOCK_SIZE
        v_block_row_base = block * v_row_stride * KV_BLOCK_SIZE

        offs_n = t0 + tl.arange(0, BLOCK_N)
        kv_pos = b * KV_BLOCK_SIZE + offs_n
        n_mask = kv_pos < total_extent

        v_ptrs = (
            v_ptr
            + v_block_row_base
            + offs_n[:, None] * v_row_stride
            + hk * HEAD_SIZE
            + offs_d[None, :]
        )
        v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        kt_ptrs = (
            k_ptr
            + k_block_row_base
            + hk * HEAD_SIZE
            + offs_d[:, None]
            + offs_n[None, :] * k_row_stride
        )
        kt = tl.load(kt_ptrs, mask=d_mask[:, None] & n_mask[None, :], other=0.0)

        qk = tl.dot(q, kt) * qk_scale
        # A query attends to every KV position up to and including its own.
        qk = tl.where(kv_pos[None, :] <= global_q_pos[:, None], qk, float("-inf"))
        qk = tl.where(n_mask[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])

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
    """Flash attention forward over the blocked KV-cache (Iluvatar).

    Public interface matches the NVIDIA generic operator: pointer-layout atoms,
    no ``kv_block_idx``, optional ``head_size``/``dtype``, causal only.
    """
    logger.debug("TRAIN_ILUVATAR BLOCKED_FLASH")

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

    device_index = q.device.index if q.device.index is not None else 0
    num_warps = _num_warps_for(BLOCK_M)

    candidates = _block_n_candidates(kv_block_size)
    limit = _max_shared_memory(device_index)
    if limit is None:
        BLOCK_N = candidates[0]
    else:

        def needed_for_tile(candidate):
            return _allocated_shared_memory(
                out,
                q,
                k,
                v,
                attention_atoms,
                softmax_scale * _LOG2E,
                kv_block_size,
                head_size,
                num_heads_q // num_heads_kv,
                num_heads_q,
                num_warps,
                BLOCK_M,
                candidate,
                BLOCK_D,
            )

        BLOCK_N = _widest_tile_that_fits(
            needed_for_tile, limit, candidates, BLOCK_M, head_size
        )

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
