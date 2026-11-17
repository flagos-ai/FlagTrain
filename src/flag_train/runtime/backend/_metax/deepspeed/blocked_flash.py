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
"""MetaX-tuned blocked flash attention.

Same public contract as the generic operator. The launcher caps the query tile
at 64 rows and selects a KV tile against the compiler's shared-memory figure;
the kernel uses the transposed KV load."""

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
    _kv_tile,
    _resolve_compat_call,
    get_q_block_size,
)
from flag_train.utils import libentry

logger = logging.getLogger(__name__)

_MIN_BLOCK_N = 16

_BLOCK_M_MAX = 64

_BLOCK_N_CAP_WIDE_HEAD = 32
_BLOCK_N_CAP_NARROW_HEAD = 64
_NARROW_HEAD_D = 64

_NUM_STAGES_WIDE_HEAD = 1
_NUM_STAGES_NARROW_HEAD = 3

_MAX_THREADS_PER_CTA = 512


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
    hq = pid % n_heads_q
    rest = pid // n_heads_q
    tq = rest % n_q_tiles
    atom = tl.num_programs(0) // (n_heads_q * n_q_tiles) - 1 - rest // n_q_tiles

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
    m_start = tq * BLOCK_M
    if m_start >= q_len:
        return
    offs_m = m_start + tl.arange(0, BLOCK_M)
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
        total_extent // KV_BLOCK_SIZE,
        (global_q_idx + m_start + 1) // KV_BLOCK_SIZE,
    )
    peeled = tl.minimum(peeled, kv_blocks)

    for b in range(0, peeled):
        block = tl.load(block_ptr + b)
        k_block_row_base = block * k_row_stride * KV_BLOCK_SIZE
        v_block_row_base = block * v_row_stride * KV_BLOCK_SIZE

        for t0 in range(0, KV_BLOCK_SIZE, BLOCK_N):
            offs_n = t0 + tl.arange(0, BLOCK_N)
            m_i, l_i, acc = _kv_tile(
                q,
                k_ptr,
                v_ptr,
                k_block_row_base,
                v_block_row_base,
                offs_n,
                b * KV_BLOCK_SIZE + offs_n,
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
                TRANSPOSED=True,
            )

    for b in range(peeled, kv_blocks):
        block = tl.load(block_ptr + b)
        k_block_row_base = block * k_row_stride * KV_BLOCK_SIZE
        v_block_row_base = block * v_row_stride * KV_BLOCK_SIZE
        for t0 in range(0, KV_BLOCK_SIZE, BLOCK_N):
            offs_n = t0 + tl.arange(0, BLOCK_N)
            m_i, l_i, acc = _kv_tile(
                q,
                k_ptr,
                v_ptr,
                k_block_row_base,
                v_block_row_base,
                offs_n,
                b * KV_BLOCK_SIZE + offs_n,
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
                TRANSPOSED=True,
            )

    # A row whose keys were all masked out has no contribution to renormalize.
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]

    o_ptrs = out_ptr + q_rows[:, None] * o_row_stride + hq * HEAD_SIZE + offs_d[None, :]
    tl.store(
        o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & d_mask[None, :]
    )


@functools.lru_cache(maxsize=None)
def _device_properties(device_index):
    """The device's property dict, or ``None`` if it cannot be read.

    Cached. The call is cheap here -- about 2 us measured on a C550, against
    kernels that run in tens of microseconds, and it is asked for once per
    launch -- so this is not load-bearing the way it is on parts where the
    property read is milliseconds. It is kept because the read is still on the
    per-call path and device properties do not change under a running process,
    so there is nothing to re-read and nothing to invalidate.
    """
    try:
        from triton.runtime.driver import driver as triton_driver

        return triton_driver.active.utils.get_device_properties(device_index)
    except Exception:
        return None


def _warp_size(device_index):
    """Threads per warp, or 32 when the device will not say.

    A warp is not a fixed width across devices -- 32 on the NVIDIA parts this
    operator was first tuned on, 64 here -- and the per-CTA sizing below is a
    count of *threads*, so it has to know which. 32 is the value that sizing
    already assumed before it could ask, so an unreadable device keeps that
    behaviour rather than getting a new one.
    """
    props = _device_properties(device_index)
    size = props.get("warpSize") if props else None
    return int(size) if size else 32


_SELECTED_BLOCK_N = {}


def _max_shared_memory(device_index):
    """The device's shared-memory limit in bytes, or ``None`` if unreadable.

    ``None`` means "cannot tell", and the caller then leaves the tiling alone
    rather than guessing a limit it did not state -- the same rule the input
    checks follow for a device that will not report a compute capability.
    """
    props = _device_properties(device_index)
    limit = props.get("max_shared_mem") if props else None
    return int(limit) if limit else None


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
    n_heads_q,
    n_q_tiles,
    num_warps,
    num_stages,
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
        n_heads_q,
        n_q_tiles,
        grid=(1,),
        KV_BLOCK_SIZE=kv_block_size,
        HEAD_SIZE=head_size,
        num_warps=num_warps,
        num_stages=num_stages,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )
    # ``warmup`` returns the same ``(kernel, hooks)`` pair a launch does.
    compiled = compiled[0] if isinstance(compiled, tuple) else compiled
    return compiled.metadata.shared


def _block_n_for(block_d, kv_block_size):
    """The KV tile this head size wants, before the device narrows it."""
    cap = (
        _BLOCK_N_CAP_WIDE_HEAD if block_d > _NARROW_HEAD_D else _BLOCK_N_CAP_NARROW_HEAD
    )
    # The largest power of two dividing ``kv_block_size`` (its lowest set bit).
    block_n = min(cap, int(kv_block_size) & -int(kv_block_size))
    if block_n < _MIN_BLOCK_N:
        raise ValueError(
            "blocked flash walks a cache block in power-of-two KV tiles of at "
            "least %d rows, so the key cache's block size must be divisible by "
            "such a tile; this one is %d" % (_MIN_BLOCK_N, kv_block_size)
        )
    return block_n


def _select_block_n(block_n, allocated, limit, block_m, head_size):
    """Narrow the KV tile until the device can hold the tiling, before launching."""
    while True:
        needed = allocated(block_n)
        if needed <= limit or block_n <= _MIN_BLOCK_N:
            break
        block_n //= 2
    if needed > limit:
        smaller = (
            " %d rows is the next tile down." % (block_m // 2)
            if block_m > _MIN_BLOCK_N
            else " %d rows is already the narrowest query tile there is." % _MIN_BLOCK_N
        )
        raise RuntimeError(
            "blocked flash needs %d bytes of shared memory for a %d-row query "
            "tile at head_size %d, and the device has %d. That is already at the "
            "narrowest KV tile that compiles (%d rows).%s The query tile is fixed "
            "by the DeepSpeed geometry for this head size."
            % (needed, block_m, head_size, limit, _MIN_BLOCK_N, smaller)
        )
    return block_n


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
    """Flash attention forward over the blocked KV-cache (MetaX).

    Public interface matches the NVIDIA generic operator: pointer-layout atoms,
    no ``kv_block_idx``, optional ``head_size``/``dtype``, causal only.
    """
    logger.debug("TRAIN_METAX BLOCKED_FLASH")

    head_size, kv_block_size = _resolve_compat_call(
        out, q, k, v, attention_atoms, head_size, dtype
    )
    num_heads_q = q.size(-1) // head_size
    num_heads_kv = k.size(-2)
    num_atoms = attention_atoms.size(0)

    _check_launch_arguments(q, k, v, head_size, num_heads_q, num_heads_kv)
    q_len_max = get_q_block_size(head_size)

    BLOCK_M = min(_BLOCK_M_MAX, max(triton.next_power_of_2(q_len_max), 16))
    n_q_tiles = max(1, (q_len_max + BLOCK_M - 1) // BLOCK_M)
    BLOCK_D = triton.next_power_of_2(head_size)
    BLOCK_N = _block_n_for(BLOCK_D, kv_block_size)

    device_index = q.device.index if q.device.index is not None else 0
    element_size = q.element_size()

    warp_size = _warp_size(device_index)
    num_warps = max(4, (BLOCK_M * BLOCK_D) // (64 * warp_size))
    num_warps = min(num_warps, max(1, _MAX_THREADS_PER_CTA // warp_size))
    num_stages = (
        _NUM_STAGES_WIDE_HEAD if BLOCK_D > _NARROW_HEAD_D else _NUM_STAGES_NARROW_HEAD
    )

    key = (device_index, BLOCK_M, BLOCK_D, kv_block_size, element_size, BLOCK_N)
    if key not in _SELECTED_BLOCK_N:
        limit = _max_shared_memory(device_index)

        def allocated(block_n):
            return _allocated_shared_memory(
                out,
                q,
                k,
                v,
                attention_atoms,
                softmax_scale,
                kv_block_size,
                head_size,
                num_heads_q // num_heads_kv,
                num_heads_q,
                n_q_tiles,
                num_warps,
                num_stages,
                BLOCK_M,
                block_n,
                BLOCK_D,
            )

        _SELECTED_BLOCK_N[key] = (
            BLOCK_N
            if limit is None
            else _select_block_n(BLOCK_N, allocated, limit, BLOCK_M, head_size)
        )
    block_n = _SELECTED_BLOCK_N[key]

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
        num_stages=num_stages,
        BLOCK_M=BLOCK_M,
        BLOCK_N=block_n,
        BLOCK_D=BLOCK_D,
    )
    return out
