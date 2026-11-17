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
"""Ascend multi_tensor_adam.

The generic implementation is ``flag_train.deepspeed.multi_tensor_adam`` and is
what every backend without an override runs. Its launch shape is a CUDA one --
one 1024-lane masked program per 1024 elements, seven runtime scalars, and a
per-program metadata chain -- and on this backend (910B4, CANN 9.0 /
BiShengHIR, card 6, ``do_bench_npu`` KERNEL mode) that shape loses most of the
machine, worst on exactly the shapes Adam is used at:

* Adam moves seven streams (read g, p, m, v; write p, m, v). At 1024 lanes per
  program the per-program fixed cost -- tens of nanoseconds whatever it carries
  -- dominates, so 16 x 1M elements ran at 0.36 TB/s of a 1.08 TB/s HBM wall.
* It is lanes-per-program, not run length and not bytes, that sets the
  achievable bandwidth here; the plateau is 8192 lanes.
* Every runtime scalar argument costs throughput (~40% for the second one on
  this backend), and the generic kernel takes seven.

Measured per benchmark row (16 parameter tensors per case; ``vs Base`` before ->
after, train latency before -> after). fp32: 16 x 1024 15.8 -> 15.4 (3.10 ->
3.17 us), 16 x 4096 7.9 -> 12.5 (6.94 -> 4.40), 16 x 16384 2.8 -> 7.5 (19.7 ->
7.3), 16 x 65536 1.03 -> 4.0 (81.1 -> 20.7), 16 x 262144 0.52 -> 2.5 (291 ->
60.2), 16 x 1M 0.39 -> 1.15 (1238 -> 420). fp16: 18.0 -> 16.5, 8.3 -> 12.5,
2.9 -> 8.3, 1.05 -> 4.1, 0.37 -> 1.8, 0.30 -> 1.5 (1209 -> 240). The mean over
the six rows goes 4.73 -> 7.18 (fp32) and 5.16 -> 7.46 (fp16): the three large
rows -- the ones below parity -- land at 1.1-4.1, and the smallest row gives
back 0.1-0.3 us, which is its launch floor.

What this module does instead:

* **8192-lane tiles.** The tile is the largest power of two dividing the
  tensor's size (capped at 8192 lanes and at 32 KB per program), so its block
  grid is exact and the kernel runs *unmasked*. Parameter sizes are multiples of
  a hidden dimension, so that is the common case rather than a special one.
  8192 is also the largest fp32 tile the 192 KB UB holds for this much live
  state; a 16384-lane fp32 tile fails to compile outright.
* **Pre-offset pointer tables.** ``g_ptrs[i]`` is the address of block ``i``, so
  the device side is four independent loads and no address arithmetic.
* **A persistent walk.** When a launch would need many programs, each program
  walks several blocks: grid = ``clamp(cdiv(blocks, 16384/block), 32, 2048)``,
  i.e. two 8192-lane blocks per program. Measured optima were 2 blocks per
  program on 4M and 16M elements and 4 on 1M elements (within 5%), and the
  32-program floor is what keeps every vector core busy on the small rows.
* **A masked path for odd sizes**, because a size that is not a whole number of
  tiles cannot run the exact-grid kernel. Its index keeps a runtime block
  offset: a masked access indexed by a compile-time ``tl.arange`` alone on a
  runtime-loaded base miscompiles on this backend (see multi_tensor_lion.py).

The arithmetic is the generic kernel's, in the same order, with fp32
intermediates for every storage dtype (``MATH_T = float``), so results match it
to within one ulp. ``beta1``, ``beta2``, ``epsilon``, ``weight_decay`` and
``mode`` are compile-time constants (the first four are fixed for a training
run; ``mode`` selects the Adam vs AdamW branch, a constexpr branch that is free
when untaken). ``lr``, ``bc1`` and ``bc2`` stay runtime values: the bias
corrections change every step, and lr changes with the schedule.

Where the rows ended up: 16 x 1M fp32 at 420 us is 1.07 TB/s, the HBM wall;
16 x 262144 at 60 us is 1.9 TB/s (117 MB working set against a 168 MB L2);
16 x 1024 is at the launch floor. Dead ends, measured: 512/1024/2048-lane tiles
for anything above ~16 K elements (up to 2x worse), walks with fewer than 32
programs, and masking the exact path. Replacing the two bias-correction divides
with host-computed reciprocals (a <= 1 ulp change) was also measured: it buys
4% on the 16-bit 1M row and nothing anywhere else, so the generic divide form is
kept and the operator stays a faithful port.

Registration is by name. ``_ascend/deepspeed/__init__.py`` re-exports this
module's ``multi_tensor_adam``, and ``SpecOpRegistrar`` writes it over the
same-named generic function in ``flag_train.deepspeed`` when that package is
imported -- the namespace the tests and the benchmark read the operator from.
This module therefore imports nothing from ``flag_train.deepspeed``, which would
re-enter the registrar mid-scan (see ``runtime/backend/__init__.py``).
"""

import logging

import numpy as np
import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry

logger = logging.getLogger(__name__)

# Lanes per tile: measured throughput plateau and the largest fp32 tile the
# 192 KB UB holds for this kernel. ``_BLOCK_BYTES`` bounds the same tile by
# element size, so a wider element does not overrun the UB at that lane count.
_BLOCK_MAX = 8192
_BLOCK_BYTES = 32768
# The smallest tile that earns a block per program. A tensor below it is handled
# by the masked kernel at the smallest power of two covering it.
_BLOCK_MIN = 512

# Walk sizing: two 8192-lane blocks (~0.5 MB of traffic over Adam's seven
# streams) per program, at least 32 programs so every vector core has work, and
# never more than 2048. Measured optima: 2 blocks per program on 4M and 16M
# elements, 4 on 1M (within 5%), 32 programs on the small rows.
_LANES_PER_PROGRAM = 16384
_MIN_PROGRAMS = 32
_MAX_WALK_PROGRAMS = 2048

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _next_pow2(n):
    return 1 << (n - 1).bit_length() if n > 1 else 1


def _block_for(numel, itemsize):
    """The tile size for one tensor, or 0 for an empty tensor.

    The tile is the largest power of two dividing ``numel`` (capped at
    ``_BLOCK_MAX`` lanes and ``_BLOCK_BYTES`` bytes), so the tensor's block grid
    is exact -- ``numel % block == 0`` -- and the kernel can run unmasked. A
    size with a small power-of-two factor (an odd size) takes the masked kernel
    at the smallest tile that covers it.
    """
    if numel == 0:
        return 0
    cap = min(_BLOCK_MAX, _BLOCK_BYTES // itemsize)
    low = numel & -numel
    if low >= _BLOCK_MIN:
        return min(low, cap)
    return min(cap, max(_BLOCK_MIN, _next_pow2(numel)))


def _plan_launch(numels, block):
    """``(grid, walk, full)`` for a group of tensors sharing ``block``.

    ``full`` means every tensor is a whole number of blocks, so the unmasked
    kernels can run. ``walk`` is False when every block can have its own
    program, which is both the lowest-overhead form (no loop) and the fastest
    while the blocks are small.
    """
    n_blocks = sum((n + block - 1) // block for n in numels)
    full = all(n % block == 0 for n in numels)
    if n_blocks == 0:
        return 0, False, full
    per_program = max(1, _LANES_PER_PROGRAM // block)
    wanted = -(-n_blocks // per_program)
    grid = min(max(wanted, _MIN_PROGRAMS), _MAX_WALK_PROGRAMS, n_blocks)
    return grid, grid < n_blocks, full


@libentry()
@triton.jit
def adam_block_kernel(
    g_ptrs,
    p_ptrs,
    m_ptrs,
    v_ptrs,
    lr,
    bc1,
    bc2,
    anchor,
    BETA1: tl.constexpr,
    BETA2: tl.constexpr,
    EPSILON: tl.constexpr,
    WEIGHT_DECAY: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One complete Adam block, addressed by a pre-offset pointer table.

    ``*_ptrs[i]`` is the address of block ``i`` of that stream, so the kernel
    needs neither the tensor's length nor a block offset: one program per block,
    and the block is known to be complete.

    Mirrors ``multi_tensor_adam_cuda`` in DeepSpeed, with ``MATH_T = float``:
    ``m = beta1*m + (1-beta1)*g``, ``v = beta2*v + (1-beta2)*g*g``, and
    ``p -= lr * m_hat / (sqrt(v_hat) + eps)`` (plus ``decay*p`` inside the
    update for the decoupled / AdamW ``MODE == 1``, outside it -- folded into
    ``g`` -- for ``MODE == 0``).
    """
    dtype = anchor.dtype.element_ty
    i = tl.program_id(0)
    g_ptr = tl.load(g_ptrs + i).to(tl.pointer_type(dtype))
    p_ptr = tl.load(p_ptrs + i).to(tl.pointer_type(dtype))
    m_ptr = tl.load(m_ptrs + i).to(tl.pointer_type(dtype))
    v_ptr = tl.load(v_ptrs + i).to(tl.pointer_type(dtype))

    offs = tl.arange(0, BLOCK)
    g = tl.load(g_ptr + offs).to(tl.float32)
    p = tl.load(p_ptr + offs).to(tl.float32)
    m = tl.load(m_ptr + offs).to(tl.float32)
    v = tl.load(v_ptr + offs).to(tl.float32)

    if MODE == 0:
        g = g + WEIGHT_DECAY * p
        m_new = BETA1 * m + (1.0 - BETA1) * g
        v_new = BETA2 * v + (1.0 - BETA2) * g * g
        update = (m_new / bc1) / (tl.sqrt(v_new / bc2) + EPSILON)
        p_new = p - lr * update
    else:
        m_new = BETA1 * m + (1.0 - BETA1) * g
        v_new = BETA2 * v + (1.0 - BETA2) * g * g
        update = (m_new / bc1) / (tl.sqrt(v_new / bc2) + EPSILON) + WEIGHT_DECAY * p
        p_new = p - lr * update

    tl.store(p_ptr + offs, p_new.to(dtype))
    tl.store(m_ptr + offs, m_new.to(dtype))
    tl.store(v_ptr + offs, v_new.to(dtype))


@libentry()
@triton.jit
def adam_walk_kernel(
    g_ptrs,
    p_ptrs,
    m_ptrs,
    v_ptrs,
    lr,
    bc1,
    bc2,
    n_blocks,
    anchor,
    BETA1: tl.constexpr,
    BETA2: tl.constexpr,
    EPSILON: tl.constexpr,
    WEIGHT_DECAY: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``adam_block_kernel`` with each program walking several blocks.

    ``n_blocks`` stays a runtime value on purpose: a constant trip count is
    unrolled by this compiler, which multiplies the live UB buffers and
    overflows them.
    """
    dtype = anchor.dtype.element_ty
    n_programs = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    for i in range(tl.program_id(0), n_blocks, n_programs):
        g_ptr = tl.load(g_ptrs + i).to(tl.pointer_type(dtype))
        p_ptr = tl.load(p_ptrs + i).to(tl.pointer_type(dtype))
        m_ptr = tl.load(m_ptrs + i).to(tl.pointer_type(dtype))
        v_ptr = tl.load(v_ptrs + i).to(tl.pointer_type(dtype))

        g = tl.load(g_ptr + offs).to(tl.float32)
        p = tl.load(p_ptr + offs).to(tl.float32)
        m = tl.load(m_ptr + offs).to(tl.float32)
        v = tl.load(v_ptr + offs).to(tl.float32)

        if MODE == 0:
            g = g + WEIGHT_DECAY * p
            m_new = BETA1 * m + (1.0 - BETA1) * g
            v_new = BETA2 * v + (1.0 - BETA2) * g * g
            update = (m_new / bc1) / (tl.sqrt(v_new / bc2) + EPSILON)
            p_new = p - lr * update
        else:
            m_new = BETA1 * m + (1.0 - BETA1) * g
            v_new = BETA2 * v + (1.0 - BETA2) * g * g
            update = (m_new / bc1) / (tl.sqrt(v_new / bc2) + EPSILON) + WEIGHT_DECAY * p
            p_new = p - lr * update

        tl.store(p_ptr + offs, p_new.to(dtype))
        tl.store(m_ptr + offs, m_new.to(dtype))
        tl.store(v_ptr + offs, v_new.to(dtype))


@libentry()
@triton.jit
def adam_partial_kernel(
    g_ptrs,
    p_ptrs,
    m_ptrs,
    v_ptrs,
    starts,
    remains,
    lr,
    bc1,
    bc2,
    n_blocks,
    anchor,
    BETA1: tl.constexpr,
    BETA2: tl.constexpr,
    EPSILON: tl.constexpr,
    WEIGHT_DECAY: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Adam for tensors whose block grid has a partial last block.

    ``starts[i]`` is the element offset of block ``i`` inside its tensor and
    ``remains[i]`` how many of its elements are real, so the last block of each
    tensor is masked while every other block runs with a full one. Used only for
    sizes that are not a multiple of ``BLOCK``.

    The offset is a loaded value rather than folded into the pointer on purpose:
    a masked access indexed by a compile-time ``tl.arange`` alone on a
    runtime-loaded base miscompiles here.
    """
    dtype = anchor.dtype.element_ty
    n_programs = tl.num_programs(0)
    lanes = tl.arange(0, BLOCK)
    for i in range(tl.program_id(0), n_blocks, n_programs):
        mask = lanes < tl.load(remains + i)
        start = tl.multiple_of(tl.load(starts + i), BLOCK)
        offs = tl.max_contiguous(start + lanes, BLOCK)
        g_ptr = tl.load(g_ptrs + i).to(tl.pointer_type(dtype))
        p_ptr = tl.load(p_ptrs + i).to(tl.pointer_type(dtype))
        m_ptr = tl.load(m_ptrs + i).to(tl.pointer_type(dtype))
        v_ptr = tl.load(v_ptrs + i).to(tl.pointer_type(dtype))

        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        if MODE == 0:
            g = g + WEIGHT_DECAY * p
            m_new = BETA1 * m + (1.0 - BETA1) * g
            v_new = BETA2 * v + (1.0 - BETA2) * g * g
            update = (m_new / bc1) / (tl.sqrt(v_new / bc2) + EPSILON)
            p_new = p - lr * update
        else:
            m_new = BETA1 * m + (1.0 - BETA1) * g
            v_new = BETA2 * v + (1.0 - BETA2) * g * g
            update = (m_new / bc1) / (tl.sqrt(v_new / bc2) + EPSILON) + WEIGHT_DECAY * p
            p_new = p - lr * update

        tl.store(p_ptr + offs, p_new.to(dtype), mask=mask)
        tl.store(m_ptr + offs, m_new.to(dtype), mask=mask)
        tl.store(v_ptr + offs, v_new.to(dtype), mask=mask)


def _build_metadata(g_list, p_list, m_list, v_list, block, partial):
    """Per-block pointer tables, one entry per block of every tensor.

    In the common (exact) case ``g_ptrs[i]`` is the address of block ``i``
    itself, so the device side is four independent loads and no address
    arithmetic. In the partial case the tables hold the tensor base plus the
    block's element offset and remaining length, which the masked kernel needs.

    Building the tables with numpy keeps a rebuild sub-millisecond at hundreds
    of tensors; the result is cached anyway (see ``_get_metadata``).
    """
    device = p_list[0].device
    numels = np.fromiter((t.numel() for t in g_list), dtype=np.int64, count=len(g_list))
    blocks_per_tensor = (numels + block - 1) // block
    total = int(blocks_per_tensor.sum())
    owner = np.repeat(np.arange(len(numels), dtype=np.int64), blocks_per_tensor)
    first_block = np.concatenate(([0], np.cumsum(blocks_per_tensor)[:-1]))
    within = np.arange(total, dtype=np.int64) - np.repeat(
        first_block, blocks_per_tensor
    )
    starts = within * block

    # Exact tiles point at their own start; partial ones keep the base and carry
    # the offset separately for the masked kernel.
    byte_offsets = 0 if partial else starts * g_list[0].element_size()

    def tables(tensors):
        base = np.fromiter(
            (t.data_ptr() for t in tensors), dtype=np.int64, count=len(tensors)
        )
        return torch.from_numpy(base[owner] + byte_offsets).to(device)

    g_ptrs = tables(g_list)
    p_ptrs = tables(p_list)
    m_ptrs = tables(m_list)
    v_ptrs = tables(v_list)
    if not partial:
        return g_ptrs, p_ptrs, m_ptrs, v_ptrs, None, None
    starts_dev = torch.from_numpy(starts).to(device)
    remain_dev = torch.from_numpy(numels[owner] - starts).to(device)
    return g_ptrs, p_ptrs, m_ptrs, v_ptrs, starts_dev, remain_dev


# Device-meta cache keyed by the exact tensor set. A hit always yields
# bit-identical content: the key contains every address and numel the buffer is
# built from, so even tensors freed and re-created at the same addresses with
# the same shapes produce the same key AND the same meta. Training loops call
# the optimizer on a fixed tensor set every step, so after the first step every
# call is a hit and the metadata cost disappears from the hot path.
_META_CACHE = {}


def _get_metadata(g_list, p_list, m_list, v_list, block, partial):
    key = (
        block,
        partial,
        g_list[0].dtype,
        tuple(t.data_ptr() for t in g_list),
        tuple(t.data_ptr() for t in p_list),
        tuple(t.data_ptr() for t in m_list),
        tuple(t.data_ptr() for t in v_list),
        tuple(t.numel() for t in g_list),
    )
    entry = _META_CACHE.get(key)
    if entry is None:
        if len(_META_CACHE) >= 64:
            # Bound the device memory held by cached metas. Real workloads use
            # one tensor set, so an eviction here only costs one rebuild.
            cur = next(iter(_META_CACHE))
            del _META_CACHE[cur]
        entry = _build_metadata(g_list, p_list, m_list, v_list, block, partial)
        _META_CACHE[key] = entry
    return entry


def _launch_group(triples, block, n_blocks, grid, walk, full, values, consts):
    g_list, p_list, m_list, v_list = zip(*triples)
    anchor = g_list[0]
    # An 8192-lane tile only fits the UB single-buffered for this many streams.
    stage = {"num_stages": 1} if block >= _BLOCK_MAX else {}

    if full and not walk:
        g_ptrs, p_ptrs, m_ptrs, v_ptrs, _, _ = _get_metadata(
            g_list, p_list, m_list, v_list, block, False
        )
        with torch_device_fn.device(anchor.device):
            adam_block_kernel[(n_blocks,)](
                g_ptrs,
                p_ptrs,
                m_ptrs,
                v_ptrs,
                values[0],
                values[1],
                values[2],
                anchor,
                BETA1=consts[0],
                BETA2=consts[1],
                EPSILON=consts[2],
                WEIGHT_DECAY=consts[3],
                MODE=consts[4],
                BLOCK=block,
            )
        return

    partial = not full
    g_ptrs, p_ptrs, m_ptrs, v_ptrs, starts, remains = _get_metadata(
        g_list, p_list, m_list, v_list, block, partial
    )
    with torch_device_fn.device(anchor.device):
        if partial:
            adam_partial_kernel[(grid,)](
                g_ptrs,
                p_ptrs,
                m_ptrs,
                v_ptrs,
                starts,
                remains,
                values[0],
                values[1],
                values[2],
                n_blocks,
                anchor,
                BETA1=consts[0],
                BETA2=consts[1],
                EPSILON=consts[2],
                WEIGHT_DECAY=consts[3],
                MODE=consts[4],
                BLOCK=block,
                **stage,
            )
        else:
            adam_walk_kernel[(grid,)](
                g_ptrs,
                p_ptrs,
                m_ptrs,
                v_ptrs,
                values[0],
                values[1],
                values[2],
                n_blocks,
                anchor,
                BETA1=consts[0],
                BETA2=consts[1],
                EPSILON=consts[2],
                WEIGHT_DECAY=consts[3],
                MODE=consts[4],
                BLOCK=block,
                **stage,
            )


def multi_tensor_adam(
    chunk_size,
    noop_flag,
    tensor_lists,
    lr,
    beta1,
    beta2,
    epsilon,
    step,
    mode,
    bias_correction,
    weight_decay,
):
    """Batched Adam/AdamW step over ``tensor_lists = [g, p, m, v]`` (Ascend).

    The interface is the generic operator's -- see
    ``flag_train.deepspeed.multi_tensor_adam`` for the argument contract.
    ``chunk_size`` is accepted for parity and ignored (this implementation picks
    its own element-per-program granularity); ``noop_flag`` is ignored, as
    DeepSpeed's own noop check is commented out; ``step`` and ``bias_correction``
    only decide ``bc1``/``bc2`` on the host. Everything else is baked into the
    kernel as compile-time constants except ``lr``, ``bc1`` and ``bc2``; a
    caller that changes ``beta1``, ``beta2``, ``epsilon``, ``weight_decay`` or
    ``mode`` pays one recompile.
    """
    logger.debug("TRAIN_ASCEND MULTI_TENSOR_ADAM")

    g_list, p_list, m_list, v_list = tensor_lists
    if not g_list:
        return

    dtype = g_list[0].dtype
    assert (
        dtype in _SUPPORTED_DTYPES
    ), f"multi_tensor_adam only supports fp16/bf16/fp32, got {dtype}"

    if bias_correction == 1:
        bc1 = 1.0 - beta1**step
        bc2 = 1.0 - beta2**step
    else:
        bc1 = 1.0
        bc2 = 1.0

    for g, p, m, v in zip(g_list, p_list, m_list, v_list):
        assert g.is_contiguous() and p.is_contiguous()
        assert m.is_contiguous() and v.is_contiguous()
        assert g.dtype == p.dtype == m.dtype == v.dtype
        assert (
            p.device.type == runtime_device.name
        ), f"multi_tensor_adam only supports {runtime_device.name} tensors"

    # Group by dtype and by tile size, one launch per group.
    groups = {}
    itemsize = g_list[0].element_size()
    for g, p, m, v in zip(g_list, p_list, m_list, v_list):
        block = _block_for(g.numel(), itemsize)
        if block == 0:
            continue
        groups.setdefault((p.dtype, block), []).append((g, p, m, v))

    values = (float(lr), float(bc1), float(bc2))
    consts = (
        float(beta1),
        float(beta2),
        float(epsilon),
        float(weight_decay),
        int(mode),
    )
    for (_, block), triples in groups.items():
        numels = [t[0].numel() for t in triples]
        n_blocks = sum((n + block - 1) // block for n in numels)
        grid, walk, full = _plan_launch(numels, block)
        if grid == 0:
            continue
        _launch_group(triples, block, n_blocks, grid, walk, full, values, consts)
