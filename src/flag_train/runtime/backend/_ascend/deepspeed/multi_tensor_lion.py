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
"""Ascend multi_tensor_lion.

The generic implementation is ``flag_train.deepspeed.multi_tensor_lion`` and is
what every backend without an override runs. Its launch shape is a CUDA one --
one 2048-lane masked program per 2048 elements, one launch for the whole tensor
set -- and on this backend (910B4, CANI 9.0 / BiShengHIR, card 6,
``do_bench_npu`` KERNEL mode) that shape loses most of the machine:

* A program's *fixed* cost dominates -- tens of nanoseconds whatever it carries
  -- and it is lanes-per-program, not run length and not bytes, that sets the
  achievable bandwidth, plateauing at 8192 lanes. So small tensors are priced
  by their program count (1024 tensors x 1024 elements: 76 us for 20 MB of
  traffic) and large ones never reach the memory path (8 x 1M elements: 513 us
  where the same five streams copied take 101 us).
* Every runtime scalar argument costs throughput (~40% for the second one on
  this backend), and the generic kernel takes six.

Measured against ``--base-data`` (mean ``vs Base`` over the seven benchmark
rows, before -> after): 5.7 -> 13.0, and every row improved; the three worst,
all large tensors, went 0.44 -> 2.70 (8 x 1M), 0.90 -> 1.83 (64 x 1M) and
1.47 -> 1.80 (256 x 1M) in fp32, with fp16/bf16 tracking it.

What this module does instead:

* **8192-lane tiles.** The tile is the largest power of two dividing the
  tensor's size (capped at 8192 lanes and at 32 KB per program), so its block
  grid is exact and the kernel runs *unmasked*. Parameter sizes are multiples of
  a hidden dimension, so that is the common case rather than a special one.
  8192 is also the largest fp32 tile the 192 KB UB holds: a 16384-lane fp32
  tile fails to compile ("requires 2360064 bits while 1572864 bits available").
* **Pre-offset pointer tables.** ``g_ptrs[i]`` is the address of block ``i``, so
  the device side is three independent loads and no address arithmetic, and the
  kernel keeps a single runtime scalar (``lr``). The betas and the weight decay
  are ``tl.constexpr`` at the measured price of extra runtime scalars; they are
  fixed for a training run, so a caller that changes one pays one recompile.
* **A persistent walk.** When a launch would need many programs, each program
  walks several blocks: grid = ``clamp(cdiv(blocks, 32768/block), 32, 2048)``,
  i.e. ~32768 lanes of work per program. Measured optima: 32 programs for 64 to
  1024 tiny tensors, 256 for 8 x 1M, 2048 for 64-256 x 1M. That is what makes a
  thousand tiny tensors (24 us where the generic shape took 76 us) and the
  multi-hundred-MB cases pay the per-program cost once per program.
* **A masked path for odd sizes**, because a size that is not a whole number of
  tiles cannot run the exact-grid kernel. It keeps the same walking shape. Note
  the offset in its index is a loaded value rather than folded into the pointer:
  a masked access indexed by a compile-time ``tl.arange`` alone on a
  runtime-loaded base miscompiles here, while the same access with a runtime
  term in the index compiles and is bit-correct.

Where the rows ended up: 8 x 1M fp32 at 84-101 us is the L2 bandwidth (the
working set is 160 MB against a 168 MB L2) and matches a five-stream copy
oracle; 64 x 1M at 1.18 ms and 256 x 1M at 4.74 ms are 1.08 TB/s, the HBM wall;
1024 x 1024 at 24 us is flat across 32-256 programs. Dead ends, measured: 4096
and 16384-lane tiles, walks with fewer than 128 programs for large blocks,
masking the exact path (2-10% worse on the tiny-tensor rows), and ``num_stages``
below 8192 (a no-op).

Registration is by name. ``_ascend/deepspeed/__init__.py`` re-exports this
module's ``multi_tensor_lion``, and ``SpecOpRegistrar`` writes it over the
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
# 192 KB UB holds. ``_BLOCK_BYTES`` bounds the same tile by element size, so a
# wider element (fp64) does not overrun the UB at that lane count.
_BLOCK_MAX = 8192
_BLOCK_BYTES = 32768
# The smallest tile that earns a block per program. A tensor below it is handled
# by the masked kernel at the smallest power of two covering it, which bounds
# the wasted lanes of its last block.
_BLOCK_MIN = 512

# Walk sizing: ~32768 lanes of work per program (about 0.6 MB of traffic for the
# five fp32 streams), at least 32 programs so every vector core has work, and
# never more than 2048 -- past that the per-program cost outweighs the walk, and
# this backend rejects a grid product above 65535 anyway.
_LANES_PER_PROGRAM = 32768
_MIN_PROGRAMS = 32
_MAX_WALK_PROGRAMS = 2048


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
    program, which is both the lowest-overhead form (one runtime scalar, no
    loop) and the fastest while the blocks are small.
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
def lion_block_kernel(
    g_ptrs,
    p_ptrs,
    m_ptrs,
    lr,
    anchor,
    BETA1: tl.constexpr,
    BETA2: tl.constexpr,
    WEIGHT_DECAY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One complete Lion block, addressed by a pre-offset pointer table.

    ``g_ptrs``/``p_ptrs``/``m_ptrs`` hold the three base addresses *of this
    block* for every block of every tensor in the launch, so the kernel needs
    neither the tensor's length nor a block offset: one program per block, and
    the block is known to be complete.

    Mirrors ``multi_tensor_lion_cuda`` in DeepSpeed: fp32 intermediate math,
    ``c = beta1*m + (1-beta1)*g``, ``update = c > 0 ? -lr : +lr`` (c == 0 takes
    +lr), ``p = p*(1 - lr*weight_decay) + update``, ``m = beta2*m + (1-beta2)*g``.
    ``m`` is stored before ``p`` so the compiler can retire the moment's buffer
    before the parameter arithmetic starts.
    """
    dtype = anchor.dtype.element_ty
    i = tl.program_id(0)
    g_ptr = tl.load(g_ptrs + i).to(tl.pointer_type(dtype))
    p_ptr = tl.load(p_ptrs + i).to(tl.pointer_type(dtype))
    m_ptr = tl.load(m_ptrs + i).to(tl.pointer_type(dtype))

    offs = tl.arange(0, BLOCK)
    g = tl.load(g_ptr + offs).to(tl.float32)
    p = tl.load(p_ptr + offs).to(tl.float32)
    m = tl.load(m_ptr + offs).to(tl.float32)

    cc = BETA1 * m + (1.0 - BETA1) * g
    m_new = BETA2 * m + (1.0 - BETA2) * g
    tl.store(m_ptr + offs, m_new.to(dtype))

    update = tl.where(cc > 0, -lr, lr)
    tl.store(p_ptr + offs, (p * (1.0 - lr * WEIGHT_DECAY) + update).to(dtype))


@libentry()
@triton.jit
def lion_walk_kernel(
    g_ptrs,
    p_ptrs,
    m_ptrs,
    lr,
    n_blocks,
    anchor,
    BETA1: tl.constexpr,
    BETA2: tl.constexpr,
    WEIGHT_DECAY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``lion_block_kernel`` with each program walking several blocks.

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

        g = tl.load(g_ptr + offs).to(tl.float32)
        p = tl.load(p_ptr + offs).to(tl.float32)
        m = tl.load(m_ptr + offs).to(tl.float32)

        cc = BETA1 * m + (1.0 - BETA1) * g
        m_new = BETA2 * m + (1.0 - BETA2) * g
        tl.store(m_ptr + offs, m_new.to(dtype))

        update = tl.where(cc > 0, -lr, lr)
        tl.store(p_ptr + offs, (p * (1.0 - lr * WEIGHT_DECAY) + update).to(dtype))


@libentry()
@triton.jit
def lion_partial_kernel(
    g_ptrs,
    p_ptrs,
    m_ptrs,
    starts,
    remains,
    lr,
    n_blocks,
    anchor,
    BETA1: tl.constexpr,
    BETA2: tl.constexpr,
    WEIGHT_DECAY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Lion for tensors whose block grid has a partial last block.

    ``starts[i]`` is the element offset of block ``i`` inside its tensor and
    ``remains[i]`` how many of its elements are real, so the last block of each
    tensor is masked while every other block runs with a full one. Used only for
    sizes that are not a multiple of ``BLOCK``.

    The offset is a loaded value rather than folded into the pointer on purpose
    (see the module docstring): a masked access indexed by a compile-time
    ``tl.arange`` alone on a runtime-loaded base miscompiles here.
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

        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        cc = BETA1 * m + (1.0 - BETA1) * g
        m_new = BETA2 * m + (1.0 - BETA2) * g
        tl.store(m_ptr + offs, m_new.to(dtype), mask=mask)

        update = tl.where(cc > 0, -lr, lr)
        tl.store(
            p_ptr + offs, (p * (1.0 - lr * WEIGHT_DECAY) + update).to(dtype), mask=mask
        )


def _build_metadata(g_list, p_list, m_list, block, partial):
    """Per-block pointer tables, one entry per block of every tensor.

    In the common (exact) case ``g_ptrs[i]`` is the address of block ``i``
    itself, so the device side is three independent loads and no address
    arithmetic. In the partial case the tables hold the tensor base plus the
    block's element offset and remaining length, which the masked kernel needs.

    Building the tables with numpy keeps a rebuild sub-millisecond at 1024
    tensors; the result is cached anyway (see ``_get_metadata``).
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
    if not partial:
        return g_ptrs, p_ptrs, m_ptrs, None, None
    starts_dev = torch.from_numpy(starts).to(device)
    remain_dev = torch.from_numpy(numels[owner] - starts).to(device)
    return g_ptrs, p_ptrs, m_ptrs, starts_dev, remain_dev


# Device-meta cache keyed by the exact tensor set. A hit always yields
# bit-identical content: the key contains every address and numel the buffer is
# built from, so even tensors freed and re-created at the same addresses with
# the same shapes produce the same key AND the same meta. Training loops call
# the optimizer on a fixed tensor set every step, so after the first step every
# call is a hit and the metadata cost disappears from the hot path.
_META_CACHE = {}


def _get_metadata(g_list, p_list, m_list, block, partial):
    key = (
        block,
        partial,
        g_list[0].dtype,
        tuple(t.data_ptr() for t in g_list),
        tuple(t.data_ptr() for t in p_list),
        tuple(t.data_ptr() for t in m_list),
        tuple(t.numel() for t in g_list),
    )
    entry = _META_CACHE.get(key)
    if entry is None:
        if len(_META_CACHE) >= 64:
            # Bound the device memory held by cached metas. Real workloads use
            # one tensor set, so an eviction here only costs one rebuild.
            cur = next(iter(_META_CACHE))
            del _META_CACHE[cur]
        entry = _build_metadata(g_list, p_list, m_list, block, partial)
        _META_CACHE[key] = entry
    return entry


def _launch_group(triples, block, n_blocks, grid, walk, full, lr, params):
    g_list = [t[0] for t in triples]
    p_list = [t[1] for t in triples]
    m_list = [t[2] for t in triples]
    anchor = g_list[0]
    # An 8192-lane walking tile only fits the UB single-buffered.
    stage = {"num_stages": 1} if block >= _BLOCK_MAX else {}

    if full and not walk:
        g_ptrs, p_ptrs, m_ptrs, _, _ = _get_metadata(
            g_list, p_list, m_list, block, False
        )
        with torch_device_fn.device(anchor.device):
            lion_block_kernel[(n_blocks,)](
                g_ptrs,
                p_ptrs,
                m_ptrs,
                lr,
                anchor,
                BETA1=params[0],
                BETA2=params[1],
                WEIGHT_DECAY=params[2],
                BLOCK=block,
            )
        return

    partial = not full
    g_ptrs, p_ptrs, m_ptrs, starts, remains = _get_metadata(
        g_list, p_list, m_list, block, partial
    )
    with torch_device_fn.device(anchor.device):
        if partial:
            lion_partial_kernel[(grid,)](
                g_ptrs,
                p_ptrs,
                m_ptrs,
                starts,
                remains,
                lr,
                n_blocks,
                anchor,
                BETA1=params[0],
                BETA2=params[1],
                WEIGHT_DECAY=params[2],
                BLOCK=block,
                **stage,
            )
        else:
            lion_walk_kernel[(grid,)](
                g_ptrs,
                p_ptrs,
                m_ptrs,
                lr,
                n_blocks,
                anchor,
                BETA1=params[0],
                BETA2=params[1],
                WEIGHT_DECAY=params[2],
                BLOCK=block,
                **stage,
            )


def multi_tensor_lion(
    chunk_size,
    noop_flag,
    tensor_lists,
    lr,
    beta1,
    beta2,
    step,
    weight_decay,
):
    """Fused Lion optimizer step over a list of parameter tensors (Ascend).

    Faithful port of DeepSpeed's ``multi_tensor_lion`` CUDA operator. For each
    tensor, with gradient ``g``, parameter ``p`` and first moment ``m``::

        c      = beta1 * m + (1 - beta1) * g
        update = -lr if c > 0 else +lr      # sign update; c == 0 takes +lr
        p      = p * (1 - lr * weight_decay) + update
        m      = beta2 * m + (1 - beta2) * g

    Meanwhile every block of every tensor of one dtype and tile size is fused
    into a single kernel launch, addressed through a device-side table of
    per-block pointers that is cached across calls (see ``_get_metadata``).

    The interface is the generic operator's -- see
    ``flag_train.deepspeed.multi_tensor_lion`` for the argument contract.
    ``chunk_size`` is accepted for parity and ignored (this implementation picks
    its own element-per-program granularity); ``step`` is unused because Lion
    has no bias correction; ``noop_flag`` is ignored, as DeepSpeed's own noop
    check is commented out. ``beta1``, ``beta2`` and ``weight_decay`` are baked
    into the kernel as compile-time constants, so a caller that changes one pays
    one recompile; ``lr`` stays a runtime value and may change freely.
    """
    logger.debug("TRAIN_ASCEND MULTI_TENSOR_LION")

    g_list, p_list, m_list = tensor_lists
    assert len(g_list) == len(p_list) == len(m_list) and len(g_list) > 0
    for g, p, m in zip(g_list, p_list, m_list):
        assert g.is_contiguous() and p.is_contiguous() and m.is_contiguous()
        assert g.dtype == p.dtype == m.dtype
        assert (
            p.device.type == runtime_device.name
        ), f"multi_tensor_lion only supports {runtime_device.name} tensors"

    # Group by dtype and by tile size, one launch per group (mirrors
    # FusedLion.step, which groups by dtype alone).
    groups = {}
    itemsize = g_list[0].element_size()
    for g, p, m in zip(g_list, p_list, m_list):
        block = _block_for(g.numel(), itemsize)
        if block == 0:
            continue
        groups.setdefault((p.dtype, block), []).append((g, p, m))

    params = (float(beta1), float(beta2), float(weight_decay))
    lr = float(lr)
    for (_, block), triples in groups.items():
        numels = [t[0].numel() for t in triples]
        n_blocks = sum((n + block - 1) // block for n in numels)
        grid, walk, full = _plan_launch(numels, block)
        if grid == 0:
            continue
        _launch_group(triples, block, n_blocks, grid, walk, full, lr, params)
