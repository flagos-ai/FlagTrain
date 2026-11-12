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
"""Ascend LAMB: no atomics, one launch where one program will do.

The generic ``lamb_part1_kernel`` folds the partial norm reduction inside the
same launch: every program bumps an arrival counter with ``tl.atomic_add`` and
the program that completes the count reduces the partials. On Ascend that
kernel wedges the device when it is launched repeatedly back-to-back -- which
is exactly what ``do_bench_npu`` does -- and the runtime then reports a vector
core timeout (507034), taking the whole benchmark run with it.

Measured on 910B4 here at 65536 elements (64 programs): 2, 4 and 8 consecutive
launches complete, 16 and 35 never return, while the same kernel with the
atomic removed runs 35 launches without trouble. A single launch, or a launch
followed by a host sync, is always fine -- which is why an accuracy test can
look healthy while the benchmark hangs.

So this override never synchronises between programs. Part 1 keeps the generic
arithmetic and writes one pair of partial norms; a one-program
``lamb_fold_kernel`` combines them when there are enough programs to justify a
third launch, and ``lamb_apply_kernel`` folds them itself when there are not.
Below ``_FUSED_MAX`` elements one program covers the tensor, so its own sums are
the global norms and the whole step is a single launch.

The other half of the speedup is the block width. The generic ``1024`` cap is a
CUDA figure; on Ascend it leaves the vector cores with a 4 KiB tile each and the
launch is dominated by per-program cost. 8192 lanes per program is the plateau
here (16384 overflows the 192 KiB UB buffer), so ``_block_size`` steps from 2048
up to 8192 as the tensor grows.

Together, against the split-only version that first replaced the atomic fold,
``--level core`` went 4.89 -> 9.67 mean, reaching 11.8 / 11.2 / 11.4 / 13.5 /
9.9 / 5.7 / 4.3x at 1024, 4096, 16384, 65536, 262144, 1M and 11M elements.

The values above are the measured optima, not starting points, so changing one
should mean beating these numbers. Block width: part 1 and the apply kernel
have the same optimum at every size (16384 -> 2048, 65536 -> 4096, 262144 and
1M -> 8192), which is why they share a single constant. Fold threshold:
inlining wins by 12% at 8 programs and loses by 11% at 1349, so the crossover is
inside 8..1349, not above it.

What is left is the hardware. 11M moves the 40 B/element floor at 1.17-1.32
TB/s against a measured 1.19-1.6 TB/s ceiling, and 1M at 1.4 TB/s; 16K-256K are
launch- and program-throughput-bound, costing about the 1.2 us launch rather
than the ~0.4 us of traffic. Traffic is already minimal: caching ``update`` for
the apply kernel to read is exactly neutral, because the 4 B saved there is the
4 B extra part 1 must store. Measured and rejected while looking for more:
``tl.join`` to fuse the two reductions into one (150x slower, then UB overflow),
``m * (1 / (sqrt(v) + eps))`` (a wash -- the whole sqrt/divide is 4% at 262144),
a one-launch variant where every program recomputes the global norms (2.5x
slower, the tile walk is latency-bound), and baking the hyper-parameters as
``tl.constexpr`` (1-2%, not worth a recompile whenever ``decay`` or
``grad_scale`` is scheduled).
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# Master weights and moments are kept in one of these dtypes.  fp64 is only
# usable on a device that declares fp64 support; fp32 master weights may be
# driven by fp16 gradients (mixed precision).
_MASTER_DTYPES = (torch.float32, torch.float64)
_GRAD_DTYPES = {
    torch.float32: (torch.float32, torch.float16),
    torch.float64: (torch.float64,),
}

# Match the CUDA operator's 32-bit indexing bound; offsets are 32-bit here too.
_INT32_MAX = torch.iinfo(torch.int32).max

# Block width. 8192 is a hard ceiling: a part 1 tile of 16384 lanes needs 320
# KiB of UB and only 192 KiB exist, so it fails to compile at all.
_BLOCK_MAX = 8192

# Below this many elements the whole step is one program's work and the three
# launches cost more than the arithmetic -- each launch is ~1.2 us on this
# device, which is the entire budget at 1024 elements.
_FUSED_MAX = 8192

# Up to this many programs, part 3 folds the norm partials itself instead of
# waiting for the fold launch: the redundant read is 8*num_blocks bytes per
# program (512 KiB at the limit) and cheaper than a ~1.4 us launch.
_INLINE_FOLD_MAX_BLOCKS = 256


def _block_size(n):
    """Lanes per program for both part 1 and part 3.

    The measured optimum tracks the tensor size rather than sitting at one
    value. Re-measured 2026-11-12 by interleaving the whole step per width:
    2048 through 24K elements (24K 5.99 vs 6.16 for 4096), 4096 from 28K
    through 64K (28K 6.35 vs 6.52; 32K 6.48 vs 6.75; 64K 7.65 vs 8.80), and the
    8192 plateau above that (96K 9.39 vs 9.84; 128K 10.00 vs 10.81). Widening
    is not monotone below the plateau -- at 16K the 8-program split still wins
    over both 4096 (5.86) and 8192 (7.99) -- and 8192 is a hard ceiling: a
    16384-lane part 1 tile needs 320 KiB of UB and only 192 KiB exist.
    """
    if n <= 24576:
        return 2048
    if n <= 1 << 16:
        return 4096
    return _BLOCK_MAX


@libentry()
@triton.jit
def lamb_part1_kernel(
    p_ptr,
    m_ptr,
    v_ptr,
    g_ptr,
    n,
    inv_grad_scale,
    b1,
    b2,
    eps,
    decay,
    norms_ptr,
    mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Adam moment update plus this program's partial weight/update norms.

    Same arithmetic as the generic part 1 up to the point where it would fold:
    the squared norms are reduced inside the program and written as one pair of
    partial sums, which ``lamb_fold_kernel`` combines afterwards.
    """
    pid = tle.program_id(0)
    num_blocks = tle.num_programs(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Moments and update follow the master dtype; an fp16 gradient is promoted
    # before the arithmetic, so fp32 master weights keep fp32 state.
    acc_dtype = p_ptr.dtype.element_ty
    p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(acc_dtype)
    m = tl.load(m_ptr + offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + offsets, mask=mask, other=0.0)

    scaled_grad = g * inv_grad_scale
    m_new = b1 * m + (1 - b1) * scaled_grad
    v_new = b2 * v + (1 - b2) * scaled_grad * scaled_grad

    if mode == 0:
        denom = tl.sqrt(v_new + eps)
    else:
        denom = tl.sqrt(v_new) + eps

    update = m_new / denom + decay * p

    tl.store(m_ptr + offsets, m_new, mask=mask)
    tl.store(v_ptr + offsets, v_new, mask=mask)

    # Masked lanes loaded 0.0, so their squares contribute nothing to the sum.
    tl.store(norms_ptr + pid, tl.sum(p * p))
    tl.store(norms_ptr + num_blocks + pid, tl.sum(update * update))


@libentry()
@triton.jit
def lamb_fold_kernel(
    norms_ptr,
    lamb_coeff_ptr,
    num_blocks,
    min_coeff,
    max_coeff,
    REDUCE_BLOCK: tl.constexpr,
):
    """Reduce the partial norms and publish the layer trust ratio.

    One program walks the partials ``REDUCE_BLOCK`` at a time, so the tile size
    does not have to cover every program. ``tl.where`` evaluates both sides, so
    the denominator is kept finite even when ``u_norm`` is zero.
    """
    red_offsets = tl.arange(0, REDUCE_BLOCK)
    w_partial = tl.zeros((REDUCE_BLOCK,), dtype=norms_ptr.dtype.element_ty)
    u_partial = tl.zeros((REDUCE_BLOCK,), dtype=norms_ptr.dtype.element_ty)

    for start in range(0, num_blocks, REDUCE_BLOCK):
        idx = start + red_offsets
        in_range = idx < num_blocks
        w_partial += tl.load(norms_ptr + idx, mask=in_range, other=0.0)
        u_partial += tl.load(norms_ptr + num_blocks + idx, mask=in_range, other=0.0)

    w_norm = tl.sqrt(tl.sum(w_partial))
    u_norm = tl.sqrt(tl.sum(u_partial))

    u_denom = tl.where(u_norm == 0.0, 1.0, u_norm)
    trust = w_norm / u_denom
    # DeepSpeed clamps max first and min second, so min_coeff wins an inverted
    # [min_coeff, max_coeff] range.
    trust = tl.maximum(tl.minimum(trust, max_coeff), min_coeff)
    tl.store(
        lamb_coeff_ptr,
        tl.where((w_norm == 0.0) | (u_norm == 0.0), 1.0, trust),
    )


@libentry()
@triton.jit
def lamb_apply_kernel(
    p_ptr,
    p_copy_ptr,
    m_ptr,
    v_ptr,
    n,
    eps,
    step_size,
    decay,
    lamb_coeff_ptr,
    norms_ptr,
    num_blocks,
    min_coeff,
    max_coeff,
    mode: tl.constexpr,
    HAS_P_COPY: tl.constexpr,
    INLINE_FOLD: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply the parameter update scaled by the layer trust ratio.

    Same arithmetic as the generic ``lamb_part3_kernel``. The difference is
    ``INLINE_FOLD``: instead of reading a ratio published by the fold launch,
    every program reduces the ``norms`` partials itself. That is only worth it
    while the redundant read stays under a launch's worth of time -- at 256
    programs each one re-reads 2 KiB, and the 0.5 MiB of extra traffic costs
    ~0.4 us against the ~1.4 us the launch costs. Above that the fold kernel runs
    as its own launch again.
    """
    if INLINE_FOLD:
        red_offsets = tl.arange(0, REDUCE_BLOCK)
        w_partial = tl.zeros((REDUCE_BLOCK,), dtype=norms_ptr.dtype.element_ty)
        u_partial = tl.zeros((REDUCE_BLOCK,), dtype=norms_ptr.dtype.element_ty)
        for start in range(0, num_blocks, REDUCE_BLOCK):
            idx = start + red_offsets
            in_range = idx < num_blocks
            w_partial += tl.load(norms_ptr + idx, mask=in_range, other=0.0)
            u_partial += tl.load(norms_ptr + num_blocks + idx, mask=in_range, other=0.0)
        w_norm = tl.sqrt(tl.sum(w_partial))
        u_norm = tl.sqrt(tl.sum(u_partial))
        u_denom = tl.where(u_norm == 0.0, 1.0, u_norm)
        trust = tl.maximum(tl.minimum(w_norm / u_denom, max_coeff), min_coeff)
        coeff = tl.where((w_norm == 0.0) | (u_norm == 0.0), 1.0, trust)
        # Every program holds the same ratio; publishing it from one of them
        # keeps the returned value exactly the one the update used.
        if tle.program_id(0) == 0:
            tl.store(lamb_coeff_ptr, coeff)
    else:
        coeff = tl.load(lamb_coeff_ptr)

    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(m_ptr + offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + offsets, mask=mask, other=0.0)

    if mode == 0:
        denom = tl.sqrt(v + eps)
    else:
        denom = tl.sqrt(v) + eps

    update = m / denom + decay * p
    p_new = p - step_size * coeff * update

    tl.store(p_ptr + offsets, p_new, mask=mask)
    if HAS_P_COPY:
        tl.store(p_copy_ptr + offsets, p_new.to(p_copy_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def lamb_fused_kernel(
    p_ptr,
    p_copy_ptr,
    m_ptr,
    v_ptr,
    g_ptr,
    n,
    inv_grad_scale,
    b1,
    b2,
    eps,
    decay,
    min_coeff,
    max_coeff,
    step_size,
    lamb_coeff_ptr,
    mode: tl.constexpr,
    HAS_P_COPY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """The whole step as a single program, for tensors of one block or less.

    One program covers every element, so the sums it reduces *are* the global
    norms: the trust ratio needs no cross-block fold, and the update is applied
    straight from the registers that computed it. That removes two of the three
    launches, which is what dominates at these sizes.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    acc_dtype = p_ptr.dtype.element_ty
    p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(acc_dtype)
    m = tl.load(m_ptr + offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + offsets, mask=mask, other=0.0)

    scaled_grad = g * inv_grad_scale
    m_new = b1 * m + (1 - b1) * scaled_grad
    v_new = b2 * v + (1 - b2) * scaled_grad * scaled_grad

    if mode == 0:
        denom = tl.sqrt(v_new + eps)
    else:
        denom = tl.sqrt(v_new) + eps

    update = m_new / denom + decay * p

    tl.store(m_ptr + offsets, m_new, mask=mask)
    tl.store(v_ptr + offsets, v_new, mask=mask)

    w_norm = tl.sqrt(tl.sum(p * p))
    u_norm = tl.sqrt(tl.sum(update * update))
    u_denom = tl.where(u_norm == 0.0, 1.0, u_norm)
    trust = tl.maximum(tl.minimum(w_norm / u_denom, max_coeff), min_coeff)
    coeff = tl.where((w_norm == 0.0) | (u_norm == 0.0), 1.0, trust)
    tl.store(lamb_coeff_ptr, coeff)

    p_new = p - step_size * coeff * update
    tl.store(p_ptr + offsets, p_new, mask=mask)
    if HAS_P_COPY:
        tl.store(p_copy_ptr + offsets, p_new.to(p_copy_ptr.dtype.element_ty), mask=mask)


def lamb(
    p,
    p_copy,
    m,
    v,
    g,
    lr,
    beta1,
    beta2,
    max_coeff,
    min_coeff,
    eps,
    grad_scale,
    step,
    mode,
    bias_correction,
    decay,
):
    """Fused LAMB optimizer step, Ascend variant.

    Same contract and results as ``flag_train.deepspeed.lamb``; see that module
    for the argument documentation.
    """
    logger.debug("TRAIN_ASCEND LAMB")

    assert (
        p.dtype in _MASTER_DTYPES
    ), f"lamb expects float32 or float64 parameters, got {p.dtype}"
    assert (
        m.dtype == p.dtype and v.dtype == p.dtype
    ), "lamb requires the moment tensors to have the parameter dtype"
    assert g.dtype in _GRAD_DTYPES[p.dtype], (
        f"lamb with {p.dtype} parameters expects a "
        f"{' or '.join(str(dtype) for dtype in _GRAD_DTYPES[p.dtype])} gradient, "
        f"got {g.dtype}"
    )
    assert (
        p.dtype != torch.float64 or runtime_device.support_fp64
    ), "lamb: float64 parameters require a device with fp64 support"
    assert (
        p.device.type == runtime_device.name
    ), f"lamb only supports {runtime_device.name} tensors"

    n = p.numel()
    assert m.numel() == n and v.numel() == n and g.numel() == n
    assert p_copy.numel() == 0 or p_copy.numel() == n
    assert n > 0, "lamb requires at least one element"
    assert n <= _INT32_MAX, "lamb tensor is too large to index with int32"
    assert (
        m.device == p.device and v.device == p.device and g.device == p.device
    ), "lamb requires p, m, v and g on the same device"
    assert (
        p_copy.numel() == 0 or p_copy.dtype.is_floating_point
    ), "lamb p_copy must be a floating-point tensor"
    # ``p_copy`` is dereferenced as a raw device pointer, so a host-side copy of
    # the weights (an offload buffer, say) would fault the device rather than
    # raise. Catch it while it is still cheap.
    assert (
        p_copy.numel() == 0 or p_copy.device == p.device
    ), "lamb requires p_copy on the same device as p"

    # The kernels address every buffer as flat memory, so a strided view would
    # silently step through the wrong elements rather than fail. Reject it here
    # instead of returning wrong weights. (The generic kernel makes the same
    # assumption without checking, so this is deliberately the stricter of the
    # two.) A one-element placeholder is always contiguous, so the empty
    # ``p_copy`` case needs no special handling.
    assert (
        p.is_contiguous()
        and m.is_contiguous()
        and v.is_contiguous()
        and g.is_contiguous()
        and p_copy.is_contiguous()
    ), "lamb requires contiguous tensors"

    # ``g / grad_scale`` is a per-lane vector divide against a runtime scalar
    # that costs several instructions per lane, where the equivalent multiply
    # costs one. Invert the scalar once on the host instead: for the
    # power-of-two scales used in practice the reciprocal is exact, so the
    # result is bit-identical, and elsewhere the change is one rounding. Zero
    # keeps the divide's inf/nan behaviour rather than raising.
    inv_grad_scale = 1.0 / grad_scale if grad_scale else math.inf

    if bias_correction == 1:
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
    else:
        step_size = lr

    # The trust ratio follows the master dtype, as in the generic implementation.
    lamb_coeff_val = torch.empty((1,), dtype=p.dtype, device=p.device)

    # Match DeepSpeed: only the fp16-gradient path writes p_copy; fp32/fp64
    # branches pass NULL even when a non-empty buffer was supplied.
    has_p_copy = p_copy.numel() > 0 and g.dtype == torch.float16
    p_copy_in = p_copy if has_p_copy else p

    if n <= _FUSED_MAX:
        # One program holds the whole tensor, so the norms it reduces are the
        # global ones and no fold is needed. The block is the tensor's own
        # width: padding it wastes whole vector instructions, and the measured
        # cost is monotone in the padding (1024 elements: 2.0 us at 1024 lanes,
        # 2.4 at 2048, 3.0 at 4096).
        fused_block = max(triton.next_power_of_2(n), 1024)
        with torch_device_fn.device(p.device):
            lamb_fused_kernel[(1,)](
                p,
                p_copy_in,
                m,
                v,
                g,
                n,
                inv_grad_scale,
                beta1,
                beta2,
                eps,
                decay,
                min_coeff,
                max_coeff,
                step_size,
                lamb_coeff_val,
                mode=mode,
                HAS_P_COPY=has_p_copy,
                BLOCK_SIZE=fused_block,
            )
        return lamb_coeff_val

    BLOCK_SIZE = _block_size(n)
    num_blocks = triton.cdiv(n, BLOCK_SIZE)

    norms = torch.empty((2 * num_blocks,), dtype=p.dtype, device=p.device)
    # The inline fold reduces a vector this wide in every program, so its width
    # is paid per program: at 16K elements (8 blocks) an 8-lane reduction costs
    # 2.67 us against 2.79 for 32 lanes. The fold kernel is never affected -- it
    # only runs above 256 blocks, where this is already 256 or more.
    reduce_block = min(max(triton.next_power_of_2(num_blocks), 8), 2048)
    inline_fold = num_blocks <= _INLINE_FOLD_MAX_BLOCKS

    with torch_device_fn.device(p.device):
        lamb_part1_kernel[(num_blocks,)](
            p,
            m,
            v,
            g,
            n,
            inv_grad_scale,
            beta1,
            beta2,
            eps,
            decay,
            norms,
            mode=mode,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        if not inline_fold:
            lamb_fold_kernel[(1,)](
                norms,
                lamb_coeff_val,
                num_blocks,
                min_coeff,
                max_coeff,
                REDUCE_BLOCK=reduce_block,
            )

        lamb_apply_kernel[(num_blocks,)](
            p,
            p_copy_in,
            m,
            v,
            n,
            eps,
            step_size,
            decay,
            lamb_coeff_val,
            norms,
            num_blocks,
            min_coeff,
            max_coeff,
            mode=mode,
            HAS_P_COPY=has_p_copy,
            INLINE_FOLD=inline_fold,
            # Dead on the fold path, but a constexpr is still part of the
            # specialization key: pin it there so that reaching a new tensor
            # size does not compile another variant of this kernel mid-run.
            REDUCE_BLOCK=reduce_block if inline_fold else 32,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return lamb_coeff_val
