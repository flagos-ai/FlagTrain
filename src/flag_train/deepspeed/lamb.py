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

# Master weights and moments are kept in one of these dtypes.  A float64 master
# needs a device with fp64 hardware; a float32 master may be driven by either
# float32 or float16 gradients (the latter is the mixed-precision path).
_MASTER_DTYPES = (torch.float32, torch.float64)
_GRAD_DTYPES = {
    torch.float32: (torch.float32, torch.float16),
    torch.float64: (torch.float64,),
}

# DeepSpeed launches at most 512 blocks and lets each thread grid-stride over
# the rest of the tensor.  Match that launch bound here.  fp64 tiles are tuned
# separately: a double-precision tile carries twice the register/UB footprint,
# so it wants a smaller block and a wider grid than fp32.
_MAX_BLOCKS = 512
_FP64_NUM_WARPS = 8
_FP64_MAX_BLOCKS = 512
# Below this many elements one program covers the tensor and the whole step is
# a single launch; above it the split path amortizes the cross-block fold.
# Measured on H20: fusing wins at 1024 and loses from 2048 on, where a 2K+ fp64
# tile spills registers.
_FUSED_MAX = 1024
_INT32_MAX = torch.iinfo(torch.int32).max


def _fp64_launch_config(n):
    """(BLOCK_SIZE, num_warps) for the split fp64 path, measured on H20.

    fp64 tiles are register-bound: a 1024-element tile spills once the moments,
    update and norm accumulator are live together.  Small tensors therefore do
    best with narrow tiles (more independent programs), while at 1M+ elements a
    wider tile amortizes the grid-stride loop.  The breakpoints are the measured
    crossovers, not guesses.
    """
    if n <= 4096:
        return 128, 8
    if n <= 16384:
        return 256, 4
    if n <= 32768:
        return 512, 8
    if n <= 65536:
        return 1024, 8
    if n <= 1 << 20:
        return 512, 8
    if n <= 1 << 22:
        return 512, 4
    return 1024, 4


@libentry()
@triton.jit
def lamb_part1_kernel(
    p_ptr,
    m_ptr,
    v_ptr,
    g_ptr,
    n,
    grad_scale,
    b1,
    b2,
    eps,
    decay,
    max_coeff,
    min_coeff,
    norms_ptr,
    counter_ptr,
    lamb_coeff_ptr,
    mode: tl.constexpr,
    IS_FP64: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Adam moment update, partial norm reduction and the trust ratio.

    Mirrors ``lamb_cuda_kernel_part1`` and ``lamb_cuda_kernel_part2`` in
    DeepSpeed: update the first/second moments, build the Adam update vector
    ``update = m/denom + decay*w``, reduce the squared weight/update norms
    inside this program, and emit one pair of partial sums per program.

    DeepSpeed then launches a second kernel to fold those partials, because no
    program can see another's result without one.  Here the arrival counter
    supplies that ordering inside this kernel instead, which removes a launch
    from the critical path of a step this small.  The program that completes the
    count folds the partials, publishes the trust ratio, and resets the counter
    so the scratch is ready for the next step.

    ``acq_rel`` on the arrival makes each program's partial stores visible to
    the one that completes the count, and that program's trust ratio visible to
    the part 3 launch that follows on the stream.
    """
    pid = tle.program_id(0)
    num_blocks = tle.num_programs(0)
    lane = tl.arange(0, BLOCK_SIZE)

    # The moments and the update follow the master-weight dtype.  Converting the
    # gradient here is what makes fp16 gradients against fp32 master weights
    # work: the load is fp16, the arithmetic and the stored moments are fp32.
    acc_dtype = p_ptr.dtype.element_ty
    # Each program may own several blocks: the launch caps the grid at
    # MAX_BLOCKS (as DeepSpeed caps it at 512) and grid-strides over the rest.
    # Accumulate per-lane products and reduce once, so a program that walks many
    # blocks pays for one reduction rather than one per block.
    w_acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)
    u_acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)

    for start in range(pid * BLOCK_SIZE, n, num_blocks * BLOCK_SIZE):
        offsets = start + lane
        mask = offsets < n

        p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
        g = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(acc_dtype)
        m = tl.load(m_ptr + offsets, mask=mask, other=0.0)
        v = tl.load(v_ptr + offsets, mask=mask, other=0.0)

        scaled_grad = g / grad_scale
        m_new = b1 * m + (1 - b1) * scaled_grad
        v_new = b2 * v + (1 - b2) * scaled_grad * scaled_grad

        if IS_FP64:
            # DeepSpeed's double path computes the denominator with sqrtf and a
            # float eps even though p/m/v are double.  Keep the tensors, moments
            # and norms in fp64 but mirror that scalar precision for speed.
            if mode == 0:
                denom = tl.sqrt((v_new + eps).to(tl.float32)).to(acc_dtype)
            else:
                denom = (tl.sqrt(v_new.to(tl.float32)) + eps).to(acc_dtype)
        elif mode == 0:
            denom = tl.sqrt(v_new + eps)
        else:
            denom = tl.sqrt(v_new) + eps

        update = m_new / denom + decay * p

        tl.store(m_ptr + offsets, m_new, mask=mask)
        tl.store(v_ptr + offsets, v_new, mask=mask)

        # Masked lanes loaded 0.0, so their squares contribute nothing.
        w_acc += p * p
        u_acc += update * update

    tl.store(norms_ptr + pid, tl.sum(w_acc))
    tl.store(norms_ptr + num_blocks + pid, tl.sum(u_acc))

    if tl.atomic_add(counter_ptr, 1, sem="acq_rel") == num_blocks - 1:
        red_offsets = tl.arange(0, REDUCE_BLOCK)
        w_partial = tl.zeros((REDUCE_BLOCK,), dtype=acc_dtype)
        u_partial = tl.zeros((REDUCE_BLOCK,), dtype=acc_dtype)

        for start in range(0, num_blocks, REDUCE_BLOCK):
            idx = start + red_offsets
            in_range = idx < num_blocks
            w_partial += tl.load(norms_ptr + idx, mask=in_range, other=0.0)
            u_partial += tl.load(norms_ptr + num_blocks + idx, mask=in_range, other=0.0)

        w_norm = tl.sqrt(tl.sum(w_partial))
        u_norm = tl.sqrt(tl.sum(u_partial))
        # Guard the division so a zero norm yields coeff == 1.0 (DeepSpeed leaves
        # it unclamped in that case); tl.where evaluates both sides, so keep the
        # denominator finite even when u_norm == 0.
        u_denom = tl.where(u_norm == 0.0, 1.0, u_norm)
        trust = w_norm / u_denom
        # DeepSpeed clamps against max_coeff first and min_coeff second, so an
        # inverted [min_coeff, max_coeff] range resolves to min_coeff.
        trust = tl.maximum(tl.minimum(trust, max_coeff), min_coeff)
        tl.store(
            lamb_coeff_ptr,
            tl.where((w_norm == 0.0) | (u_norm == 0.0), 1.0, trust),
        )
        tl.store(counter_ptr, 0)


@libentry()
@triton.jit
def lamb_part3_kernel(
    p_ptr,
    p_copy_ptr,
    m_ptr,
    v_ptr,
    n,
    eps,
    step_size,
    decay,
    lamb_coeff_ptr,
    mode: tl.constexpr,
    IS_FP64: tl.constexpr,
    HAS_P_COPY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply the parameter update scaled by the layer-wise trust ratio.

    Mirrors ``lamb_cuda_kernel_part3`` in DeepSpeed: ``trust = ||w|| / ||u||``
    (1.0 when either norm is zero), clamped to ``[min_coeff, max_coeff]``, then
    ``p = p - step_size * trust * update`` with ``update`` recomputed from the
    already-updated moments.  The ratio itself is already folded over every
    program by part 1, so this kernel only reads it.
    """
    lamb_coeff = tl.load(lamb_coeff_ptr)

    pid = tle.program_id(0)
    num_blocks = tle.num_programs(0)
    lane = tl.arange(0, BLOCK_SIZE)
    acc_dtype = p_ptr.dtype.element_ty

    for start in range(pid * BLOCK_SIZE, n, num_blocks * BLOCK_SIZE):
        offsets = start + lane
        mask = offsets < n

        p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
        m = tl.load(m_ptr + offsets, mask=mask, other=0.0)
        v = tl.load(v_ptr + offsets, mask=mask, other=0.0)

        if IS_FP64:
            # Match DeepSpeed's double path: sqrt/eps are float, the update is
            # still evaluated in the parameter dtype.
            if mode == 0:
                denom = tl.sqrt((v + eps).to(tl.float32)).to(acc_dtype)
            else:
                denom = (tl.sqrt(v.to(tl.float32)) + eps).to(acc_dtype)
        elif mode == 0:
            denom = tl.sqrt(v + eps)
        else:
            denom = tl.sqrt(v) + eps

        update = m / denom + decay * p
        p_new = p - step_size * lamb_coeff * update

        tl.store(p_ptr + offsets, p_new, mask=mask)
        if HAS_P_COPY:
            tl.store(
                p_copy_ptr + offsets,
                p_new.to(p_copy_ptr.dtype.element_ty),
                mask=mask,
            )


@libentry()
@triton.jit
def lamb_fused_kernel(
    p_ptr,
    p_copy_ptr,
    m_ptr,
    v_ptr,
    g_ptr,
    n,
    grad_scale,
    b1,
    b2,
    eps,
    decay,
    min_coeff,
    max_coeff,
    step_size,
    lamb_coeff_ptr,
    mode: tl.constexpr,
    IS_FP64: tl.constexpr,
    HAS_P_COPY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """The whole step in one program, for tensors no larger than one tile.

    One program covers every element, so the norms it reduces are already the
    global norms: the trust ratio needs neither a partial-norm workspace nor the
    arrival counter, and the update is applied from the registers that computed
    it.  This removes the second launch and the cross-program fold, which
    dominate at the small sizes this path serves.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    acc_dtype = p_ptr.dtype.element_ty
    p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(acc_dtype)
    m = tl.load(m_ptr + offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + offsets, mask=mask, other=0.0)

    scaled_grad = g / grad_scale
    m_new = b1 * m + (1 - b1) * scaled_grad
    v_new = b2 * v + (1 - b2) * scaled_grad * scaled_grad

    if IS_FP64:
        # Mirror DeepSpeed's double path: sqrt/eps are float, the update is
        # still evaluated in the parameter dtype.
        if mode == 0:
            denom = tl.sqrt((v_new + eps).to(tl.float32)).to(acc_dtype)
        else:
            denom = (tl.sqrt(v_new.to(tl.float32)) + eps).to(acc_dtype)
    elif mode == 0:
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
        tl.store(
            p_copy_ptr + offsets,
            p_new.to(p_copy_ptr.dtype.element_ty),
            mask=mask,
        )


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
    """Fused LAMB (Layer-wise Adaptive Moments) optimizer step.

    Faithful port of DeepSpeed's ``fused_lamb`` CUDA operator. Performs one
    Adam-style moment update, then scales the step per-layer by the trust ratio
    ``||w|| / ||update||`` (clamped to ``[min_coeff, max_coeff]``), which makes
    training robust to very large batch sizes.

    Args:
        p (Tensor): model parameter, float32 or float64.  float64 requires
            ``runtime_device.support_fp64``.
        p_copy (Tensor): optional reduced-precision copy of the updated weights.
            It is written only on the fp16-gradient mixed path, matching
            DeepSpeed (whose fp32/fp64 branches pass NULL).  On that path a
            non-empty ``p_copy`` must be float16, as DeepSpeed reads it through
            a Half pointer; on the fp32/fp64 paths it is ignored and its dtype
            is not constrained.  An empty tensor skips the copy.
        m (Tensor): first moment estimate, same dtype as ``p``.
        v (Tensor): second moment estimate, same dtype as ``p``.
        g (Tensor): gradient.  It must match ``p``'s dtype for a full-precision
            step, or be float16 when ``p`` is float32 (fp16 gradients with fp32
            master weights and moments).
        lr (float): learning rate.
        beta1 (float): coefficient for the running average of the gradient.
        beta2 (float): coefficient for the running average of the squared gradient.
        max_coeff (float): upper bound of the trust ratio.
        min_coeff (float): lower bound of the trust ratio.
        eps (float): term added to the denominator for numerical stability.
        grad_scale (float): factor dividing the gradient before the update.
        step (int): optimizer step, used only for bias correction.
        mode (int): 0 keeps eps under the square root, 1 keeps it outside.
        bias_correction (int): 1 enables Adam bias correction, 0 disables it.
        decay (float): weight decay added to the update vector.

    Returns:
        Tensor: the per-layer trust ratio as a one-element tensor with the
            parameter dtype (float32 for the fp16-gradient mixed path). It is
            allocated per call, so the caller may keep it.
    """
    logger.debug("TRAIN LAMB")

    # DeepSpeed dispatches on the gradient dtype with
    # AT_DISPATCH_FLOATING_TYPES_AND_HALF; a dtype outside that dispatch (bf16,
    # integer, ...) raises NotImplementedError before any tensor is touched.
    if g.dtype not in (torch.float16, torch.float32, torch.float64):
        raise NotImplementedError(f"lamb is not implemented for {g.dtype}")

    if p.dtype not in _MASTER_DTYPES:
        raise RuntimeError(f"lamb expects float32 or float64 parameters, got {p.dtype}")
    if m.dtype != p.dtype or v.dtype != p.dtype:
        raise RuntimeError(
            "lamb requires the moment tensors to have the parameter dtype"
        )
    if g.dtype not in _GRAD_DTYPES[p.dtype]:
        raise RuntimeError(
            f"lamb with {p.dtype} parameters expects a "
            f"{' or '.join(str(dtype) for dtype in _GRAD_DTYPES[p.dtype])} gradient, "
            f"got {g.dtype}"
        )
    if p.dtype == torch.float64 and not runtime_device.support_fp64:
        raise RuntimeError(
            "lamb: float64 parameters require a device with fp64 support"
        )
    # Accept whatever the active backend calls its device -- 'cuda' on
    # nvidia/hygon, 'npu' on ascend -- rather than hard-coding CUDA. A CPU
    # tensor still has to be rejected: the kernels would be launched against
    # memory the accelerator cannot see.
    if p.device.type != runtime_device.name:
        raise RuntimeError(f"lamb only supports {runtime_device.name} tensors")

    # CHECK_INPUT parity: every non-empty operand is flat-indexed by the
    # kernels, so each must be contiguous and on the parameter's device.  An
    # empty p_copy is the one tensor DeepSpeed also skips (its CHECK_INPUT is
    # guarded by numel() > 0).
    for name, tensor in (("p", p), ("m", m), ("v", v), ("g", g)):
        if tensor.device != p.device:
            raise RuntimeError(f"lamb requires {name} on the same device as p")
        if not tensor.is_contiguous():
            raise RuntimeError(f"lamb requires {name} to be contiguous")

    n = p.numel()
    if m.numel() != n or v.numel() != n or g.numel() != n:
        raise RuntimeError(
            "lamb requires p, m, v and g to have the same number of elements"
        )
    if p_copy.numel() != 0 and p_copy.numel() != n:
        raise RuntimeError(
            "lamb requires p_copy to have the same number of elements as p, "
            "or to be empty"
        )
    if p_copy.numel() > 0:
        if p_copy.device != p.device:
            raise RuntimeError("lamb requires p_copy on the same device as p")
        if not p_copy.is_contiguous():
            raise RuntimeError("lamb requires p_copy to be contiguous")
        # DeepSpeed reads p_copy through a Half pointer on the fp16-gradient
        # branch.  The fp32/fp64 branches pass NULL, so p_copy's dtype is
        # irrelevant there.
        if g.dtype == torch.float16 and p_copy.dtype != torch.float16:
            raise RuntimeError(f"expected scalar type Half but found {p_copy.dtype}")

    # DeepSpeed launches zero blocks for an empty tensor and lets the CUDA
    # launch fail; reject it here instead of returning an uninitialised trust
    # ratio.  The same bound guards the kernel's 32-bit element offsets.
    if n == 0:
        raise RuntimeError("lamb requires at least one element")
    if n > _INT32_MAX:
        raise RuntimeError("lamb tensor is too large to index with int32")

    # Adam bias correction, matching DeepSpeed's step_size computation.
    if bias_correction == 1:
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
    else:
        step_size = lr

    # DeepSpeed only writes p_copy on its fp16-gradient path; the fp32 and fp64
    # branches pass NULL regardless of p_copy being non-empty.  Match that:
    # a full-precision step leaves p_copy untouched.
    has_p_copy = p_copy.numel() > 0 and g.dtype == torch.float16
    p_copy_in = p_copy if has_p_copy else p

    if n <= _FUSED_MAX:
        # One program covers the tensor, so its norms are the global norms and
        # the entire step is a single launch.  The block is the tensor's own
        # width rather than a fixed tile.
        fused_block = max(triton.next_power_of_2(n), 128)
        fused_kwargs = (
            {"num_warps": _FP64_NUM_WARPS} if p.dtype == torch.float64 else {}
        )
        lamb_coeff_val = torch.empty((1,), dtype=p.dtype, device=p.device)
        with torch_device_fn.device(p.device):
            lamb_fused_kernel[(1,)](
                p,
                p_copy_in,
                m,
                v,
                g,
                n,
                grad_scale,
                beta1,
                beta2,
                eps,
                decay,
                min_coeff,
                max_coeff,
                step_size,
                lamb_coeff_val,
                mode=mode,
                IS_FP64=p.dtype == torch.float64,
                HAS_P_COPY=has_p_copy,
                BLOCK_SIZE=fused_block,
                **fused_kwargs,
            )
        return lamb_coeff_val

    # A wide block leaves only a handful of programs for a mid-sized tensor, so
    # the kernels spend their time waiting on memory instead of covering it.
    # Capping the block at 1024 elements keeps enough programs in flight to hide
    # that latency without making the cross-block reduction expensive.  The
    # grid is capped like DeepSpeed's (512 blocks); the kernels grid-stride.
    BLOCK_SIZE = triton.next_power_of_2(n)
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    if p.dtype == torch.float64:
        fp64_block, fp64_warps = _fp64_launch_config(n)
        BLOCK_SIZE = min(BLOCK_SIZE, fp64_block)
        max_blocks = _FP64_MAX_BLOCKS
        launch_kwargs = {"num_warps": fp64_warps}
    else:
        BLOCK_SIZE = min(BLOCK_SIZE, 1024)
        max_blocks = _MAX_BLOCKS
        launch_kwargs = {}
    num_blocks = min(triton.cdiv(n, BLOCK_SIZE), max_blocks)

    # Reduction workspace, allocated per call.  A module-level pool keyed by
    # (device, num_blocks) has no natural bound, so a caller cycling through
    # element counts would grow it without limit; these three small allocations
    # are the whole cost of not keeping one.  ``counter`` is the only buffer that
    # must start at zero, and being the only zeroed one is why it costs a memset.
    # Norm partials and the trust ratio follow the master dtype so an fp64 step
    # does not silently round its reduction to fp32.
    norms = torch.empty((2 * num_blocks,), dtype=p.dtype, device=p.device)
    counter = torch.zeros((1,), dtype=torch.int32, device=p.device)
    lamb_coeff_val = torch.empty((1,), dtype=p.dtype, device=p.device)
    reduce_block = min(max(triton.next_power_of_2(num_blocks), 32), 2048)

    with torch_device_fn.device(p.device):
        lamb_part1_kernel[(num_blocks,)](
            p,
            m,
            v,
            g,
            n,
            grad_scale,
            beta1,
            beta2,
            eps,
            decay,
            max_coeff,
            min_coeff,
            norms,
            counter,
            lamb_coeff_val,
            mode=mode,
            IS_FP64=p.dtype == torch.float64,
            REDUCE_BLOCK=reduce_block,
            BLOCK_SIZE=BLOCK_SIZE,
            **launch_kwargs,
        )

        lamb_part3_kernel[(num_blocks,)](
            p,
            p_copy_in,
            m,
            v,
            n,
            eps,
            step_size,
            decay,
            lamb_coeff_val,
            mode=mode,
            IS_FP64=p.dtype == torch.float64,
            HAS_P_COPY=has_p_copy,
            BLOCK_SIZE=BLOCK_SIZE,
            **launch_kwargs,
        )

    return lamb_coeff_val
