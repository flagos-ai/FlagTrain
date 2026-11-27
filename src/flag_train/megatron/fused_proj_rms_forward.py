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
"""Fused projection + RMS normalization (forward).

This is the proj_rms stage of Megatron-LM's
``megatron/core/fusions/fused_mhc_kernels.py`` as a standalone operator. For
``x`` of shape ``[..., K]`` and ``weight`` of shape ``[N, K]`` it computes

.. math::

    proj = x @ weight^T

    r = \\frac{1}{\\lVert x \\rVert_2 / \\sqrt{K} + \\epsilon}

or, with ``eps_inside_sqrt=True`` (Megatron main, ``mhc_norm_eps_inside_sqrt``):

.. math::

    r = \\frac{1}{\\sqrt{\\overline{x^2} + \\epsilon}}

The Megatron reference is ``native_proj_rms`` in
``megatron.core.transformer.hyper_connection``; it is the oracle the functional
suite compares against and the baseline the performance suite times. Megatron
0.17.x and main differ in this operator's contract: 0.17.x returns the input
dtype and has no ``eps_inside_sqrt``, while main returns fp32 and supports the
inside-sqrt form. ``fp32_output=None`` detects which Megatron is installed, so
the same FlagTrain call follows either contract; the keyword also forces one
for tests.

Both outputs come from a single launch, made of two kinds of program. The
first ``cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N)`` own a ``[BLOCK_M, BLOCK_N]`` tile
of ``proj`` each and stream ``x`` and ``weight`` over ``K`` with an accumulator
that is fp32 except in the legacy fp64 contract. The remaining
``cdiv(M, RMS_BLOCK_M)`` own a block of rows and
reduce ``||x||`` over ``K``. Giving the reduction its own programs is what
keeps it off the projection's critical path: at mHC's N=24 a row block has a
single projection tile, so a second pass inside that program would add its
whole streaming time to the launch. On the performance suite's shapes the split
is 1.2-1.9x over that arrangement where N is 1024 or less, and never behind it
at the wide shapes. The rows a reduction program reads are the ones a
projection program read moments earlier, and its programs are placed at the end
of the grid, so the re-read mostly lands in L2.

The RMS reduction is deliberately a separate loop rather than a branch inside
the projection loop. A value that is both a ``tl.dot`` operand and the source
of a non-MMA reduction inside a pipelined loop is miscompiled by Triton 3.6
once ``num_stages > 1``: the projection comes back wrong (silently, by tens of
percent) while ``r`` stays correct. Keeping the two passes in separate loops
avoids that entirely, at the cost of one cached re-read.

``weight`` is 2-D and is read as ``[N, K]``; ``x`` and ``weight`` may have
different floating dtypes, in which case both are promoted to
``torch.promote_types(x.dtype, weight.dtype)`` for the legacy 0.17.x contract.
The fp32-output contract instead computes both in fp32, as Megatron main does.
With equal dtypes the legacy promoted dtype is the input dtype.

CPU inputs are copied to the active accelerator and the results copied back, so
the arithmetic still runs in the Triton kernels; there is no torch reference
path. A CPU-only environment is therefore not supported. Legacy fp64 is computed
by the same kernels in fp64. Inputs are limited to real fp16/bf16/fp32/fp64;
complex, float8 and integer tensors are rejected. The Triton arithmetic ignores
ambient autocast: main mode matches Megatron main's autocast-disabled recipe,
and the legacy contract uses its promoted compute dtype.
"""

import functools
import inspect
import logging
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import triton_lang_extension as tle
from flag_train.utils.device_info import get_sm_count

__all__ = ["fused_proj_rms_forward"]

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


@functools.lru_cache(maxsize=1)
def _megatron_proj_rms_outputs_fp32() -> bool:
    """Whether the installed Megatron's ``native_proj_rms`` returns fp32.

    Megatron main computes the projection/norm in fp32 and added
    ``eps_inside_sqrt`` in the same change; 0.17.x returns the input dtype and
    has no such argument. Detect the behaviour by signature so the same
    FlagTrain call is a drop-in on both versions. No Megatron (or an
    uninspectable wrapper) keeps the 0.17.x contract.
    """
    try:
        from megatron.core.transformer.hyper_connection import native_proj_rms
    except Exception:
        return False
    fn = getattr(native_proj_rms, "_torchdynamo_orig_callable", native_proj_rms)
    try:
        return "eps_inside_sqrt" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        pass
    # Last resort: the change landed after the 0.17.x line.
    try:
        import importlib.metadata
        import re

        version = importlib.metadata.version("megatron-core")
        match = re.match(r"\s*(\d+)\.(\d+)", version)
        if match is None:
            return False
        return (int(match.group(1)), int(match.group(2))) >= (0, 21)
    except Exception:
        return False


def _resolve_fp32_output(eps_inside_sqrt: bool, fp32_output: Optional[bool]) -> bool:
    """Resolve the output-dtype compatibility mode.

    ``fp32_output=None`` follows the installed Megatron (fp32 on main, input
    dtype on 0.17.x); an explicit bool overrides it for tests or callers that
    know which contract they need. ``eps_inside_sqrt`` only exists in the
    fp32-output Megatron, so the two cannot be mixed with legacy output.
    """
    if fp32_output is None:
        fp32_output = True if eps_inside_sqrt else _megatron_proj_rms_outputs_fp32()
    if eps_inside_sqrt and not fp32_output:
        raise ValueError(
            "eps_inside_sqrt=True requires the fp32-output Megatron semantics"
        )
    return bool(fp32_output)


# ``tl.dot`` needs at least 16 rows/columns; no tile below goes under it.
_MAX_BLOCK_N = 128
_NUM_STAGES = 4
# Shared memory a pipelined tile pair may claim, in bytes. There are two
# budgets because what the pipeline buys depends on whether the projection grid
# fills the device. While it does not, a program has no neighbour to overlap
# its memory stalls with, so the shared memory is better spent buffering more K
# iterations; once it does, the same memory is worth more as extra resident
# programs. Both stay well under the 227 KB an H100/H200/H20 exposes per SM,
# which leaves headroom for the operand staging and layout conversions a
# launch adds on top of the buffers. Far larger requests are rejected at
# launch, not at the first wrong answer.
_DEEP_PIPELINE_SMEM = 128 * 1024
_WIDE_GRID_SMEM = 64 * 1024
_BLOCK_K_CHOICES = (512, 256, 128, 64, 32, 16)
# Row blocks from this many rows up get BLOCK_M=64. Below it a 64-row tile
# leaves the device with too few projection programs, and the parallelism of a
# 16-row tile is worth more than the traffic it repeats.
_LARGE_M = 2048
# At or below this many rows, an N of 24-32 is split into two 16-wide tiles: the
# second read of ``x`` is a few hundred KB there, and the extra programs buy
# more than it costs.
_TINY_M = 256
# Rows one reduction program owns, and the K tile it reads them with. 64 rows
# is one projection row block, so while the projection grid is smaller than the
# device an mHC launch adds one reduction program per projection tile -- enough
# to fill the tail without outnumbering the matmul. Once the matmul is several
# waves deep the tail is where the reduction hides, and eight rows per program
# spread it over the SMs the matmul is leaving idle: measured 2-4% at N=80 and
# N=288, and nothing either way at the wide shapes. Pairing is the wrong choice
# in that regime, because it is the program count, not the block, that the tail
# has room for.
_RMS_BLOCK_M = 64
_RMS_TAIL_BLOCK_M = 8
_RMS_BLOCK_K = 256
# Row block of the standalone reduction (the ``N == 0`` path). Nothing else runs
# in that launch, so the only question is how many programs are in flight, and
# this is a pure read: four rows by a 256-element K tile puts eight elements on
# each thread of four warps, and the launch reads at 3.1 TB/s, which is what
# this device gives a streaming load. The 64 rows a projection row block would
# suggest reach 0.8 TB/s, because 64 programs of four warps leave most of the
# memory system idle -- at M=4096 that is 40.6 us against 17.4.
_RMS_ONLY_BLOCK_M = 4


def _check_input(tensor, name):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if name == "weight":
        if tensor.dim() != 2:
            raise ValueError(f"weight must be 2-D, got shape {tuple(tensor.shape)}")
    elif tensor.dim() < 1:
        raise ValueError(f"{name} must be at least 1-D, got a scalar")
    if tensor.dtype not in _SUPPORTED_DTYPES:
        supported = ", ".join(str(dtype) for dtype in _SUPPORTED_DTYPES)
        raise ValueError(f"{name} must have dtype {supported}, got {tensor.dtype}")
    # CPU inputs are staged to the active accelerator before the Triton launch;
    # a non-CPU tensor must belong to the active accelerator backend.
    if tensor.device.type == "cpu":
        return
    if tensor.device.type != runtime_device.name:
        raise ValueError(
            f"{name} must be a {runtime_device.name} or CPU tensor, got "
            f"{tensor.device.type}"
        )


def _accelerator_device(op_name: str) -> str:
    """Move CPU inputs here so the computation still runs in Triton."""
    name = runtime_device.name
    if name == "cpu":
        raise RuntimeError(
            f"{op_name} cannot run CPU inputs without an accelerator: "
            "no device Triton backend is available"
        )
    is_available = getattr(torch_device_fn, "is_available", None)
    if callable(is_available) and not is_available():
        raise RuntimeError(f"{op_name} cannot run CPU inputs: {name} is unavailable")
    return name


def _block_n(N, M):
    """Output tile width: all of N while it is narrow, capped otherwise.

    N grows the tile because the ``weight`` tile is what is reused across row
    blocks. The two exceptions are the narrow cases where the tile count is
    what fills the device: mHC's N=24 with only a few rows, where two 16-wide
    tiles beat one 32-wide one, and N=1024, where 64 is measurably ahead of the
    wider tile that would halve the program count.
    """
    if N <= 16:
        return 16
    if N <= 32:
        return 16 if M <= _TINY_M else 32
    if N <= 512:
        return 32
    if N <= 2048:
        return 64
    return _MAX_BLOCK_N


def _block_k(block_m, block_n, K, element, budget):
    """Widest K tile whose two operand tiles fit ``budget`` across the pipeline."""
    for candidate in _BLOCK_K_CHOICES:
        if candidate <= max(K, 1) and (
            (block_m + block_n) * candidate * element * _NUM_STAGES <= budget
        ):
            return candidate
    return _BLOCK_K_CHOICES[-1]


@functools.lru_cache(maxsize=128)
def _config(M, N, K, dtype):
    """Tile geometry and launch parameters for one forward launch.

    The grid is sized before the K tile is chosen: a projection grid smaller
    than the device's SM count is the case that gets the deeper pipeline (see
    ``_DEEP_PIPELINE_SMEM``). Measured on the performance suite's shapes, the
    two budgets differ by up to 1.3x where they disagree, and the crossover
    sits where the grid stops covering the SMs.

    Cached because it is pure, and because at the mHC shapes a training step's
    worth of work is smaller than what deciding the tiles costs: the lookup
    below is the whole per-call cost once the shape has been seen.
    """
    element = torch.finfo(dtype).bits // 8
    block_m = 64 if M >= _LARGE_M else 16
    block_n = _block_n(N, M)
    num_proj = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    budget = _DEEP_PIPELINE_SMEM if num_proj < get_sm_count() else _WIDE_GRID_SMEM
    block_k = _block_k(block_m, block_n, K, element, budget)
    num_warps = 8 if block_m >= 64 else 4
    num_stages = _NUM_STAGES if K >= block_k else 1
    return block_m, block_n, block_k, num_warps, num_stages


@triton.jit
def _proj_rms_forward_kernel(
    x_ptr,
    weight_ptr,
    proj_ptr,
    r_ptr,
    M,
    N,
    K,
    eps,
    num_pid_mm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    IEEE: tl.constexpr,
    FP64: tl.constexpr,
    RMS_BLOCK_M: tl.constexpr,
    RMS_BLOCK_K: tl.constexpr,
    EPS_INSIDE_SQRT: tl.constexpr,
):
    """Projection tile for the first programs, RMS scale for the rest.

    ``num_pid_mm`` is ``cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N)``: the programs
    below it own a projection tile, the ones from there on own a block of rows
    and write their scales to ``r``. It is a runtime argument rather than a
    launch constant so one compiled kernel serves every shape that shares the
    tile geometry.

    Every operand is contiguous -- the caller makes ``x`` and ``weight`` so and
    allocates both outputs so -- which is why the strides are the row lengths
    rather than arguments. Each one dropped here is a per-call cost the host
    pays for a tile-shaped launch, and at the mHC shapes the host, not the
    card, is what a training step waits on.
    """
    pid = tle.program_id(0)
    if pid < num_pid_mm:
        # Group the m blocks (the CUDA matmul tutorial's swizzle) so that
        # programs running at the same time work through the same n tile and
        # share it through L2. A no-op when N fits in a single tile, as it does
        # for mHC.
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N

        if FP64:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            x = tl.load(
                x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            w = tl.load(
                weight_ptr + offs_n[:, None] * K + offs_k[None, :],
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
            )
            if IEEE:
                acc += tl.dot(x, tl.trans(w), input_precision="ieee")
            else:
                # Mixed x/weight dtypes were promoted on the host, so both
                # operands already agree; fp16/bf16 accumulate in fp32 on the
                # tensor cores.
                acc += tl.dot(x, tl.trans(w))

        tl.store(
            proj_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc.to(proj_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )
    else:
        # A second pass over the same rows, in its own programs; see the module
        # docstring for why it is neither paired with the projection nor fused
        # into its loop. Every name here is prefixed so that Triton does not
        # have to give the two branches' values a common type: the reduction's
        # row block is its own choice (``_RMS_BLOCK_M``), not the projection's.
        rms_offs_m = (pid - num_pid_mm) * RMS_BLOCK_M + tl.arange(0, RMS_BLOCK_M)
        rms_mask_m = rms_offs_m < M
        if FP64:
            sum_sq = tl.zeros((RMS_BLOCK_M,), dtype=tl.float64)
        else:
            sum_sq = tl.zeros((RMS_BLOCK_M,), dtype=tl.float32)
        for k_start in range(0, K, RMS_BLOCK_K):
            rms_offs_k = k_start + tl.arange(0, RMS_BLOCK_K)
            rms_mask_k = rms_offs_k < K
            rms_x = tl.load(
                x_ptr + rms_offs_m[:, None] * K + rms_offs_k[None, :],
                mask=rms_mask_m[:, None] & rms_mask_k[None, :],
                other=0.0,
            )
            if FP64:
                rms_x = rms_x.to(tl.float64)
            else:
                rms_x = rms_x.to(tl.float32)
            sum_sq += tl.sum(rms_x * rms_x, axis=1)

        if EPS_INSIDE_SQRT:
            # r = 1 / sqrt(mean(x^2) + eps), Megatron main's
            # mhc_norm_eps_inside_sqrt contract.
            scale = 1.0 / tl.sqrt(sum_sq / (K * 1.0) + eps)
        else:
            norm = tl.sqrt(sum_sq)
            scale = 1.0 / (norm / tl.sqrt(K * 1.0) + eps)
        tl.store(r_ptr + rms_offs_m, scale.to(r_ptr.dtype.element_ty), mask=rms_mask_m)


@triton.jit
def _rms_only_kernel(
    x_ptr,
    r_ptr,
    M,
    K,
    eps,
    stride_xm,
    stride_xk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FP64: tl.constexpr,
    EPS_INSIDE_SQRT: tl.constexpr,
):
    """RMS scale alone, for the degenerate ``N == 0`` case."""
    pid = tle.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    if FP64:
        sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float64)
    else:
        sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        if FP64:
            x = x.to(tl.float64)
        else:
            x = x.to(tl.float32)
        sum_sq += tl.sum(x * x, axis=1)

    if EPS_INSIDE_SQRT:
        scale = 1.0 / tl.sqrt(sum_sq / (K * 1.0) + eps)
    else:
        norm = tl.sqrt(sum_sq)
        scale = 1.0 / (norm / tl.sqrt(K * 1.0) + eps)
    tl.store(r_ptr + offs_m, scale.to(r_ptr.dtype.element_ty), mask=mask_m)


def fused_proj_rms_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    eps_inside_sqrt: bool = False,
    *,
    fp32_output: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused projection and RMS normalization.

    Args:
        x (Tensor): ``[..., K]`` fp16, bf16, fp32 or fp64 activations. CPU
            inputs are copied to the active accelerator for the Triton launch
            and the outputs copied back; leading axes are flattened into the
            2-D kernel. A non-contiguous input is copied.
        weight (Tensor): ``[N, K]`` projection weights. Its dtype may differ
            from ``x``'s; both are promoted to
            ``torch.promote_types(x.dtype, weight.dtype)`` in the legacy
            contract, or computed in fp32 in the Megatron-main contract.
        eps (float): numerical floor in the RMS scale.
        eps_inside_sqrt (bool): use Megatron main's
            ``rsqrt(mean(x**2) + eps)`` instead of
            ``1 / (||x|| / sqrt(K) + eps)``. Only the fp32-output Megatron
            defines this mode.
        fp32_output (Optional[bool]): output-dtype compatibility override.
            ``None`` follows the installed Megatron (fp32 on main, input dtype
            on 0.17.x), so the same call is a drop-in on both. ``True`` forces
            the main contract; ``False`` forces the 0.17.x contract.

    Returns:
        Tuple[Tensor, Tensor]: ``proj`` of shape ``[..., N]`` and ``r`` of
        shape ``[..., 1]``. Both are fp32 when ``fp32_output`` resolves true
        (Megatron main), otherwise ``torch.promote_types(x.dtype, weight.dtype)``.
    """
    logger.debug("TRAIN FUSED_PROJ_RMS_FORWARD")
    _check_input(x, "x")
    _check_input(weight, "weight")
    if x.shape[-1] != weight.shape[1]:
        raise ValueError(
            "x and weight must agree on the reduction dimension, got "
            f"{tuple(x.shape)} and {tuple(weight.shape)}"
        )
    if x.device != weight.device:
        raise ValueError(
            f"x and weight must be on the same device, got {x.device} and "
            f"{weight.device}"
        )
    if not isinstance(eps, (int, float)) or eps < 0:
        raise ValueError(f"eps must be a non-negative float, got {eps!r}")

    fp32_output = _resolve_fp32_output(eps_inside_sqrt, fp32_output)
    out_dtype = (
        torch.float32 if fp32_output else torch.promote_types(x.dtype, weight.dtype)
    )
    if out_dtype == torch.float64 and not runtime_device.support_fp64:
        raise RuntimeError(
            "fused_proj_rms_forward: the legacy float64 contract requires a "
            "device with fp64 support; use fp32_output=True to compute in fp32"
        )
    orig_shape = x.shape[:-1]
    K = x.shape[-1]
    N = weight.shape[0]
    M = 1
    for dim in orig_shape:
        M *= dim
    out_shape = orig_shape + (N,)
    r_shape = orig_shape + (1,)

    orig_device = x.device
    cpu_input = orig_device.type == "cpu"

    # Triton has no CPU backend, so a CPU input is staged through the active
    # accelerator: torch only copies, the arithmetic stays in the kernels.
    # With M == 0 there is nothing to compute on either device.
    if cpu_input and M != 0:
        x = x.to(_accelerator_device("fused_proj_rms_forward"))
        weight = weight.to(x.device)

    proj = x.new_empty(out_shape, dtype=out_dtype)
    r = x.new_empty(r_shape, dtype=out_dtype)
    if M == 0:
        return proj, r

    # One dtype for the whole computation, as Megatron's fused path does. Each
    # conversion is guarded rather than requested outright: at the mHC shapes
    # the caller usually hands over the promoted dtype already, and the kernel
    # takes its strides from M, N and K, so both operands have to be
    # contiguous by the time it runs.
    if x.dtype != out_dtype:
        x = x.to(out_dtype)
    if not x.is_contiguous():
        x = x.contiguous()
    if weight.dtype != out_dtype:
        weight = weight.to(out_dtype)
    if not weight.is_contiguous():
        weight = weight.contiguous()

    # The kernel is 2-D; leading axes are only shape, not index arithmetic.
    # The outputs were allocated at the full shape above, so their 2-D views
    # share that storage instead of keeping a second copy alive.
    x2 = x.reshape(M, K)
    proj2 = proj.reshape(M, N)
    r2 = r.reshape(M, 1)

    device_index = x2.device.index
    if device_index is None:
        device_index = torch_device_fn.current_device()
    guard = (
        nullcontext()
        if torch_device_fn.current_device() == device_index
        else torch_device_fn.device(x2.device)
    )

    if N == 0:
        block_k = min(_RMS_BLOCK_K, triton.next_power_of_2(max(K, 1)))
        # A program wants about 1024 elements, eight on each thread of four
        # warps; a K tile narrower than the 256 above takes more rows to get
        # there.
        block_m = min(64, max(_RMS_ONLY_BLOCK_M, 1024 // block_k))
        with guard:
            _rms_only_kernel[(triton.cdiv(M, block_m),)](
                x2,
                r2,
                M,
                K,
                float(eps),
                x2.stride(0),
                x2.stride(1),
                BLOCK_M=block_m,
                BLOCK_K=block_k,
                FP64=out_dtype == torch.float64,
                EPS_INSIDE_SQRT=eps_inside_sqrt,
                num_warps=4,
                num_stages=_NUM_STAGES if K >= block_k else 1,
            )
        if cpu_input:
            proj = proj.to(orig_device)
            r = r.to(orig_device)
        return proj, r

    block_m, block_n, block_k, num_warps, num_stages = _config(M, N, K, out_dtype)
    num_pid_mm = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    fills_device = num_pid_mm >= get_sm_count()
    rms_block_m = min(block_m, _RMS_TAIL_BLOCK_M if fills_device else _RMS_BLOCK_M)
    rms_block_k = min(_RMS_BLOCK_K, triton.next_power_of_2(max(K, 1)))
    grid = (num_pid_mm + triton.cdiv(M, rms_block_m),)
    with guard:
        _proj_rms_forward_kernel[grid](
            x2,
            weight,
            proj2,
            r2,
            M,
            N,
            K,
            float(eps),
            num_pid_mm,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            IEEE=out_dtype in (torch.float32, torch.float64),
            FP64=out_dtype == torch.float64,
            RMS_BLOCK_M=rms_block_m,
            RMS_BLOCK_K=rms_block_k,
            EPS_INSIDE_SQRT=eps_inside_sqrt,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    if cpu_input:
        proj = proj.to(orig_device)
        r = r.to(orig_device)
    return proj, r
