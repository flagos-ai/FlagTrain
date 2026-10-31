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
"""Mixed-precision GEMM with weight-only integer quantization.

Port of DeepSpeed's ``mixed_gemm``: fp16/bf16 activations against an int4/int8 weight with a
per-output-channel scale. ``weight`` arrives already packed in DeepSpeed's
``preprocess_weights_for_mixed_gemm`` layout; nothing is quantized here. The gated
activations are declared but rejected, as the runner rejects them.

MThreads: unpack the weight into a dense buffer once per call, then hand the GEMM to muBLAS
(373.8 TFLOP/s at 4096^3, against 97.5 for the best ``tl.dot`` tile here). The buffer is per
call and never cached, so the unpack is always paid. Shapes the unpack kernel cannot address
fall back to the fused kernel.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry, tl_extra_shim
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# ``ActivationType`` values, from DeepSpeed's ``utils/activation_type.h``. ``IDENTITY`` is 6.
GELU = tl.constexpr(0)
RELU = tl.constexpr(1)
SILU = tl.constexpr(2)
GEGLU = tl.constexpr(3)
ReGLU = tl.constexpr(4)
SiGLU = tl.constexpr(5)
IDENTITY = tl.constexpr(6)
INVALID_TYPE = tl.constexpr(-1)

# The runner supports exactly these four; everything else is rejected.
_SUPPORTED_ACTIVATIONS = frozenset((int(GELU), int(RELU), int(SILU), int(IDENTITY)))
_GATED_ACTIVATIONS = frozenset((int(GEGLU), int(ReGLU), int(SiGLU)))


@libentry()
@triton.jit
def mixed_gemm_kernel(
    output_ptr,
    hidden_states_ptr,
    weight_ptr,
    scales_ptr,
    bias_ptr,
    m,
    n,
    k,
    stride_hm,
    stride_hk,
    stride_om,
    stride_on,
    HAS_BIAS: tl.constexpr,
    ACT_TYPE: tl.constexpr,
    NUM_BITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Weight-only dequantization fused into a tiled GEMM.

    ``w = (q - 2**(bits-1)) * scale[n]`` is applied as the tile is fetched, the product
    accumulates in fp32, and the epilogue adds the bias and the activation.
    """
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rows_mask = rows < m

    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    cols_mask = cols < n

    sc = tl.load(scales_ptr + cols, mask=cols_mask, other=0.0)

    # Layout constants, straight from ``mixed_gemm_B_layout.h``.
    EIR = 32 // NUM_BITS  # elements per 32-bit register
    INTER = 16 // NUM_BITS  # columns_interleaved: 4 (int4), 2 (int8)
    VPT = 2 * NUM_BITS  # rows_per_column_tile // EIR
    BROWS = 8 * (16 // NUM_BITS)  # B_ROWS_PER_MMA

    # Column side of stages 2-3; ``k`` is the R of the ``[R, C] = [k, n]`` weight.
    nvr = k // EIR  # num_vec_rows
    col_part = (cols // INTER) * (nvr * INTER) + VPT * (cols % INTER)

    for kb in range(0, tl.cdiv(k, BLOCK_K)):
        koff = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        kmask = koff < k

        hidden = tl.load(
            hidden_states_ptr + rows[:, None] * stride_hm + koff[None, :] * stride_hk,
            mask=rows_mask[:, None] & kmask[None, :],
            other=0.0,
        )

        # Stage 1 (permute_B_rows) is *inverted*: using f directly works for int4, where f
        # is an involution, and is wrong for int8.
        t = koff % BROWS
        k1 = (koff // BROWS) * BROWS + ((t % 8) // 2) * EIR + 2 * (t // 8) + (t % 2)
        kr = k1 // EIR
        ks = k1 % EIR
        ks4 = (EIR // 2) * (ks % 2) + ks // 2  # stage 4 register reorder
        row_part = INTER * (kr // VPT) * VPT + (kr % VPT)

        slot = (col_part[None, :] + row_part[:, None]) * EIR + ks4[:, None]
        wmask = kmask[:, None] & cols_mask[None, :]

        if NUM_BITS == 8:
            wq = tl.load(weight_ptr + slot, mask=wmask, other=128)
        else:
            # Two elements share a byte and the *slot's* parity picks the half; 0x88 decodes
            # to zero, while a pad of 0 would give -8 * scale.
            packed = tl.load(weight_ptr + slot // 2, mask=wmask, other=0x88)
            lo = packed & 0x0F
            hi = (packed >> 4) & 0x0F
            wq = tl.where((slot % 2) == 1, hi, lo)

        # Via fp32: exact for these integers, and the only form that lowers to bf16 here.
        w = (wq.to(tl.float32) - (1 << (NUM_BITS - 1))).to(
            hidden_states_ptr.dtype.element_ty
        )

        acc = tl.dot(hidden, w, acc)

    acc *= sc[None, :]
    if HAS_BIAS:
        bias = tl.load(bias_ptr + cols, mask=cols_mask, other=0.0)
        acc += bias[None, :]

    if ACT_TYPE == RELU:
        acc = tl.maximum(acc, 0.0)
    elif ACT_TYPE == SILU:
        acc = acc * tl.sigmoid(acc)
    elif ACT_TYPE == GELU:
        c = 0.7978845608028654  # sqrt(2 / pi), tanh approximation of GELU
        z = c * (acc + 0.044715 * acc * acc * acc)
        acc = 0.5 * acc * (1.0 + tl_extra_shim.tanh(z))

    tl.store(
        output_ptr + rows[:, None] * stride_om + cols[None, :] * stride_on,
        acc.to(output_ptr.dtype.element_ty),
        mask=rows_mask[:, None] & cols_mask[None, :],
    )


# Fallback tile: ``BLOCK_N`` dominates, because the weight fetch is a gather whose volume
# scales with n. These are the generic defaults, not a tuned schedule.
_DEFAULT_TILE = {
    "BLOCK_M": 256,
    "BLOCK_N": 32,
    "BLOCK_K": 32,
    "num_warps": 4,
    "num_stages": 3,
}

# Unpack program: elements of k and n per launch. Both divide the layout's rules evenly.
_DQ_BLOCK_K = 64
_DQ_BLOCK_N = 128


@triton.jit
def _dequant_kernel(
    weight_ptr,
    out_ptr,
    scales_ptr,
    n,
    k,
    NUM_BITS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Unpack and scale a tile of the weight into a dense ``(k, n)`` buffer."""
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    koff = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    kmask = koff < k
    cols_mask = cols < n

    EIR: tl.constexpr = 32 // NUM_BITS
    INTER: tl.constexpr = 16 // NUM_BITS
    VPT: tl.constexpr = 2 * NUM_BITS
    BROWS: tl.constexpr = 8 * (16 // NUM_BITS)
    nvr = k // EIR
    col_part = (cols // INTER) * (nvr * INTER) + VPT * (cols % INTER)

    t = koff % BROWS
    k1 = (koff // BROWS) * BROWS + ((t % 8) // 2) * EIR + 2 * (t // 8) + (t % 2)
    kr = k1 // EIR
    ks = k1 % EIR
    ks4 = (EIR // 2) * (ks % 2) + ks // 2
    row_part = INTER * (kr // VPT) * VPT + (kr % VPT)

    slot = (col_part[None, :] + row_part[:, None]) * EIR + ks4[:, None]
    wmask = kmask[:, None] & cols_mask[None, :]
    if NUM_BITS == 8:
        wq = tl.load(weight_ptr + slot, mask=wmask, other=128)
    else:
        packed = tl.load(weight_ptr + slot // 2, mask=wmask, other=0x88)
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        wq = tl.where((slot % 2) == 1, hi, lo)

    # Via fp32, as in the fused kernel. Scaling here leaves the GEMM no epilogue to do.
    w = (wq.to(tl.float32) - (1 << (NUM_BITS - 1))).to(out_ptr.dtype.element_ty)
    w *= tl.load(scales_ptr + cols, mask=cols_mask, other=0.0)[None, :]
    tl.store(out_ptr + koff[:, None] * n + cols[None, :], w, mask=wmask)


# Epilogue block for the muBLAS path, which only runs with a bias or an activation.
_EPI_BLOCK = 1024


@triton.jit
def _epilogue_kernel(
    out_ptr,
    bias_ptr,
    total,
    n,
    HAS_BIAS: tl.constexpr,
    ACT_TYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Add the bias and the activation to a contiguous ``(m, n)`` buffer in place."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    cols = offs % n
    acc = tl.load(out_ptr + offs, mask=mask).to(tl.float32)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + cols, mask=mask, other=0.0)

    if ACT_TYPE == RELU:
        acc = tl.maximum(acc, 0.0)
    elif ACT_TYPE == SILU:
        acc = acc * tl.sigmoid(acc)
    elif ACT_TYPE == GELU:
        c = 0.7978845608028654  # sqrt(2 / pi), tanh approximation of GELU
        z = c * (acc + 0.044715 * acc * acc * acc)
        acc = 0.5 * acc * (1.0 + tl_extra_shim.tanh(z))

    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def _mixed_gemm_split(
    output,
    hidden_states,
    weight,
    scales,
    bias_in,
    has_bias,
    num_bits,
    activation_raw,
    m,
    n,
    k,
):
    """Unpack the weight once, then let muBLAS do the dense GEMM."""
    dense = torch.empty((k, n), dtype=hidden_states.dtype, device=hidden_states.device)

    with torch_device_fn.device(hidden_states.device):
        _dequant_kernel[(triton.cdiv(k, _DQ_BLOCK_K), triton.cdiv(n, _DQ_BLOCK_N))](
            weight,
            dense,
            scales,
            n,
            k,
            NUM_BITS=num_bits,
            BLOCK_K=_DQ_BLOCK_K,
            BLOCK_N=_DQ_BLOCK_N,
            num_warps=4,
            num_stages=2,
        )
        torch.mm(hidden_states, dense, out=output)

        # Identity with no bias is the measured path; skip the launch when nothing applies.
        if has_bias or activation_raw != int(IDENTITY):
            _epilogue_kernel[(triton.cdiv(m * n, _EPI_BLOCK),)](
                output,
                bias_in,
                m * n,
                n,
                HAS_BIAS=has_bias,
                ACT_TYPE=activation_raw,
                BLOCK=_EPI_BLOCK,
                num_warps=4,
            )

    return output


def _check_launch_arguments(
    output, hidden_states, weight, scales, bias, num_bits, activation_raw
):
    """Validate one ``mixed_gemm`` call and return ``(m, n, k)``.

    These mirror the launcher's ``TORCH_CHECK``s plus a device check, so no backend accepts
    a call another one rejects.
    """
    if num_bits not in (4, 8):
        raise ValueError(f"Data width must be 4 or 8, got {num_bits}")
    if output.dtype != hidden_states.dtype:
        raise ValueError("Output and hidden states must have the same dtype")
    if output.size(0) != hidden_states.size(0):
        raise ValueError("Token dimension mismatch")

    m = output.size(0)
    k = hidden_states.size(1)
    if weight.size(0) != k:
        raise ValueError("Weight dimension mismatch")
    n = weight.size(1) * 2 if num_bits == 4 else weight.size(1)

    if output.size(1) != n:
        raise ValueError(f"Output width {output.size(1)} does not match n={n}")
    if scales.numel() != n:
        raise ValueError(f"Expected {n} scales, got {scales.numel()}")
    if bias is not None and bias.numel() != n:
        raise ValueError(f"Expected {n} bias entries, got {bias.numel()}")

    # Outside the four epilogues the runner instantiates.
    if activation_raw not in _SUPPORTED_ACTIVATIONS:
        if activation_raw in _GATED_ACTIVATIONS:
            raise ValueError("Fused gated activations not supported")
        raise ValueError(f"Invalid activation type: {activation_raw}")

    # Against the runtime's name, not a hard-coded "cuda": the same source runs on npu/hcu.
    for name, tensor in (
        ("output", output),
        ("hidden_states", hidden_states),
        ("weight", weight),
        ("scales", scales),
    ):
        if tensor.device.type != runtime_device.name:
            raise ValueError(
                f"mixed_gemm only supports {runtime_device.name} tensors, "
                f"{name} is on {tensor.device.type}"
            )

    if not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")
    # The packed layout is addressed flat, so the 2-D shape carries no meaning here.
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous")

    return m, n, k


def mixed_gemm(
    output,
    hidden_states,
    weight,
    scales,
    bias,
    num_bits,
    activation_raw,
    *,
    BLOCK_M=None,
    BLOCK_N=None,
    BLOCK_K=None,
    num_warps=None,
    num_stages=None,
):
    """Mixed-precision GEMM with weight-only integer quantization.

    ``output = act(hidden_states @ dequantize(weight) + bias)``, written into ``output`` in
    place; ``hidden_states`` is never quantized.

    Args:
        output (Tensor): ``[m, n]``, same dtype as ``hidden_states``; written in place.
        hidden_states (Tensor): activations ``[m, k]`` (fp16 or bf16).
        weight (Tensor): packed quantized weight, ``[k, ceil(n/2)]`` uint8 for int4 or
            ``[k, n]`` uint8 for int8.
        scales (Tensor): ``[n]`` per-output-channel scales, in the activation dtype.
        bias (Tensor or None): ``[n]`` bias, in the activation dtype.
        num_bits (int): 4 or 8.
        activation_raw (int): an ``ActivationType`` value.

    Returns:
        Tensor: ``output``, updated in place.
    """
    logger.debug("TRAIN MIXED_GEMM")

    m, n, k = _check_launch_arguments(
        output, hidden_states, weight, scales, bias, num_bits, activation_raw
    )

    # A pinned tile means a measurement; only the default schedule is chosen by shape.
    pinned = BLOCK_M is not None or BLOCK_N is not None or BLOCK_K is not None

    BLOCK_M = _DEFAULT_TILE["BLOCK_M"] if BLOCK_M is None else BLOCK_M
    BLOCK_N = _DEFAULT_TILE["BLOCK_N"] if BLOCK_N is None else BLOCK_N
    BLOCK_K = _DEFAULT_TILE["BLOCK_K"] if BLOCK_K is None else BLOCK_K
    num_warps = _DEFAULT_TILE["num_warps"] if num_warps is None else num_warps
    num_stages = _DEFAULT_TILE["num_stages"] if num_stages is None else num_stages

    has_bias = bias is not None
    # Only read when HAS_BIAS, so the output doubles as the stand-in pointer.
    bias_in = bias if has_bias else output

    # muBLAS carries the GEMM; the unpack kernel needs the layout's divisibility rules.
    if not pinned and n % _DQ_BLOCK_N == 0 and k % 64 == 0:
        return _mixed_gemm_split(
            output,
            hidden_states,
            weight,
            scales,
            bias_in,
            has_bias,
            num_bits,
            activation_raw,
            m,
            n,
            k,
        )

    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))
    with torch_device_fn.device(hidden_states.device):
        mixed_gemm_kernel[grid](
            output,
            hidden_states,
            weight,
            scales,
            bias_in,
            m,
            n,
            k,
            hidden_states.stride(0),
            hidden_states.stride(1),
            output.stride(0),
            output.stride(1),
            HAS_BIAS=has_bias,
            ACT_TYPE=activation_raw,
            NUM_BITS=num_bits,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return output
