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
"""Iluvatar ``mixed_gemm``: one Triton decode pass, then the vendor BLAS.

The generic module is a port of DeepSpeed's single fused kernel. Here the fused form is not
competitive: Triton 3.1 in CoreX 4.4.0 aborts its pipelined shared-to-dot-operand copy for
every ``num_stages > 1``, and unpipelined it reaches 23-28 TFLOPS against the vendor BLAS's
94 -- its gather strides 32 bytes per lane along ``n``.

So the packing stays in Triton, with the lanes along ``k`` (contiguous bytes), and the GEMM
goes to the vendor BLAS; bias and activation follow. Nothing is cached: one decode, one GEMM
and at most one pointwise pass per call, the same work the benchmark's dequantize-then-BLAS
baseline does minus one round trip. The price is a temporary ``[n, k]`` buffer.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_train.deepspeed.mixed_gemm import (
    GELU,
    IDENTITY,
    RELU,
    SILU,
    _check_launch_arguments,
)
from flag_train.runtime import torch_device_fn
from flag_train.utils import tl_extra_shim

logger = logging.getLogger(__name__)

# Largest decode tile tried on the widest benchmarked shape won (16384x4096: 2.04 / 0.88 /
# 0.67 / 0.52 ms for 128 / 256 / 512 / 1024).
_BLOCK_K = 1024
_ACT_BLOCK = 1024


@triton.jit
def dequantize_kernel(
    weight_ptr,
    scales_ptr,
    dense_ptr,
    k,
    EIR: tl.constexpr,
    INTER: tl.constexpr,
    VPT: tl.constexpr,
    BROWS: tl.constexpr,
    NUM_BITS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Packed weight -> dense ``[n, k]``, scaled, one program per column."""
    pid_k = tl.program_id(0)
    col = tl.program_id(1)

    # Layout constants from ``mixed_gemm_B_layout.h``.
    nvr = k // EIR  # num_vec_rows
    col_part = (col // INTER) * (nvr * INTER) + VPT * (col % INTER)

    koff = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    # Stage 1 (permute_B_rows) is *inverted*: using f directly works for int4, where f is
    # an involution, and is wrong for int8.
    t = koff % BROWS
    k1 = (koff // BROWS) * BROWS + ((t % 8) // 2) * EIR + 2 * (t // 8) + (t % 2)
    kr = k1 // EIR
    ks = k1 % EIR
    ks4 = (EIR // 2) * (ks % 2) + ks // 2  # stage 4 register reorder
    row_part = INTER * (kr // VPT) * VPT + (kr % VPT)

    slot = (col_part + row_part) * EIR + ks4
    mask = koff < k

    if NUM_BITS == 8:
        val = tl.load(weight_ptr + slot, mask=mask, other=128).to(tl.int16) - 128
    else:
        # Slot parity picks the nibble; the 0x88 pad dequantizes to zero.
        packed = tl.load(weight_ptr + slot // 2, mask=mask, other=0x88)
        nib = tl.where((slot % 2) == 1, packed >> 4, packed & 0x0F)
        val = nib.to(tl.int16) - 8

    sc = tl.load(scales_ptr + col).to(tl.float32)
    tl.store(
        dense_ptr + col * k + koff,
        (val.to(tl.float32) * sc).to(dense_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def activation_kernel(output_ptr, numel, ACT_TYPE: tl.constexpr, BLOCK: tl.constexpr):
    """Pointwise epilogue; same formulas as the generic kernel."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(output_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    if ACT_TYPE == RELU:
        x = tl.maximum(x, 0.0)
    elif ACT_TYPE == SILU:
        x = x * tl.sigmoid(x)
    elif ACT_TYPE == GELU:
        c = 0.7978845608028654  # sqrt(2 / pi)
        z = c * (x + 0.044715 * x * x * x)
        x = 0.5 * x * (1.0 + tl_extra_shim.tanh(z))

    tl.store(output_ptr + offs, x.to(output_ptr.dtype.element_ty), mask=mask)


def _decode_dense(weight, scales, k, n, num_bits, dtype, block_k):
    dense = torch.empty((n, k), dtype=dtype, device=weight.device)
    grid = (triton.cdiv(k, block_k), n)
    with torch_device_fn.device(weight.device):
        dequantize_kernel[grid](
            weight,
            scales,
            dense,
            k,
            EIR=32 // num_bits,
            INTER=16 // num_bits,
            VPT=2 * num_bits,
            BROWS=8 * (16 // num_bits),
            NUM_BITS=num_bits,
            BLOCK_K=block_k,
        )
    return dense


def _apply_activation(output, activation_raw):
    numel = output.numel()
    grid = (triton.cdiv(numel, _ACT_BLOCK),)
    with torch_device_fn.device(output.device):
        activation_kernel[grid](
            output, numel, ACT_TYPE=activation_raw, BLOCK=_ACT_BLOCK
        )


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

    # BLOCK_K is the decode tile; the other keywords tuned the fused kernel this backend
    # does not run, and the vendor BLAS picks its own tiling.
    block_k = _BLOCK_K if BLOCK_K is None else BLOCK_K

    dense = _decode_dense(weight, scales, k, n, num_bits, hidden_states.dtype, block_k)
    torch.mm(hidden_states, dense.t(), out=output)

    if bias is not None:
        output.add_(bias)
    if activation_raw != int(IDENTITY):
        _apply_activation(output, activation_raw)

    return output
