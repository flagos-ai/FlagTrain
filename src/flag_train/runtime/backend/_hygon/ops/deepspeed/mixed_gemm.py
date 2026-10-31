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
"""Hygon ``mixed_gemm``: dequantize in a kernel, then hand the GEMM to hipBLAS.

The generic implementation is :mod:`flag_train.deepspeed.mixed_gemm` -- what every
backend without an override runs, and a faithful port of DeepSpeed's single fused
kernel. This file keeps that operator's *contract* (the same seven positional
arguments, the same input checks, the same epilogue) and replaces its *kernel* with
two steps: dequantize the whole weight into a staging buffer, then call the vendor
BLAS GEMM on it.

Everything shared is imported from the generic module so the two cannot drift: the
activation codes, and ``_check_launch_arguments`` -- the input checks are one
implementation used by both, so this backend cannot accept a call the generic one
rejects.

Why
---

On the target part the fused kernel's weight fetch is what costs. Relative to the
torch reference the harness measures against, ``reference / this implementation``:

    shape (m, k, n)        int4    int8
    1 x 4096 x 4096        2.75    2.68
    64 x 4096 x 4096       2.61    2.55
    1024 cubed             1.21    1.23
    2048 cubed             1.20    1.21
    4096 cubed             1.32    1.32
    8192 x 4096 x 4096     1.19    1.19
    2048 x 4096 x 16384    1.67    1.62

(median of 100 reps after 1 s of warmup; the same run measured the generic fused
kernel at 0.64x, 0.63x and 0.31x on the first, second and fifth row, and at 0.31x
on 4096 cubed -- 2.307 ms against the reference's 0.735 ms.)

Two thirds of what this route wins is the reference's own dequantization: it is
``hidden @ (q.to(fp16) * scales)``, three kernels, and at 4096 cubed the two
elementwise ones alone are 0.244 ms of the 0.735 ms -- they materialize two
full-size fp16 weights to feed a GEMM with one. Reading the *packed* bytes and
writing the staging buffer once, in a single Triton kernel, costs 0.066 ms for that
same weight. The GEMM that follows is the same vendor BLAS call the reference makes
(0.492 ms, 279 TFLOP/s at 4096 cubed), so the two-step total is 0.558 ms.

That is the whole argument: this file does not port anything DeepSpeed does not
already do -- it removes the gather from the inner loop and pays one staging buffer
for it. It is deliberately *not* an attempt to make the fused kernel faster.

What is not claimed: why the fused kernel is this far behind here. Its tile is
bounded by this part's 64 KB workgroup LDS (100 KB on Ada, 227 KB on Hopper), and
the tile that fits is small enough that the gather dominates; but tile ranking on
this part is not monotone in that budget, so the LDS limit is a candidate
explanation, not an established one.

Dequantization tile
-------------------

``BLOCK_K``/``BLOCK_N``/``num_warps`` select the *dequantization* kernel's tile,
with the meaning they have in the generic operator. Measured at 4096 cubed:

    (BLOCK_K, BLOCK_N, num_warps)   int4     int8
    8, 256, 4                       0.0848   0.0867
    16, 256, 4                      0.0734   0.0686
    32, 256, 4                      0.0662   0.0696
    64, 128, 8                      0.0720   0.0776

so the default is per bit width. The kernel reads and writes the whole weight once:
41.9 MB in 0.066 ms (int4), 50.3 MB in 0.069 ms (int8).

Numerics
--------

The scale is applied inside the dequantization kernel and the product is stored in
the activation dtype, which rounds it once more than the fused kernel does: that one
keeps ``q - 2**(bits-1)`` exact in the activation dtype and multiplies the fp32
accumulator by the scale after the reduction. Applying it up front is what keeps the
staging buffer a single pass, and it is the order of operations the torch reference
uses as well. Against the fused kernel on the same packed buffer the two agree to a
relative 3e-4 - 9e-4, which is fp16 rounding of the dequantized weight, and the
dequantization error bound the test asserts (``scale[n] * sum_k |A[m,k]|``) is
unaffected: it is a bound on the *quantization* error, which is far larger.

Epilogue
--------

Bias and activation need one more pass over ``output`` (the GEMM already wrote it),
because the staging buffer removed the only place where the epilogue could be fused
into the reduction. That pass is skipped entirely when there is no bias and the
activation is IDENTITY, which is the case the harness measures; when it does run it
is one read-modify-write of an ``m x n`` matrix, small next to the dequantization of
an ``k x n`` one. It necessarily reads back the stored output, so where the fused
kernel applies the epilogue to its fp32 accumulator this one applies it to a value
already rounded to the activation dtype -- one rounding's worth of error (~5e-4
relative in fp16) in the bias-and-activation path, and none at all in the path the
harness measures.

Like the generic implementation, this file is only a *definition*: the operator is
wired up by name from the vendor package's ``__init__`` (see
``flag_train.runtime.backend``), and nothing imports it by hand.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from flag_train.deepspeed.mixed_gemm import (
    GELU,
    IDENTITY,
    RELU,
    SILU,
    _check_launch_arguments,
)
from flag_train.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# (BLOCK_K, BLOCK_N, num_warps) of the dequantization kernel, per bit width; see the
# module docstring for the sweep these come from.
_DEQUANT_TILE = {
    4: {"BLOCK_K": 32, "BLOCK_N": 256, "num_warps": 4},
    8: {"BLOCK_K": 16, "BLOCK_N": 256, "num_warps": 4},
}
# The epilogue is elementwise, so a flat tile; 1024 elements is 4 KB per lane group
# at NUM_WARPS = 4, enough to keep the loads coalesced without an oversized grid.
_EPILOGUE_BLOCK = 1024
_EPILOGUE_WARPS = 4


@triton.jit
def mixed_gemm_dequant_kernel(
    weight_ptr,
    scales_ptr,
    out_ptr,
    k,
    n,
    stride_ok,
    stride_on,
    NUM_BITS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Write the logical, dequantized weight ``(q - 2**(bits-1)) * scale[n]``.

    This is the generic kernel's operand fetch with the GEMM taken out of it, and the
    index arithmetic is that kernel's, unchanged: it maps a logical ``(krow, ncol)``
    onto the byte and the nibble DeepSpeed's preprocessing stored it in. What changes
    is the *shape* of the iteration -- a tile of the weight is visited once instead of
    once per output row -- and that the scale is applied here rather than to the
    accumulator, since ``scales`` carries no ``k``.

    Args:
        weight_ptr: the packed weight, addressed flat, exactly as the GEMM kernel
            addresses it.
        scales_ptr: per-output-channel scales ``[n]``, in the activation dtype.
        out_ptr: staging buffer ``[k, n]`` in the activation dtype, written in full.
        k, n: the *logical* weight shape; ``out_ptr`` is dense, so the tile is masked
            by the logical shape and not by the packed buffer's.
        stride_ok, stride_on: strides of ``out_ptr``.
        NUM_BITS: 4 or 8.
        BLOCK_K, BLOCK_N: tile sizes.
    """
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    koff = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (koff < k)[:, None] & (cols < n)[None, :]

    # Layout constants, straight from `mixed_gemm_B_layout.h` (see the generic
    # module). The column side depends only on `cols` and the row side only on
    # `koff`, so both are computed once for the tile rather than per element.
    EIR = 32 // NUM_BITS  # elements per 32-bit register
    INTER = 16 // NUM_BITS  # columns_interleaved: 4 (int4), 2 (int8)
    VPT = 2 * NUM_BITS  # rows_per_column_tile // EIR
    BROWS = 8 * (16 // NUM_BITS)  # B_ROWS_PER_MMA

    nvr = k // EIR  # num_vec_rows
    col_part = (cols // INTER) * (nvr * INTER) + VPT * (cols % INTER)

    t = koff % BROWS
    k1 = (koff // BROWS) * BROWS + ((t % 8) // 2) * EIR + 2 * (t // 8) + (t % 2)
    kr = k1 // EIR
    ks = k1 % EIR
    ks4 = (EIR // 2) * (ks % 2) + ks // 2  # stage 4 register reorder
    row_part = INTER * (kr // VPT) * VPT + (kr % VPT)

    slot = (col_part[None, :] + row_part[:, None]) * EIR + ks4[:, None]

    if NUM_BITS == 8:
        wq = tl.load(weight_ptr + slot, mask=mask, other=128)
    else:
        # Two elements share a byte; which half is the *slot's* parity, not the
        # column's. 0x88 is the pad value, and it dequantizes to zero.
        packed = tl.load(weight_ptr + slot // 2, mask=mask, other=0x88)
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        wq = tl.where((slot % 2) == 1, hi, lo)

    sc = tl.load(scales_ptr + cols, mask=cols < n, other=0.0)
    w = (wq.to(tl.float32) - (1 << (NUM_BITS - 1))) * sc[None, :]

    tl.store(
        out_ptr + koff[:, None] * stride_ok + cols[None, :] * stride_on,
        w.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def mixed_gemm_epilogue_kernel(
    output_ptr,
    bias_ptr,
    total,
    n,
    HAS_BIAS: tl.constexpr,
    ACT_TYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Add the bias and apply the activation to ``output``, in place.

    The expressions are the generic kernel's, so the two epilogues cannot disagree.
    The row is only needed to find the column of the bias, which is why it is not
    computed when there is none.

    Args:
        output_ptr: the GEMM's output ``[m, n]``, updated in place.
        bias_ptr: ``[n]`` bias. Read only when ``HAS_BIAS``; the output doubles as
            the stand-in pointer otherwise.
        total, n: ``m * n`` and ``n``.
        HAS_BIAS: whether ``bias_ptr`` is meaningful.
        ACT_TYPE: an ``ActivationType`` value.
        BLOCK: elements per program.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < total

    # Up to fp32 for the epilogue, as the generic kernel's accumulator is: the
    # stored value has already been rounded to the activation dtype, and rounding
    # it a second time per operation is not worth it.
    acc = tl.load(output_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + idx % n, mask=mask, other=0.0)

    if ACT_TYPE == RELU:
        acc = tl.maximum(acc, 0.0)
    elif ACT_TYPE == SILU:
        acc = acc * tl.sigmoid(acc)
    elif ACT_TYPE == GELU:
        c = 0.7978845608028654  # sqrt(2 / pi), tanh approximation of GELU
        acc = 0.5 * acc * (1.0 + libdevice.tanh(c * (acc + 0.044715 * acc * acc * acc)))

    tl.store(output_ptr + idx, acc.to(output_ptr.dtype.element_ty), mask=mask)


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

    The contract, the argument checks and the epilogue are the generic operator's;
    the kernel is not. See the module docstring.

    Args:
        output (Tensor): ``[m, n]``, same dtype as ``hidden_states``; written in
            place.
        hidden_states (Tensor): activations ``[m, k]`` (fp16 or bf16).
        weight (Tensor): packed quantized weight ``[k, ceil(n/2)]`` uint8 for
            int4 or ``[k, n]`` uint8 for int8. Already quantized and packed.
        scales (Tensor): ``[n]`` per-output-channel scales, in the activation
            dtype.
        bias (Tensor or None): ``[n]`` bias, in the activation dtype. May be
            ``None``.
        num_bits (int): 4 or 8.
        activation_raw (int): an ``ActivationType`` value. See the module
            constants.

    Keyword Args:
        BLOCK_K, BLOCK_N, num_warps: the dequantization kernel's tile, defaulted to
            the measured value for ``num_bits``.
        BLOCK_M, num_stages: ignored. There is no output-row tile here and nothing
            is pipelined; they are accepted so that a caller can pass one set of
            overrides to any backend.

    Returns:
        Tensor: ``output``, updated in place.
    """
    logger.debug("TRAIN MIXED_GEMM (dequantize + BLAS)")

    m, n, k = _check_launch_arguments(
        output, hidden_states, weight, scales, bias, num_bits, activation_raw
    )

    tile = _DEQUANT_TILE[num_bits]
    BLOCK_K = tile["BLOCK_K"] if BLOCK_K is None else BLOCK_K
    BLOCK_N = tile["BLOCK_N"] if BLOCK_N is None else BLOCK_N
    num_warps = tile["num_warps"] if num_warps is None else num_warps

    # The staging buffer is what this route trades for the fused fetch. It is the
    # size of the *logical* weight in the activation dtype -- one rank of the
    # two-thirds the reference materializes, and only half of the activation
    # dtype's byte count for int4's packed input.
    dequantized = torch.empty(
        (k, n), dtype=hidden_states.dtype, device=hidden_states.device
    )

    with torch_device_fn.device(hidden_states.device):
        mixed_gemm_dequant_kernel[(triton.cdiv(k, BLOCK_K), triton.cdiv(n, BLOCK_N))](
            weight,
            scales,
            dequantized,
            k,
            n,
            dequantized.stride(0),
            dequantized.stride(1),
            NUM_BITS=num_bits,
            BLOCK_K=BLOCK_K,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )

        # `output` is contiguous and of the right shape (checked above), and the
        # staging buffer is in the activation's dtype, so this is the vendor BLAS.
        torch.mm(hidden_states, dequantized, out=output)

        if bias is not None or int(activation_raw) != int(IDENTITY):
            # Only read when HAS_BIAS, so the output doubles as the stand-in
            # pointer and no dummy tensor is allocated -- as in the generic
            # launcher.
            mixed_gemm_epilogue_kernel[(triton.cdiv(m * n, _EPILOGUE_BLOCK),)](
                output,
                bias if bias is not None else output,
                m * n,
                n,
                HAS_BIAS=bias is not None,
                ACT_TYPE=activation_raw,
                BLOCK=_EPILOGUE_BLOCK,
                num_warps=_EPILOGUE_WARPS,
            )

    return output
