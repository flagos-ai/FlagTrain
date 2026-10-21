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

Port of DeepSpeed's ``mixed_gemm`` (``csrc/quantization/mixed_gemm.cpp`` and
``fastertransformer``'s ``CutlassFpAIntBGemmRunner``). The activations stay in
their input dtype while the weights are int4/int8 with a per-output-channel
scale; the dequantization is folded into the GEMM's operand fetch.

The operator contract is taken verbatim from DeepSpeed's launcher: ``weight`` has
shape ``[k, n]`` (checked by ``TORCH_CHECK(weight.size(0) == k)``), ``scales`` has
shape ``[n]``, and ``weight`` arrives already quantized and packed -- this operator
performs no quantization. Producing that packed weight is DeepSpeed's offline
``symmetric_quantize`` + ``preprocess_weights_for_mixed_gemm`` preprocessing, which
is a separate concern and is not part of this port.

Weight layout
-------------

``weight`` is what ``preprocess_weights_for_mixed_gemm`` produces, so this
operator can be dropped in wherever DeepSpeed's ``mixed_gemm`` is used. That
layout is pinned by ``mixed_gemm_B_layout.h``: ``ColumnMajorTileInterleave`` with
``ThreadblockK = 64`` and ``ColumnsInterleaved = (128*8/bits)/64`` (4 for int4,
2 for int8), followed by four transforms in order -- row permutation, 64x64
sub-byte transpose, column interleave, then ``+8``/``+128`` with an
intra-register reorder.

The kernel recomputes the resulting address arithmetically rather than walking
those four transforms. Each logical ``(k, n)`` maps to an element *slot*, whose
parity picks the nibble for int4. Because the layout is a permutation of the
weight's bytes, a 2-D shape no longer describes it, so ``weight`` is addressed
flat and must be contiguous.

Deviations from DeepSpeed
-------------------------

1. **Gated activations.** ``ActivationType`` also declares ``GEGLU``/``ReGLU``/
   ``SiGLU``, but DeepSpeed's ``fpA_intB`` runner rejects them at runtime
   ("Fused gated activations not supported"), so this port rejects them too.

2. **Activation dispatch.** DeepSpeed selects the epilogue at C++ template
   instantiation time. This port makes the same choice at kernel-compile time via
   a ``tl.constexpr``, which keeps one kernel instead of four.

This is the generic implementation. A backend with a specialised kernel replaces
this *name* through ``runtime.backend.SpecOpRegistrar`` when
``flag_train.deepspeed`` is imported, so that is where to import the operator
from -- importing it from this module gets the generic kernel on every backend.
"""

import logging

import triton
import triton.language as tl
from triton.language.extra import libdevice

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# ``ActivationType`` values, verbatim from DeepSpeed's ``utils/activation_type.h``.
# ``IDENTITY`` is 6, not 0. ``tl.constexpr`` so the kernel can read them; host code
# wraps them back with ``int(...)``.
GELU = tl.constexpr(0)
RELU = tl.constexpr(1)
SILU = tl.constexpr(2)
GEGLU = tl.constexpr(3)
ReGLU = tl.constexpr(4)
SiGLU = tl.constexpr(5)
IDENTITY = tl.constexpr(6)
INVALID_TYPE = tl.constexpr(-1)

# The runner supports exactly these four; ``fpA_intB_gemm_template.h`` switches
# over them and rejects everything else.
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

    Mirrors ``CutlassFpAIntBGemmRunner::gemm_bias_act``: the weight tile is
    dequantized as it is fetched (``w = (q - 2**(bits-1)) * scale[n]``), the
    product accumulates in fp32 through the tensor cores, and the epilogue adds
    the bias and applies the activation before storing in the output dtype.

    Args:
        output_ptr: output ``[m, n]``, same dtype as ``hidden_states``.
        hidden_states_ptr: activations ``[m, k]`` (fp16 or bf16). Never quantized.
        weight_ptr: quantized weight in DeepSpeed's preprocessed layout, as a flat
            uint8 buffer of ``k * ceil(n/2)`` bytes for int4 or ``k * n`` for int8.
        scales_ptr: per-output-channel scales ``[n]``, in the activation dtype.
        bias_ptr: optional ``[n]`` bias, in the activation dtype. Only
            dereferenced when ``HAS_BIAS``.
        m, n, k: problem dimensions.
        stride_hm, stride_hk: row/column strides of ``hidden_states``.
        stride_om, stride_on: row/column strides of ``output``.
        HAS_BIAS: whether ``bias_ptr`` is meaningful.
        ACT_TYPE: an ``ActivationType`` value; see the module constants.
        NUM_BITS: 4 or 8.
        BLOCK_M, BLOCK_N, BLOCK_K: tile sizes.

    ``scale[n]`` carries no ``k``, so it is hoisted out of the reduction:
    ``sum_k a[m,k]*q[k,n]*s[n] == s[n] * sum_k a[m,k]*q[k,n]``. That is why one
    scale load outside the K loop suffices.
    """
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rows_mask = rows < m

    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    cols_mask = cols < n

    sc = tl.load(scales_ptr + cols, mask=cols_mask, other=0.0)

    # Layout constants, straight from `mixed_gemm_B_layout.h`:
    #   ThreadblockK = 64, ElementsPerCacheLine = 128*8/bits,
    #   ColumnsInterleaved = ElementsPerCacheLine / ThreadblockK.
    EIR = 32 // NUM_BITS  # elements per 32-bit register
    INTER = 16 // NUM_BITS  # columns_interleaved: 4 (int4), 2 (int8)
    VPT = 2 * NUM_BITS  # rows_per_column_tile // EIR
    BROWS = 8 * (16 // NUM_BITS)  # B_ROWS_PER_MMA

    # Column side of stages 2-3. `k` is the reduction length, which is the R of
    # the [R, C] = [k, n] weight.
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

        # Row side of stages 1-3. Stage 1 (permute_B_rows) is *inverted*: the C++
        # copies input[read_row] into output[write_row], so a row's content moves
        # to f^-1(row). Using f directly still works for int4, where f happens to
        # be an involution, and is wrong for int8.
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
            # Two elements share a byte. Which half is decided by the *slot's*
            # parity, not by the column. The 0x88 pad dequantizes to
            # (8 - 8) * scale == 0; a pad of 0 would give -8 * scale instead.
            packed = tl.load(weight_ptr + slot // 2, mask=wmask, other=0x88)
            lo = packed & 0x0F
            hi = (packed >> 4) & 0x0F
            wq = tl.where((slot % 2) == 1, hi, lo)

        w = wq.to(hidden_states_ptr.dtype.element_ty) - (1 << (NUM_BITS - 1))

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
        acc = 0.5 * acc * (1.0 + libdevice.tanh(c * (acc + 0.044715 * acc * acc * acc)))

    tl.store(
        output_ptr + rows[:, None] * stride_om + cols[None, :] * stride_on,
        acc.to(output_ptr.dtype.element_ty),
        mask=rows_mask[:, None] & cols_mask[None, :],
    )


# Tile sizes, warp count and pipeline depth. These decide the kernel's shared-memory
# footprint and its arithmetic intensity, and they are the operator's only tuning knobs.
#
# Measured on the target part with the benchmark framework's own settings rather than
# taken from a default. `BLOCK_N` dominates: the weight fetch is a gather whose volume
# scales with N -- every column of the tile reads its own set of cache lines -- so halving
# N halves the gather, and `BLOCK_M` is then raised to put the arithmetic intensity back.
# Ratio to the torch reference baseline, int4, on an H20 (sm_90, 78 SMs):
#
#     BM/BN/BK     1024 cubed  2048 cubed  4096 cubed  8192x4096x4096  2048x4096x16384
#     128/64/32        0.715       0.575       0.796            0.516            0.543
#     256/64/32        0.576       0.716       0.886            0.806            0.601
#     256/32/32        0.794       0.760       0.920            0.861            0.911
#
# 256/32/32 wins 12 of the 14 benchmarked shapes and averages +0.139 over 128/64/32. The
# two it does not win are the thinnest (m = 1 and m = 64), where the weight fetch is the
# whole cost and a larger BLOCK_M only means more masked-off rows.
#
# ``num_stages`` and ``num_warps`` make no measurable difference at this configuration:
# 2, 3 and 4 stages and 4 or 8 warps all land within 0.5% of one another.
_DEFAULT_TILE = {
    "BLOCK_M": 256,
    "BLOCK_N": 32,
    "BLOCK_K": 32,
    "num_warps": 4,
    "num_stages": 3,
}


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

    Faithful port of DeepSpeed's ``mixed_gemm`` launcher. Computes
    ``output = act(hidden_states @ dequantize(weight) + bias)`` writing into
    ``output`` in place; ``hidden_states`` is never quantized, which is what makes
    this a weight-only scheme.

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

    Returns:
        Tensor: ``output``, updated in place.

    The checks below reproduce the launcher's ``TORCH_CHECK``s so a caller gets
    the same diagnosis the reference gives rather than a failure inside the Triton
    compiler or the driver. The device check is the one addition: upstream is
    CUDA-only and so has no other device to name.
    """
    logger.debug("TRAIN MIXED_GEMM")

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

    # gemm_bias_act rejects anything outside the four epilogues it instantiates;
    # the gated ones get their own message (fpA_intB_gemm_template.h:527-532).
    if activation_raw not in _SUPPORTED_ACTIVATIONS:
        if activation_raw in _GATED_ACTIVATIONS:
            raise ValueError("Fused gated activations not supported")
        raise ValueError(f"Invalid activation type: {activation_raw}")

    # Against the runtime's name rather than a hard-coded "cuda": the same source
    # runs on npu and hcu. A CPU tensor is still rejected, since the kernels would
    # address memory the accelerator cannot see.
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
    # The preprocessed layout is a permutation of the weight's *bytes*, addressed
    # flat, so the tensor's own 2-D shape carries no meaning here.
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous")

    # Keyword-only and defaulted, so the seven-argument call every caller uses still
    # gets one fixed configuration. Overriding them is for measuring on a new part, not
    # something a caller should need.
    BLOCK_M = _DEFAULT_TILE["BLOCK_M"] if BLOCK_M is None else BLOCK_M
    BLOCK_N = _DEFAULT_TILE["BLOCK_N"] if BLOCK_N is None else BLOCK_N
    BLOCK_K = _DEFAULT_TILE["BLOCK_K"] if BLOCK_K is None else BLOCK_K
    num_warps = _DEFAULT_TILE["num_warps"] if num_warps is None else num_warps
    num_stages = _DEFAULT_TILE["num_stages"] if num_stages is None else num_stages

    has_bias = bias is not None
    # Only read when HAS_BIAS, so the output doubles as the stand-in pointer and
    # no dummy tensor is allocated.
    bias_in = bias if has_bias else output

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
