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
"""Ascend ``mixed_gemm``: decode the weight once, then hand the GEMM to CANN.

The fused DeepSpeed kernel is unusable on this part: its operand fetch compiles to
per-element addressing and takes 2315 ms at 4096 cubed. This override writes the logical,
scaled weight into a buffer with a Triton kernel and gives the GEMM to ``torch.mm``. The
decode depends on the weight and the scales alone, so it is cached per weight (below): a
repeated weight skips it, a weight seen for the first time pays it.
"""

import logging
import threading
from collections import OrderedDict

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

# Dequantization tile per bit width. BLOCK_N stays 128: a 64 x 128 tile overflows
# this part's unified buffer.
_DEQUANT_TILE = {
    4: {"BLOCK_K": 32, "BLOCK_N": 128, "num_warps": 4},
    8: {"BLOCK_K": 16, "BLOCK_N": 128, "num_warps": 4},
}
# Flat tile for the elementwise epilogue.
_EPILOGUE_BLOCK = 1024
_EPILOGUE_WARPS = 4

# Decoded weight per (weight, scales, num_bits, dtype), oldest first. The key is
# data_ptr and _version with the tensors themselves held in the entry, so a cached
# address cannot be handed to another tensor.
_DECODE_CACHE = OrderedDict()
_DECODE_CACHE_LOCK = threading.Lock()
_DECODE_CACHE_MAX_BYTES = 256 << 20
_DECODE_CACHE_MAX_ENTRIES = 8


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
    """Write the logical, scaled weight ``(q - 2**(bits-1)) * scale[n]``.

    The index arithmetic is the generic kernel's operand fetch; the scale is applied
    here because ``scales`` carries no ``k``.
    """
    # Flat 1-D grid over the (row block, column block) space.
    pid = tl.program_id(0)
    num_n = (n + BLOCK_N - 1) // BLOCK_N
    pid_k = pid // num_n
    pid_n = pid % num_n

    koff = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (koff < k)[:, None] & (cols < n)[None, :]

    # Layout constants from mixed_gemm_B_layout.h (see the generic module).
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
        # Two elements share a byte, picked by the slot's parity; 0x88 pads to zero.
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

    The expressions are the generic kernel's, so the two cannot disagree.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < total

    # Up to fp32, as the generic kernel's accumulator is.
    acc = tl.load(output_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + idx % n, mask=mask, other=0.0)

    if ACT_TYPE == RELU:
        acc = tl.maximum(acc, 0.0)
    elif ACT_TYPE == SILU:
        acc = acc * tl.sigmoid(acc)
    elif ACT_TYPE == GELU:
        c = 0.7978845608028654  # sqrt(2 / pi), tanh approximation of GELU
        acc = (
            0.5
            * acc
            * (1.0 + tl_extra_shim.tanh(c * (acc + 0.044715 * acc * acc * acc)))
        )

    tl.store(output_ptr + idx, acc.to(output_ptr.dtype.element_ty), mask=mask)


def _cached_bytes(cache):
    """Bytes of decoded weight held by ``cache``."""
    return sum(
        decoded.numel() * decoded.element_size() for _, _, decoded in cache.values()
    )


def _decoded_weight(weight, scales, num_bits, k, n, dtype, BLOCK_K, BLOCK_N, num_warps):
    """The logical, scaled weight in ``dtype``, decoded on first use.

    The result is shared with every later call, so it is not the caller's to write to.
    """
    key = (
        weight.data_ptr(),
        weight._version,
        tuple(weight.shape),
        weight.dtype,
        scales.data_ptr(),
        scales._version,
        tuple(scales.shape),
        scales.dtype,
        num_bits,
        dtype,
    )
    with _DECODE_CACHE_LOCK:
        entry = _DECODE_CACHE.get(key)
        if entry is not None:
            _DECODE_CACHE.move_to_end(key)
            return entry[2]

    decoded = torch.empty((k, n), dtype=dtype, device=weight.device)
    with torch_device_fn.device(weight.device):
        num_k = triton.cdiv(k, BLOCK_K)
        num_n = triton.cdiv(n, BLOCK_N)
        mixed_gemm_dequant_kernel[(num_k * num_n, 1, 1)](
            weight,
            scales,
            decoded,
            k,
            n,
            decoded.stride(0),
            decoded.stride(1),
            NUM_BITS=num_bits,
            BLOCK_K=BLOCK_K,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )

    # Filled outside the lock: a race decodes twice and drops one of the two results.
    with _DECODE_CACHE_LOCK:
        _DECODE_CACHE[key] = (weight, scales, decoded)
        while (
            len(_DECODE_CACHE) > _DECODE_CACHE_MAX_ENTRIES
            or _cached_bytes(_DECODE_CACHE) > _DECODE_CACHE_MAX_BYTES
        ):
            if len(_DECODE_CACHE) == 1:
                break
            _DECODE_CACHE.popitem(last=False)
    return decoded


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

    The contract, the argument checks and the epilogue are the generic operator's; the
    kernel is not. See the module docstring. ``BLOCK_M`` and ``num_stages`` are accepted
    and ignored: there is no output-row tile here and nothing is pipelined.
    """
    logger.debug("TRAIN MIXED_GEMM (dequantize + torch.mm)")

    m, n, k = _check_launch_arguments(
        output, hidden_states, weight, scales, bias, num_bits, activation_raw
    )

    tile = _DEQUANT_TILE[num_bits]
    BLOCK_K = tile["BLOCK_K"] if BLOCK_K is None else BLOCK_K
    BLOCK_N = tile["BLOCK_N"] if BLOCK_N is None else BLOCK_N
    num_warps = tile["num_warps"] if num_warps is None else num_warps

    # Staging buffer: the logical weight in the activation dtype, decoded once per weight.
    dequantized = _decoded_weight(
        weight,
        scales,
        num_bits,
        k,
        n,
        hidden_states.dtype,
        BLOCK_K,
        BLOCK_N,
        num_warps,
    )

    with torch_device_fn.device(hidden_states.device):
        # ``output`` is contiguous and correctly shaped (checked above); CANN's own GEMM.
        torch.mm(hidden_states, dequantized, out=output)

        if bias is not None or int(activation_raw) != int(IDENTITY):
            # Read only when HAS_BIAS, so the output doubles as the stand-in pointer.
            mixed_gemm_epilogue_kernel[(triton.cdiv(m * n, _EPILOGUE_BLOCK), 1, 1)](
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
