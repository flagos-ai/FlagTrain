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
"""T-Head PPU ``mixed_gemm``: two schedules, chosen by shape.

Contract, checks and epilogue are the generic module's; only the launch differs.

* ``m >= 256``: unpack the weight once into a dense buffer, then a plain GEMM. The
  fused kernel re-unpacks the weight in every m-tile. The buffer is per call, never
  cached.
* otherwise: the fused run-based kernel, which serves small m where a full unpack
  pass would dominate.

PPU specifics: ``tl.program_id`` instead of ``tle.program_id`` (the extension upcasts
to int64), and ``BLOCK_N=128`` for the fused kernel.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_train.deepspeed.mixed_gemm import (
    GELU,
    RELU,
    SILU,
    _DEFAULT_TILE,
    _check_launch_arguments,
)
from flag_train.deepspeed.mixed_gemm import mixed_gemm as _generic_mixed_gemm
from flag_train.runtime import torch_device_fn
from flag_train.utils import tl_extra_shim

logger = logging.getLogger(__name__)

# Threshold where the unpack pass starts to pay off.
_SPLIT_MIN_M = 256
# BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages.
_SPLIT_TILE = (128, 128, 64, 8, 4)
# k-rows of one column group per unpack program: BKB * 64.
_DQ_BKB = 8
# Fused kernel tile for large m.
_WIDE_TILE = {
    "BLOCK_M": 256,
    "BLOCK_N": 128,
    "BLOCK_K": 64,
    "num_warps": 8,
    "num_stages": 3,
}


@triton.jit
def _dequant_kernel(
    weight_ptr,
    out_ptr,
    k,
    n,
    NUM_BITS: tl.constexpr,
    BKB: tl.constexpr,
):
    """Unpack one column group into a dense (k, n) tile.

    Stores ``value - zero_point``, the same integers the fused kernel feeds ``tl.dot``.
    Scales stay out: they belong to the epilogue.
    """
    INTER: tl.constexpr = 16 // NUM_BITS
    RUN: tl.constexpr = 128

    grp = tl.program_id(0)
    kb = tl.program_id(1)

    rb = tl.arange(0, BKB * RUN)
    b = tl.load(weight_ptr + grp * (2 * k) + kb * (BKB * RUN) + rb)
    # Same permutation as the fused kernel, without the group axis.
    if NUM_BITS == 4:
        lo = b & 0x0F
        hi = (b >> 4) & 0x0F
        v = tl.reshape(tl.join(lo, hi), (BKB, INTER, 2, 8, 4))
        v = tl.permute(v, (1, 0, 2, 4, 3))
    else:
        v = tl.reshape(b, (BKB, INTER, 2, 2, 8, 2))
        v = tl.permute(v, (1, 0, 2, 3, 5, 4))

    w = tl.reshape(v, (INTER, BKB * 64)).to(out_ptr.dtype.element_ty)
    w = w - (1 << (NUM_BITS - 1))

    kk = kb * (BKB * 64) + tl.arange(0, BKB * 64)
    cols = grp * INTER + tl.arange(0, INTER)
    tl.store(out_ptr + kk[None, :] * n + cols[:, None], w)


@triton.jit
def _split_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    scales_ptr,
    bias_ptr,
    m,
    n,
    k,
    stride_am,
    stride_bk,
    stride_cm,
    HAS_BIAS: tl.constexpr,
    ACT_TYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Dense GEMM over the unpacked weight, with the operator's epilogue."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rows_mask = rows < m
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_ptrs = a_ptr + rows[:, None] * stride_am + rk[None, :]
    b_ptrs = b_ptr + rk[:, None] * stride_bk + cols[None, :]
    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(a_ptrs, mask=rows_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_bk

    acc *= tl.load(scales_ptr + cols)[None, :]
    if HAS_BIAS:
        acc += tl.load(bias_ptr + cols)[None, :]

    if ACT_TYPE == RELU:
        acc = tl.maximum(acc, 0.0)
    elif ACT_TYPE == SILU:
        acc = acc * tl.sigmoid(acc)
    elif ACT_TYPE == GELU:
        # sqrt(2 / pi), tanh approximation
        c = 0.7978845608028654
        z = c * (acc + 0.044715 * acc * acc * acc)
        acc = 0.5 * acc * (1.0 + tl_extra_shim.tanh(z))

    tl.store(
        c_ptr + rows[:, None] * stride_cm + cols[None, :],
        acc.to(c_ptr.dtype.element_ty),
        mask=rows_mask[:, None],
    )


@triton.jit
def mixed_gemm_flat_kernel(
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
    """Fused run-based kernel, reading whole contiguous weight runs.

    Unmasked: the caller guarantees ``n % BLOCK_N == 0`` and ``k % BLOCK_K == 0``.
    Other shapes go to the generic gather kernel.
    """
    INTER: tl.constexpr = 16 // NUM_BITS  # columns_interleaved: 4 (int4), 2 (int8)
    BKB: tl.constexpr = BLOCK_K // 64  # runs per k-block
    NG: tl.constexpr = BLOCK_N // INTER  # column groups per tile
    RUN: tl.constexpr = 128  # bytes per run

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rows_mask = rows < m
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    sc = tl.load(scales_ptr + cols)

    # A group is INTER columns wide and 2*k bytes long, for either bit width.
    grp = pid_n * NG + tl.arange(0, NG)
    wbase = grp[:, None] * (2 * k)
    rb = tl.arange(0, BKB * RUN)

    for kb in range(0, tl.cdiv(k, BLOCK_K)):
        koff = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        hidden = tl.load(
            hidden_states_ptr + rows[:, None] * stride_hm + koff[None, :] * stride_hk,
            mask=rows_mask[:, None],
            other=0.0,
        )

        b = tl.load(weight_ptr + wbase + kb * (BKB * RUN) + rb[None, :])
        # Within a run only the last two axes of the k-order swap.
        if NUM_BITS == 4:
            lo = b & 0x0F
            hi = (b >> 4) & 0x0F
            v = tl.reshape(tl.join(lo, hi), (NG, BKB, INTER, 2, 8, 4))
            v = tl.permute(v, (0, 2, 1, 3, 5, 4))
        else:
            v = tl.reshape(b, (NG, BKB, INTER, 2, 2, 8, 2))
            v = tl.permute(v, (0, 2, 1, 3, 4, 6, 5))
        w = tl.trans(tl.reshape(v, (NG * INTER, BLOCK_K)))
        w = w.to(hidden_states_ptr.dtype.element_ty) - (1 << (NUM_BITS - 1))

        acc = tl.dot(hidden, w, acc)

    acc *= sc[None, :]
    if HAS_BIAS:
        bias = tl.load(bias_ptr + cols)
        acc += bias[None, :]

    if ACT_TYPE == RELU:
        acc = tl.maximum(acc, 0.0)
    elif ACT_TYPE == SILU:
        acc = acc * tl.sigmoid(acc)
    elif ACT_TYPE == GELU:
        # sqrt(2 / pi), tanh approximation
        c = 0.7978845608028654
        z = c * (acc + 0.044715 * acc * acc * acc)
        acc = 0.5 * acc * (1.0 + tl_extra_shim.tanh(z))

    tl.store(
        output_ptr + rows[:, None] * stride_om + cols[None, :] * stride_on,
        acc.to(output_ptr.dtype.element_ty),
        mask=rows_mask[:, None],
    )


def _mixed_gemm_unpacked(
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
    """Unpack the weight once, then a dense GEMM."""
    dense = torch.empty((k, n), dtype=hidden_states.dtype, device=hidden_states.device)
    block_m, block_n, block_k, warps, stages = _SPLIT_TILE
    inter = 16 // num_bits

    with torch_device_fn.device(hidden_states.device):
        _dequant_kernel[(n // inter, k // (_DQ_BKB * 64))](
            weight,
            dense,
            k,
            n,
            NUM_BITS=num_bits,
            BKB=_DQ_BKB,
            num_warps=4,
        )
        _split_gemm_kernel[(triton.cdiv(m, block_m), n // block_n)](
            hidden_states,
            dense,
            output,
            scales,
            bias_in,
            m,
            n,
            k,
            hidden_states.stride(0),
            dense.stride(0),
            output.stride(0),
            HAS_BIAS=has_bias,
            ACT_TYPE=activation_raw,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=warps,
            num_stages=stages,
        )

    return output


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
    """Mixed-precision GEMM with weight-only integer quantization."""
    logger.debug("TRAIN MIXED_GEMM")

    m, n, k = _check_launch_arguments(
        output, hidden_states, weight, scales, bias, num_bits, activation_raw
    )

    # A pinned tile means a measurement; the generic entry already handles that case.
    tile = None
    if BLOCK_M is None and BLOCK_N is None and BLOCK_K is None:
        # The generic module's flat default, copied: that dict is module level and the
        # Hygon fallback rewrites it in place.
        tile = dict(_DEFAULT_TILE)
        if m > 128 and n % 128 == 0:
            tile = dict(_WIDE_TILE)
    if tile is None:
        return _generic_mixed_gemm(
            output,
            hidden_states,
            weight,
            scales,
            bias,
            num_bits,
            activation_raw,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    has_bias = bias is not None
    # Only read when HAS_BIAS, so the output doubles as the stand-in pointers.
    bias_in = bias if has_bias else output

    if m >= _SPLIT_MIN_M and n % _SPLIT_TILE[1] == 0 and k % (_DQ_BKB * 64) == 0:
        return _mixed_gemm_unpacked(
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

    BLOCK_M = tile["BLOCK_M"]
    BLOCK_N = tile["BLOCK_N"]
    BLOCK_K = tile["BLOCK_K"]
    num_warps = tile["num_warps"]
    num_stages = tile["num_stages"]

    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))
    with torch_device_fn.device(hidden_states.device):
        mixed_gemm_flat_kernel[grid](
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
