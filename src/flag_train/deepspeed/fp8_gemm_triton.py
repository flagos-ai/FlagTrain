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
"""Fused DeepSpeed FP8-weight GEMM for NVIDIA GPUs.

Weights use DeepSpeed's custom uint8 E4M3 encoding, not torch.float8_e4m3fn.
The scale tensor is the float32 tensor returned by FP_Quantize.get_scales().
The public entry point does not import or dispatch to a DeepSpeed kernel.
"""

import logging

import torch
import triton
import triton.language as tl

import flag_train
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _matmul_kernel(
    A,
    B,
    S,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    GROUP: tl.constexpr,
    IS_BF16: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
):
    pid = tle.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    group = pid // (GM * nn)
    first_m = group * GM
    group_m = tl.minimum(nm - first_m, GM)
    pm = first_m + (pid % (GM * nn)) % group_m
    pn = (pid % (GM * nn)) // group_m

    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + ks
        a = tl.load(
            A + rows[:, None] * AM + kk[None, :] * AK,
            (rows[:, None] < M) & (kk[None, :] < K),
            other=0,
        )
        code = tl.load(
            B + kk[:, None] * N + cols[None, :],
            (kk[:, None] < K) & (cols[None, :] < N),
            other=0,
        ).to(tl.uint16)
        # Fast broadcast applies only when each tile lies inside one group.
        if N % GROUP == 0 and GROUP % BN == 0:
            scales = tl.load(
                S + kk[:, None] * (N // GROUP) + pn * BN // GROUP,
                kk[:, None] < K,
                other=0,
            )
        else:
            scales = tl.load(
                S + (kk[:, None] * N + cols[None, :]) // GROUP,
                (kk[:, None] < K) & (cols[None, :] < N),
                other=0,
            )
        if IS_BF16:
            bits = (((code & 0x80) << 8) | ((code & 0x7F) << 4)) + 0x3C00
            value = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
            weight = (value * scales).to(tl.bfloat16)
        else:
            bits = (((code & 0x80) << 8) | ((code & 0x7F) << 7)) + 0x2000
            value = bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            weight = (value * scales).to(tl.float16)
        # Preserve the upstream blockwise FP32 accumulation semantics.
        acc += tl.dot(a, weight)

    tl.store(
        C + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def _validate(inp, weight, scale, group_size):
    if inp.ndim != 2 or weight.ndim != 2:
        raise ValueError("input and weight must both be two-dimensional")
    if inp.shape[1] != weight.shape[0]:
        raise ValueError("input and weight have incompatible K dimensions")
    if inp.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("input must be torch.bfloat16 or torch.float16")
    if weight.dtype != torch.uint8 or scale.dtype != torch.float32:
        raise TypeError(
            "weight must be uint8; scales must be float32 from get_scales()"
        )
    if not inp.is_cuda or weight.device != inp.device or scale.device != inp.device:
        raise ValueError("input, weight and scales must be on the same CUDA device")
    if not weight.is_contiguous() or not scale.is_contiguous():
        raise ValueError("weight and scales must be contiguous")
    if (
        not isinstance(group_size, int)
        or isinstance(group_size, bool)
        or group_size <= 0
    ):
        raise ValueError("quantization_group_size must be a positive integer")
    expected = triton.cdiv(weight.numel(), group_size)
    if scale.numel() != expected:
        raise ValueError(f"expected {expected} scales, got {scale.numel()}")


def _launch(inp, weight, scale, group_size, config=None):
    m, k = inp.shape
    n = weight.shape[1]
    out = torch.empty((m, n), dtype=inp.dtype, device=inp.device)
    if m == 0 or n == 0:
        return out
    if config is None:
        bm = 16 if m <= 16 else 32 if m <= 32 else 64 if m <= 64 else 128
        bn, bk, warps, stages = 64, 128 if m <= 16 else 64, 4, 3
        if m >= 256:
            bm, bn, bk, warps, stages = 128, 128, 64, 4, 3
    else:
        bm, bn, bk, warps, stages = config
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    with torch_device_fn.device(inp.device):
        _matmul_kernel[grid](
            inp,
            weight,
            scale,
            out,
            m,
            n,
            k,
            inp.stride(0),
            inp.stride(1),
            group_size,
            inp.dtype == torch.bfloat16,
            bm,
            bn,
            bk,
            8,
            num_warps=warps,
            num_stages=stages,
        )
    return out


def matmul_fp8_triton(inp, weight, scale, quantization_group_size):
    """Compute [M,K] @ [K,N] with fused FP8 decoding and FP32 accumulation.

    Supports BF16/FP16 activations (including noncontiguous activations),
    contiguous uint8 weights and float32 scales grouped in flattened K,N order.
    Tail tiles and zero-size dimensions are supported. No quantization,
    dequantization workspace, output cache, or host-device copy is used.
    """
    logger.debug("TRAIN FP8 GEMM")
    if flag_train.vendor_name != "nvidia":
        raise NotImplementedError(
            "fp8_gemm_triton is currently validated only on NVIDIA"
        )
    _validate(inp, weight, scale, quantization_group_size)
    return _launch(inp, weight, scale, quantization_group_size)


def matmul_fp8_bf16(inp, weight, scale, quantization_group_size):
    if inp.dtype != torch.bfloat16:
        raise TypeError("matmul_fp8_bf16 requires BF16 activations")
    return matmul_fp8_triton(inp, weight, scale, quantization_group_size)


def matmul_fp8_fp16(inp, weight, scale, quantization_group_size):
    if inp.dtype != torch.float16:
        raise TypeError("matmul_fp8_fp16 requires FP16 activations")
    return matmul_fp8_triton(inp, weight, scale, quantization_group_size)
