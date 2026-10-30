# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend FP8 GEMM: decode once before large BF16 matrix multiplications.

The generic fused kernel repeats weight decoding for each M tile. On 910B4,
large M benefits from a per-call BF16 workspace and a separate Triton GEMM.
Both kernels and allocations belong to the public call; no weights are cached.
Small M and other layouts/dtypes retain the generic implementation.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_train.deepspeed.fp8_gemm_triton import _launch, _validate
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _decode_fp8_bf16(W, S, D, NUMEL: tl.constexpr, BG: tl.constexpr):
    groups = tl.program_id(0) * BG + tl.arange(0, BG)
    offsets = groups[:, None] * 128 + tl.arange(0, 128)[None, :]
    codes = tl.load(W + offsets, offsets < NUMEL, other=0).to(tl.uint16)
    scales = tl.load(S + groups, groups * 128 < NUMEL, other=0)
    bits = (((codes & 0x80) << 8) | ((codes & 0x7F) << 4)) + 0x3C00
    values = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    decoded = (values * scales[:, None]).to(tl.bfloat16)
    tl.store(D + offsets, decoded, offsets < NUMEL)


@libentry()
@triton.jit
def _matmul_bf16(
    A,
    B,
    C,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    group = pid // (8 * nn)
    first_m = group * 8
    group_m = tl.minimum(nm - first_m, 8)
    pm = first_m + (pid % (8 * nn)) % group_m
    pn = (pid % (8 * nn)) // group_m
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
        b = tl.load(
            B + kk[:, None] * N + cols[None, :],
            (kk[:, None] < K) & (cols[None, :] < N),
            other=0,
        )
        acc = tl.dot(a, b, acc)
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def matmul_fp8_triton(inp, weight, scale, quantization_group_size):
    """Same contract as the generic API; large aligned BF16 uses two kernels.

    The intermediate is K*N BF16 elements (64 MiB for the core test weights).
    It is allocated and populated afresh in every call. No Torch computation
    or DeepSpeed operator is used by this implementation.
    """
    logger.debug("TRAIN_ASCEND FP8 GEMM")
    _validate(inp, weight, scale, quantization_group_size)
    m, k = inp.shape
    n = weight.shape[1]
    use_two_stage = (
        inp.dtype == torch.bfloat16
        and m >= 512
        and k > 0
        and n > 0
        and k % 128 == 0
        and n % 128 == 0
        and quantization_group_size == 128
        # These kernels use int32 offsets; the generic kernel uses wider
        # indexing. Route large tensors/strides before launching either stage.
        and k * n <= 2**31 - 1
        and m * n <= 2**31 - 1
        and (m - 1) * inp.stride(0) + (k - 1) * inp.stride(1) <= 2**31 - 1
    )
    if not use_two_stage:
        return _launch(inp, weight, scale, quantization_group_size)

    with torch_device_fn.device(inp.device):
        decoded = torch.empty((k, n), dtype=inp.dtype, device=inp.device)
        out = torch.empty((m, n), dtype=inp.dtype, device=inp.device)
        _decode_fp8_bf16[(triton.cdiv(k * n, 4096),)](
            weight, scale, decoded, k * n, 32, num_warps=4, num_stages=1
        )
        _matmul_bf16[(triton.cdiv(m, 128) * triton.cdiv(n, 256),)](
            inp,
            decoded,
            out,
            m,
            k,
            n,
            inp.stride(0),
            inp.stride(1),
            128,
            256,
            128,
            num_warps=4,
            num_stages=1,
        )
    return out
