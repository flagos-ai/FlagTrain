# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MetaX FP8 GEMM with a fused small-M path and per-call BF16 workspace."""

import torch
import triton
import triton.language as tl

from flag_train.deepspeed.fp8_gemm_triton import _launch, _validate
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry


@libentry()
@triton.jit
def _decode_fp8_bf16(W, S, D, NUMEL: tl.constexpr, BG: tl.constexpr):
    groups = tl.program_id(0) * BG + tl.arange(0, BG)
    offsets = groups[:, None] * 128 + tl.arange(0, 128)[None, :]
    codes = tl.load(W + offsets, offsets < NUMEL, other=0).to(tl.uint16)
    scales = tl.load(S + groups, groups * 128 < NUMEL, other=0)
    bits = (((codes & 0x80) << 8) | ((codes & 0x7F) << 4)) + 0x3C00
    values = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    tl.store(D + offsets, (values * scales[:, None]).to(tl.bfloat16), offsets < NUMEL)


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


def _use_two_stage(inp, weight, group_size):
    m, k = inp.shape
    n = weight.shape[1]
    return (
        inp.dtype == torch.bfloat16
        and m > 16
        and k > 0
        and n > 0
        and k % 128 == 0
        and n % 128 == 0
        and group_size == 128
        and k * n <= 2**31 - 1
        and m * n <= 2**31 - 1
        and (m - 1) * inp.stride(0) + (k - 1) * inp.stride(1) <= 2**31 - 1
    )


def matmul_fp8_triton(inp, weight, scale, quantization_group_size):
    """Decode once for aligned BF16 M>16; fuse other supported inputs.

    Both stages and allocations run on every call, without cross-call caching.
    The BF16 workspace uses 2*K*N bytes. The fused path disables the compiler's
    failing rank-one scale broadcast and uses full flattened group indexing.
    The shared kernel's default remains unchanged for other vendors.
    """
    _validate(inp, weight, scale, quantization_group_size)
    m, k = inp.shape
    n = weight.shape[1]
    if _use_two_stage(inp, weight, quantization_group_size):
        if m <= 128:
            bm, bn, bk = 64, 64, 64
        elif m <= 512:
            bm, bn, bk = 64, 128, 32
        else:
            bm, bn, bk = 128, 128, 32
        with torch_device_fn.device(inp.device):
            decoded = torch.empty((k, n), dtype=inp.dtype, device=inp.device)
            out = torch.empty((m, n), dtype=inp.dtype, device=inp.device)
            _decode_fp8_bf16[(triton.cdiv(k * n, 4096),)](
                weight,
                scale,
                decoded,
                k * n,
                32,
                num_warps=4,
                num_stages=1,
            )
            _matmul_bf16[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
                inp,
                decoded,
                out,
                m,
                k,
                n,
                inp.stride(0),
                inp.stride(1),
                bm,
                bn,
                bk,
                num_warps=4,
                num_stages=1,
            )
        return out
    bm = 16 if m <= 16 else 32 if m <= 32 else 64 if m <= 64 else 128
    return _launch(
        inp,
        weight,
        scale,
        quantization_group_size,
        (bm, 128, 64, 4, 1),
        broadcast_scales=False,
    )
