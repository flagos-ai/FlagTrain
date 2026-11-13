# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""T-Head fused FP8 GEMM and a decode-once path for large BF16 inputs."""

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
def _decode_fp8_bf16(W, S, D, NUMEL: tl.constexpr):
    groups = tl.program_id(0) * 32 + tl.arange(0, 32)
    offsets = groups[:, None] * 128 + tl.arange(0, 128)[None, :]
    codes = tl.load(W + offsets, offsets < NUMEL, other=0).to(tl.uint16)
    scales = tl.load(S + groups, groups * 128 < NUMEL, other=0)
    bits = (((codes & 0x80) << 8) | ((codes & 0x7F) << 4)) + 0x3C00
    values = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    tl.store(D + offsets, (values * scales[:, None]).to(tl.bfloat16), offsets < NUMEL)


def fp8_gemm_triton(inp, weight, scale, quantization_group_size):
    """Decode weights once for aligned large-M BF16, then use the PPU GEMM."""
    logger.debug("TRAIN_THEAD FP8_GEMM_TRITON")
    _validate(inp, weight, scale, quantization_group_size)
    m, k = inp.shape
    n = weight.shape[1]
    if (
        inp.dtype == torch.bfloat16
        and m >= 512
        and k > 0
        and n > 0
        and k % 128 == 0
        and n % 128 == 0
        and quantization_group_size == 128
        and k * n < 2**31
    ):
        # The fused path decodes each weight tile once per output row tile.
        # Materialize one BF16 copy per call and reuse it in the vendor GEMM.
        with torch_device_fn.device(inp.device):
            decoded = torch.empty((k, n), dtype=inp.dtype, device=inp.device)
            _decode_fp8_bf16[(triton.cdiv(k * n, 4096),)](
                weight, scale, decoded, k * n, num_warps=4, num_stages=1
            )
            return torch.matmul(inp, decoded)
    config = None
    if (
        inp.dtype == torch.bfloat16
        and 16 < m <= 64
        and k > 0
        and n > 0
        and k % 128 == 0
        and n % 128 == 0
        and quantization_group_size == 128
    ):
        config = (32 if m <= 32 else 64, 64, 128, 4, 3)
    return _launch(inp, weight, scale, quantization_group_size, config)
