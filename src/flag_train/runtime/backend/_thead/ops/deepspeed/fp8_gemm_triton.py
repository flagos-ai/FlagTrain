# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""T-Head launch policy for the shared fused FP8 GEMM kernel."""

import logging

import torch

from flag_train.deepspeed.fp8_gemm_triton import _launch, _validate

logger = logging.getLogger(__name__)


def matmul_fp8_triton(inp, weight, scale, quantization_group_size):
    """Reuse the generic kernel with a larger K tile for aligned medium-M BF16.

    BK=128 halves the K-loop count compared with the generic BK=64 policy in
    this range. No extra workspace, weight cache or baseline call is used.
    Other inputs retain the generic launch policy and validation contract.
    """
    logger.debug("TRAIN_THEAD FP8 GEMM")
    _validate(inp, weight, scale, quantization_group_size)
    m, k = inp.shape
    n = weight.shape[1]
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
