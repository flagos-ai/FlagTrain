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
"""Iluvatar (IxUDA) tuned scaled masked softmax backward.

Same operator as the generic implementation, whose kernel is correct here, so
this module reuses it verbatim and changes only the launch tile: ``_TILING``
(measured on a BI-V150) batches short rows and keeps long rows one per
program. Checks and the contract are imported from the generic module.
"""

import logging

import torch
import triton

from flag_train.megatron.scaled_masked_softmax_cuda_backward import (
    _I64_THRESHOLD,
    _MAX_KEY_LEN,
    _check_scale,
    _check_tensor,
    _scaled_masked_softmax_backward_kernel,
    _softmax_results_as,
)
from flag_train.runtime import torch_device_fn

logger = logging.getLogger(__name__)


# ``(max padded key length, rows per program, warps)``, measured on a BI-V150.
_TILING = ((128, 8, 4), (256, 4, 4), (512, 4, 4), (1024, 2, 4), (2048, 1, 4))
_DEFAULT_TILE = (1, 4)


def _tile(block_n: int) -> tuple:
    """Rows per program and warps for one tile, from the key length."""
    for max_key_len, rows, warps in _TILING:
        if block_n <= max_key_len:
            return rows, warps
    return _DEFAULT_TILE


@torch.compiler.disable
def scaled_masked_softmax_cuda_backward(output_grads, softmax_results, scale):
    """Fused scaled-softmax gradient, Megatron's masked-attention backward.

    The generic kernel with an Iluvatar tile; see the generic module for the
    contract. The mask is not an operand -- masked positions have
    ``softmax_results == 0`` and therefore zero gradient.

    Raises:
        RuntimeError: when Apex would reject the call (wrong rank or dtype, key
            length over 4096), or when the operands are on different devices.
        TypeError: when ``scale`` is not a real number or one-element tensor.
    """
    logger.debug("TRAIN_ILUVATAR SCALED_MASKED_SOFTMAX_CUDA_BACKWARD")
    _check_tensor(output_grads, "scaled masked softmax backward grad")
    _check_tensor(softmax_results, "scaled masked softmax backward results")
    if output_grads.device != softmax_results.device:
        raise RuntimeError(
            "scaled masked softmax backward needs both operands on one device"
        )
    sk = output_grads.shape[-1]
    if sk > _MAX_KEY_LEN:
        raise RuntimeError("key_seq_len >= 0 && key_seq_len <= 4096")
    scale = _check_scale(scale)
    if output_grads.numel() == 0:
        return torch.empty_like(output_grads, memory_format=torch.contiguous_format)

    dy = output_grads.contiguous()
    y = _softmax_results_as(dy, softmax_results)
    dx = torch.empty_like(dy)
    rows = dy.numel() // sk
    block_n = triton.next_power_of_2(sk)
    rows_per_program, num_warps = _tile(block_n)
    grid = (triton.cdiv(rows, rows_per_program),)
    with torch_device_fn.device(output_grads.device):
        _scaled_masked_softmax_backward_kernel[grid](
            dy,
            y,
            dx,
            sk,
            rows,
            scale,
            BLOCK_N=block_n,
            ROWS=rows_per_program,
            EVEN=sk == block_n and rows % rows_per_program == 0,
            I64=rows * sk >= _I64_THRESHOLD,
            num_warps=num_warps,
        )
    return dx
