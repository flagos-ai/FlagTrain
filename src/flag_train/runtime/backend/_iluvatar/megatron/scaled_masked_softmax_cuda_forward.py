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
"""Iluvatar (IxUDA) tuned scaled masked softmax forward.

Same operator as the generic implementation; only the launch differs, because
this part's Triton miscompiles the generic pipelined head walk
(``num_stages=4``): at ``(8, 16, 512, 1024)`` fp16 it produced 1572864 NaNs in
rows with ``query % 32 == 31``, while ``num_stages=1`` is exact. Each program
therefore owns one head block of one query position, so there is no walk to
pipeline, and the mask row is loaded once per program instead of once per
query (not measurable against the score traffic). Tiling follows ``_TILING``,
measured on a BI-V150. Checks and the contract are imported from the generic
module.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_train.megatron.scaled_masked_softmax_cuda_forward import (
    _I64_THRESHOLD,
    _LOG2E,
    _MAX_KEY_LEN,
    _check_input,
    _check_mask,
    _check_scale,
)
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


# ``(max padded key length, heads per program, warps)``, measured on a BI-V150.
_TILING = (
    (128, 16, 4),
    (256, 4, 4),
    (512, 4, 4),
    (1024, 2, 4),
    (2048, 1, 4),
    (4096, 1, 4),
)
_DEFAULT_TILE = (1, 4)


def _tile(block_n: int, heads: int) -> tuple:
    """Heads per program and warps for one block."""
    # ``ROWS`` is the extent of a ``tl.arange``, so it must be a power of two.
    limit = 1 << (heads.bit_length() - 1)
    for max_key_len, rows, warps in _TILING:
        if block_n <= max_key_len:
            return min(rows, limit), warps
    rows, warps = _DEFAULT_TILE
    return min(rows, limit), warps


@libentry()
@triton.jit
def _scaled_masked_softmax_forward_kernel(
    x_ptr,
    mask_ptr,
    y_ptr,
    sq,
    heads,
    n,
    groups,
    scale,
    mask_stride_b,
    BLOCK_N: tl.constexpr,
    ROWS: tl.constexpr,
    EVEN_COL: tl.constexpr,
    EVEN_HEAD: tl.constexpr,
    I64: tl.constexpr,
    EXP2: tl.constexpr,
):
    """Scale, mask, and softmax ``ROWS`` heads of one query position.

    All rows share one mask row; programs are ordered with the head group
    innermost so neighbours share a mask row or walk contiguous query rows.
    """
    # ``tle.program_id`` returns int64; narrow unless offsets can pass 2**31.
    program = tle.program_id(0)
    if not I64:
        program = program.to(tl.int32)
    head_block = program % groups
    rest = program // groups
    query = rest % sq
    batch = rest // sq

    col = tl.arange(0, BLOCK_N)
    # One mask row for every head; padding past ``n`` loads as True.
    mask_row = batch * mask_stride_b + query * n
    if EVEN_COL:
        attended = tl.load(mask_ptr + mask_row + col) == 0
    else:
        attended = (tl.load(mask_ptr + mask_row + col, mask=col < n, other=1) == 0) & (
            col < n
        )

    head = head_block * ROWS + tl.arange(0, ROWS)
    rows = (batch * heads + head) * sq + query
    offsets = rows[:, None] * n + col[None, :]

    if EVEN_COL and EVEN_HEAD:
        x = tl.load(x_ptr + offsets).to(tl.float32)
    else:
        valid = (col[None, :] < n) & (head < heads)[:, None]
        x = tl.load(x_ptr + offsets, mask=valid, other=0.0).to(tl.float32)

    # Non-attended lanes go to ``-inf``, so their exponential is exactly zero
    # after the max shift and no second select is needed.
    scores = tl.where(attended[None, :], x * scale, -float("inf"))
    max_value = tl.max(scores, 1)
    # An all-masked row is ``-inf`` across; pivot at zero so it comes out zeros
    # rather than ``0 / 0`` NaNs.
    max_value = tl.where(max_value == -float("inf"), 0.0, max_value)
    if EXP2:
        numerator = tl.exp2(scores - max_value[:, None])
    else:
        numerator = tl.exp(scores - max_value[:, None])
    denominator = tl.sum(numerator, 1)

    # Only an all-masked row sums to zero -- any attended lane contributes 1.
    inv = tl.where(denominator == 0, 0.0, 1.0 / denominator)

    out = (numerator * inv[:, None]).to(y_ptr.dtype.element_ty)
    if EVEN_COL and EVEN_HEAD:
        tl.store(y_ptr + offsets, out)
    else:
        tl.store(y_ptr + offsets, out, mask=valid)


@torch.compiler.disable
def scaled_masked_softmax_cuda_forward(inputs, mask, scale):
    """Fused scale + mask + softmax, Megatron's masked-attention forward.

    See the generic module for the contract; this backend only changes the
    launch. ``inputs`` is ``[b, np, sq, sk]`` fp16/bf16/fp32, ``mask`` is the
    bool ``[b or 1, 1, sq, sk]`` rejection mask, and ``scale`` is a real number
    or a one-element tensor.
    """
    logger.debug("TRAIN_ILUVATAR SCALED_MASKED_SOFTMAX_CUDA_FORWARD")
    _check_input(inputs)
    _check_mask(mask, inputs)
    b, np_, sq, sk = inputs.shape
    if sk > _MAX_KEY_LEN:
        raise ValueError(
            f"scaled masked softmax: key length {sk} exceeds Megatron's "
            f"fused-softmax limit of {_MAX_KEY_LEN}"
        )
    if sk == 0:
        raise ValueError("scaled masked softmax: key length must be positive")
    scale = _check_scale(scale)
    if inputs.numel() == 0:
        return torch.empty_like(inputs, memory_format=torch.contiguous_format)

    # 16-bit kernels fold log2(e) into the scale and use exp2; fp32 keeps exp.
    exp2 = inputs.dtype != torch.float32
    if exp2:
        scale *= _LOG2E

    x = inputs.contiguous()
    mask = mask.contiguous()
    out = torch.empty_like(x)
    block_n = triton.next_power_of_2(sk)
    rows, num_warps = _tile(block_n, np_)
    head_blocks = triton.cdiv(np_, rows)
    grid = (b * sq * head_blocks,)
    with torch_device_fn.device(inputs.device):
        _scaled_masked_softmax_forward_kernel[grid](
            x,
            mask,
            out,
            sq,
            np_,
            sk,
            head_blocks,
            scale,
            0 if mask.shape[0] == 1 else mask.stride(0),
            BLOCK_N=block_n,
            ROWS=rows,
            EVEN_COL=sk == block_n,
            EVEN_HEAD=np_ % rows == 0,
            I64=b * np_ * sq * sk >= _I64_THRESHOLD,
            EXP2=exp2,
            num_warps=num_warps,
        )
    return out
