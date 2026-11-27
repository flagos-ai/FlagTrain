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
"""Ascend fused bias + SwiGLU (forward only).

Same contract, validation, promotion, staging and arithmetic as
``flag_train.megatron.bias_swiglu_impl`` -- the per-tile kernel and the host
helpers are imported from there. Only the launch differs: the generic kernel
launches one program per flattened row, so a large token count needs one grid
slot per row and can exceed Ascend's 65535-program ``coreDim`` limit. Here the
grid is one program per vector core and each program walks the
``(row, column block)`` tiles with a stride, so the launch stays far below that
limit. Tiles never cross a row boundary, so the row and column block come from
one scalar division per tile rather than a vector division per element.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_train.megatron.bias_swiglu_impl import (
    _MODE_PLAIN,
    _NUM_WARPS,
    _bias_swiglu_tile,
    _finish,
    _prepare,
)
from flag_train.runtime import torch_device_fn
from flag_train.runtime.backend._ascend.utils import CORE_NUM
from flag_train.utils import libentry
from flag_train.utils import triton_lang_extension as tle

__all__ = ["bias_swiglu_impl"]

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 1024
_MIN_BLOCK_SIZE = 16


@libentry()
@triton.jit
def _bias_swiglu_persistent_kernel(
    x_ptr,
    bias_ptr,
    y_ptr,
    n_tiles,
    blocks_per_row,
    n_cols,
    clamp_value,
    gate_scale,
    linear_scale,
    BIAS_MODE: tl.constexpr,
    MODE: tl.constexpr,
    HAS_LINEAR_CLAMP: tl.constexpr,
    ADD_DTYPE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Bias add and activation over every (row, column block) tile."""
    num_programs = tl.num_programs(0)
    for tile in range(tle.program_id(0), n_tiles, num_programs):
        row = tile // blocks_per_row
        start = (tile % blocks_per_row) * BLOCK
        _bias_swiglu_tile(
            x_ptr,
            bias_ptr,
            y_ptr,
            row,
            start,
            n_cols,
            clamp_value,
            gate_scale,
            linear_scale,
            BIAS_MODE,
            MODE,
            HAS_LINEAR_CLAMP,
            ADD_DTYPE,
            COMPUTE_DTYPE,
            BLOCK,
        )


def _launch(prep):
    """Launch the persistent tile walk over ``prep``."""
    block = max(_MIN_BLOCK_SIZE, min(_BLOCK_SIZE, triton.next_power_of_2(prep.n_cols)))
    blocks_per_row = triton.cdiv(prep.n_cols, block)
    n_tiles = prep.rows * blocks_per_row
    add_dtype = tl.float64 if prep.out_dtype == torch.float64 else tl.float32
    compute_dtype = (
        tl.float64
        if prep.mode == _MODE_PLAIN and prep.out_dtype == torch.float64
        else tl.float32
    )
    # One program per vector core, far below Ascend's 65535-program coreDim
    # limit; the kernel's tile walk covers the rest.
    grid = (min(n_tiles, CORE_NUM),)
    with torch_device_fn.device(prep.device):
        _bias_swiglu_persistent_kernel[grid](
            prep.x_flat,
            prep.bias_arg,
            prep.out_flat,
            n_tiles,
            blocks_per_row,
            prep.n_cols,
            prep.clamp,
            prep.gate_scale,
            1.0 if prep.linear_scale is None else prep.linear_scale,
            BIAS_MODE=prep.bias_mode,
            MODE=prep.mode,
            HAS_LINEAR_CLAMP=prep.linear_scale is not None,
            ADD_DTYPE=add_dtype,
            COMPUTE_DTYPE=compute_dtype,
            BLOCK=block,
            num_warps=_NUM_WARPS,
        )


def bias_swiglu_impl(
    input,
    bias,
    fp8_input_store=False,
    cpu_offload_input=False,
    clamp_value=None,
    gate_clamp_scale=None,
    linear_clamp_scale=None,
):
    """Bias plus SwiGLU / SiTU-GLU with the Ascend launch."""
    logger.debug("TRAIN_ASCEND BIAS_SWIGLU_IMPL")
    prep = _prepare(input, bias, clamp_value, gate_clamp_scale, linear_clamp_scale)
    if prep.rows and prep.n_cols:
        _launch(prep)
    return _finish(prep)
