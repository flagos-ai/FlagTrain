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
"""Ascend tuned ``bias_geglu_impl``.

Same contract, promotion and arithmetic as
``flag_train.megatron.bias_geglu_impl`` -- the per-row kernel is imported from
there, including the integer-width wrap, the bool OR and the compute-dtype
rounding. What differs is the launch:

* **Persistence, not one program per row.** The grid is bounded by the vector
  core count (``CORE_NUM``) and each program walks its rows with a runtime
  loop, so a large token count does not pay the per-program floor once per
  row.
* **The shared ``[hidden]`` bias is loaded once per program.** When one column
  tile covers the row (the usual MLP case), the bias tiles stay live across
  the row walk, so each row issues two loads and one store instead of four
  loads and one store; measured up to 2x on the largest shapes. Rows that
  span several column tiles keep the per-row load: nesting the column loop
  outside a row loop miscompiles on this backend.
* **The column tile follows the row width** instead of the portable launch's
  fixed 1024 lanes: ``next_power_of_2(half)`` clamped to a per-element-size
  ceiling (8192 lanes for 2-byte operands, 4096 otherwise, 2048 for fp64).
  Four loaded tensors and two compute tensors share the 192 KiB UB, so the
  ceiling shrinks as the element width grows.
* **One persistent launch per call** replaces the row grid, so the small
  token counts stop being launch-bound.

The host-side validation, promotion, staging and copies are shared through
``_prepare`` / ``_finish`` imported from the generic module; the runtime's
registration runs after the generic package is fully imported (see
``flag_train/__init__.py``), so the import cannot re-enter the operator scan.
"""

import logging

import triton
import triton.language as tl

from flag_train.megatron.bias_geglu_impl import (
    _BIAS_VECTOR,
    _NUM_WARPS,
    _activate,
    _bias_geglu_add,
    _bias_geglu_row,
    _finish,
    _prepare,
)
from flag_train.runtime import torch_device_fn
from flag_train.runtime.backend._ascend.utils import CORE_NUM
from flag_train.utils import libentry
from flag_train.utils import triton_lang_extension as tle

__all__ = ["bias_geglu_impl"]

logger = logging.getLogger(__name__)

# Column-tile ceiling per operand element size. A tile holds four loaded
# tensors (gate/linear for both the input and the bias) plus two compute
# tensors, so a 16384-lane 2-byte tile with a bias needs ~256 KiB and overflows
# the 192 KiB UB; these caps stay below it (measured: 2-byte needs < 16384).
_TILE_LANES = {1: 4096, 2: 8192, 4: 4096, 8: 2048}
_MIN_TILE_LANES = 64


@libentry()
@triton.jit
def _bias_geglu_persistent_kernel(
    x_ptr,
    bias_ptr,
    y_ptr,
    n_cols,
    rows,
    BIAS_MODE: tl.constexpr,
    ADD_KIND: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Bias add and GEGLU over every row, walked by a bounded grid."""
    for row in range(tle.program_id(0), rows, tl.num_programs(0)):
        _bias_geglu_row(
            x_ptr,
            bias_ptr,
            y_ptr,
            row,
            n_cols,
            BIAS_MODE,
            ADD_KIND,
            IS_FP64,
            BLOCK,
        )


@libentry()
@triton.jit
def _bias_geglu_shared_bias_kernel(
    x_ptr,
    bias_ptr,
    y_ptr,
    n_cols,
    rows,
    ADD_KIND: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The ``[2 * hidden]`` bias shared by every row, loaded once per program.

    Only used when one column tile covers the whole row, so the bias tiles
    stay live while the program walks its rows: the row loop then issues two
    loads and one store instead of four loads and one store.
    """
    COMPUTE_DTYPE: tl.constexpr = tl.float64 if IS_FP64 else tl.float32
    hidden = n_cols * 2
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    bg = tl.load(bias_ptr + cols, mask=mask, other=0)
    bl = tl.load(bias_ptr + n_cols + cols, mask=mask, other=0)
    for row in range(tle.program_id(0), rows, tl.num_programs(0)):
        x_row = x_ptr + row * hidden
        xg = tl.load(x_row + cols, mask=mask, other=0)
        xl = tl.load(x_row + n_cols + cols, mask=mask, other=0)
        gate, linear = _bias_geglu_add(xg, xl, bg, bl, ADD_KIND, COMPUTE_DTYPE)
        out = _activate(gate, linear)
        tl.store(y_ptr + row * n_cols + cols, out.to(y_ptr.dtype.element_ty), mask=mask)


def _launch(prep):
    """Launch the persistent kernel over ``prep``."""
    itemsize = max(prep.x_flat.element_size(), prep.bias_arg.element_size())
    lane_cap = _TILE_LANES[itemsize]
    block = max(_MIN_TILE_LANES, min(lane_cap, triton.next_power_of_2(prep.half)))
    grid = (min(prep.rows, CORE_NUM),)
    shared_bias = prep.bias_mode == _BIAS_VECTOR and prep.half <= block
    with torch_device_fn.device(prep.device):
        # Same LibEntry.run path as kernel[grid], without its temporary
        # closure/dict merge. warmup=False is required on cold JIT misses.
        if shared_bias:
            _bias_geglu_shared_bias_kernel.run(
                prep.x_flat,
                prep.bias_arg,
                prep.out_flat,
                prep.half,
                prep.rows,
                grid=grid,
                warmup=False,
                ADD_KIND=prep.add_kind,
                IS_FP64=prep.is_fp64,
                BLOCK=block,
                num_warps=_NUM_WARPS,
                num_stages=1,
            )
        else:
            _bias_geglu_persistent_kernel.run(
                prep.x_flat,
                prep.bias_arg,
                prep.out_flat,
                prep.half,
                prep.rows,
                prep.bias_mode,
                grid=grid,
                warmup=False,
                ADD_KIND=prep.add_kind,
                IS_FP64=prep.is_fp64,
                BLOCK=block,
                num_warps=_NUM_WARPS,
                num_stages=1,
            )


def bias_geglu_impl(input, bias):
    """Return Megatron's bias + GEGLU forward with the Ascend launch."""
    logger.debug("TRAIN_ASCEND BIAS_GEGLU_IMPL")
    prep = _prepare(input, bias)
    if prep.rows and prep.half:
        _launch(prep)
    return _finish(prep)
