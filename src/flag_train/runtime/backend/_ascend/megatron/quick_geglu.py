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
"""Ascend quick GEGLU (forward only).

Same math as ``flag_train.megatron.quick_geglu``; the kernel is persistent, one
program per vector core, and each program strides over several blocks, so the
grid also stays below Ascend's ``coreDim`` limit (the core benchmark shapes
need 1048576 blocks otherwise). Tiles are aligned to one row, so the row and
column block come from one scalar division per tile instead of a vector
division per element.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.runtime.backend._ascend.utils import CORE_NUM
from flag_train.utils import libentry, tl_extra_shim
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

BLOCK = 1024
# One program per vector core: the kernel strides over the remaining blocks, so
# the grid stays below Ascend's coreDim limit (the core benchmark shapes need
# 1048576 blocks otherwise).
_QUICK_GEGLU_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _check_input(tensor, name, dtypes):
    if tensor.dtype not in dtypes:
        wanted = " / ".join(str(dtype) for dtype in dtypes)
        raise ValueError(f"{name} needs a {wanted} tensor, got {tensor.dtype}")
    if tensor.device.type == "cpu":
        raise ValueError(f"{name} needs a device tensor, not one on the CPU")
    if tensor.device.type != runtime_device.name:
        raise ValueError(
            f"{name} needs a {runtime_device.name} tensor, got {tensor.device.type}"
        )


@libentry()
@triton.jit
def quick_geglu_kernel(
    y_ptr, out_ptr, d, n_blocks, blocks_per_row, linear_offset, BLOCK: tl.constexpr
):
    num_programs = tl.num_programs(0)
    for pid in range(tle.program_id(0), n_blocks, num_programs):
        row = pid // blocks_per_row
        col = (pid % blocks_per_row) * BLOCK + tl.arange(0, BLOCK)
        mask = col < d
        base = row * 2 * d
        y1 = tl.load(y_ptr + base + col, mask=mask, other=0.0).to(tl.float32)
        y2 = tl.load(y_ptr + base + col + d, mask=mask, other=0.0).to(tl.float32)
        out = (y1 / (1 + tl_extra_shim.exp(-1.702 * y1))) * (y2 + linear_offset)
        tl.store(out_ptr + row * d + col, out.to(out_ptr.dtype.element_ty), mask=mask)


def quick_geglu(y, linear_offset=0.0):
    logger.debug("TRAIN_ASCEND QUICK_GEGLU")
    _check_input(y, "quick_geglu", _QUICK_GEGLU_DTYPES)
    if y.shape[-1] % 2 != 0:
        raise ValueError(
            f"quick_geglu needs an even last dimension to split the gate and "
            f"info halves, got {y.shape[-1]}"
        )
    if y.numel() == 0:
        return torch.empty(
            y.shape[:-1] + (y.shape[-1] // 2,), dtype=y.dtype, device=y.device
        )
    y_flat = y.contiguous().view(-1)
    n = y_flat.numel() // 2
    d = y.shape[-1] // 2
    n_rows = n // d
    blocks_per_row = triton.cdiv(d, BLOCK)
    n_blocks = n_rows * blocks_per_row
    out = torch.empty(n, dtype=y.dtype, device=y.device)
    grid = (min(n_blocks, CORE_NUM),)
    with torch_device_fn.device(y.device):
        quick_geglu_kernel[grid](
            y_flat,
            out,
            d,
            n_blocks,
            blocks_per_row,
            linear_offset=linear_offset,
            BLOCK=BLOCK,
        )
    return out.view(y.shape[:-1] + (d,))
