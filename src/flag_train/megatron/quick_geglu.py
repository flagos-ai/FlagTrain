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
"""Quick GEGLU activation (forward only)."""

import logging

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry, tl_extra_shim
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

BLOCK = 1024
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
def quick_geglu_kernel(y_ptr, out_ptr, n, d, linear_offset, BLOCK: tl.constexpr):
    pid = tle.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    row = offsets // d
    col = offsets % d
    y1_offsets = row * 2 * d + col
    y2_offsets = row * 2 * d + col + d
    mask = offsets < n
    y1 = tl.load(y_ptr + y1_offsets, mask=mask, other=0.0).to(tl.float32)
    y2 = tl.load(y_ptr + y2_offsets, mask=mask, other=0.0).to(tl.float32)
    out = (y1 / (1 + tl_extra_shim.exp(-1.702 * y1))) * (y2 + linear_offset)
    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


def quick_geglu(y, linear_offset=0.0):
    logger.debug("TRAIN QUICK_GEGLU")
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
    out = torch.empty(y_flat.numel() // 2, dtype=y.dtype, device=y.device)
    grid = (triton.cdiv(y.numel() // 2, BLOCK),)
    d = y.shape[-1] // 2
    with torch_device_fn.device(y.device):
        quick_geglu_kernel[grid](
            y_flat, out, y.numel() // 2, d, linear_offset=linear_offset, BLOCK=BLOCK
        )
    return out.view(y.shape[:-1] + (d,))
