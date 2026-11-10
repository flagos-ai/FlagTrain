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
import logging
import numbers

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.runtime.backend._ascend.utils import CORE_NUM
from flag_train.utils import libentry

logger = logging.getLogger(__name__)

_NAME = "scaled_upper_triang_masked_softmax_backward"

_SCALED_UPPER_TRIANG_MASKED_SOFTMAX_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)

# Elements (not bytes) in one program's tile, which fixes BLOCK_M for a given
# BLOCK_N. dy and y are widened to fp32 before the row sum, so the fp32 budget
# is tighter than the forward's -- see ``_MAX_ONE_TILE_FP32``.
_TILE_ELEMENTS = 8192

# Cap rows per program so short rows still spread over the cores.
_MAX_BLOCK_M = 32

# Widest row the single-tile kernel serves; wider rows, and fp32 rows past
# ``_MAX_ONE_TILE_FP32``, take the chunked kernel.
_MAX_BLOCK_N = 8192

# Widest fp32 row the single-tile kernel can hold. At 8192 lanes dy, y and the
# fp32 temporaries are 32 KB each and the tile no longer fits the 192 KB UB
# (``ConvertLinalgIRToBinary`` fails); chunk those rows instead.
_MAX_ONE_TILE_FP32 = 4096

# ``coreDim`` cap for the chunked wide-row path.
_MAX_PROGRAMS = 65535

# Columns one chunk of the wide-row path walks.
_WIDE_BLOCK_N = 1024


def _check_input(x, name, dtypes):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} needs a tensor, got {type(x).__name__}")
    if x.dtype not in dtypes:
        wanted = " / ".join(str(dtype) for dtype in dtypes)
        raise ValueError(f"{name} needs a {wanted} tensor, got {x.dtype}")
    if x.device.type == "cpu":
        raise ValueError(f"{name} needs a device tensor, not one on the CPU")
    if x.device.type != runtime_device.name:
        raise ValueError(
            f"{name} needs a {runtime_device.name} tensor, got {x.device.type}"
        )


def _check_scale(scale):
    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            raise ValueError(f"scale needs a single element, got {scale.numel()}")
        scale = scale.item()
    if not isinstance(scale, numbers.Real):
        raise TypeError(f"scale needs a real number, got {type(scale).__name__}")
    return float(scale)


def _check_shape(x):
    if x.dim() != 3:
        raise ValueError(
            f"{_NAME} needs a [attn_batches, sq, sk] tensor, got {tuple(x.shape)}"
        )
    if x.shape[1] != x.shape[2]:
        raise ValueError(
            f"{_NAME} needs sq == sk for the upper-triangular mask, got "
            f"sq={x.shape[1]} sk={x.shape[2]}"
        )


@libentry()
@triton.jit
def _scaled_upper_triang_masked_softmax_backward_kernel(
    dy_ptr,
    y_ptr,
    dx_ptr,
    n_rows,
    n_cols,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    """``dx = scale * y * (dy - rowsum(dy * y))``, zero above the diagonal.

    Kept columns satisfy ``col <= row_pos``. A program owns ``BLOCK_M``
    consecutive rows and strides over the row blocks, so the grid can be capped
    at the core count instead of the row count.
    """
    n_blocks = tl.cdiv(n_rows, BLOCK_M)
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    for block in range(pid, n_blocks, n_progs):
        rows = block * BLOCK_M + tl.arange(0, BLOCK_M)
        row_pos = rows % n_cols
        base = rows.to(tl.int64) * n_cols
        cols = tl.arange(0, BLOCK_N)
        in_range = cols < n_cols
        causal = in_range[None, :] & (cols[None, :] <= row_pos[:, None])
        store_mask = in_range[None, :]
        # The stride walk never starts past n_rows; the guard is only needed
        # when the tile does not divide the row count.
        if not EVEN_M:
            row_mask = (rows < n_rows)[:, None]
            causal = causal & row_mask
            store_mask = store_mask & row_mask

        offsets = base[:, None] + cols[None, :]
        # A masked lane loads as zero, so it adds nothing to the row sum.
        dy = tl.load(dy_ptr + offsets, mask=causal, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=causal, other=0.0).to(tl.float32)
        total = tl.sum(dy * y, 1)
        # Selected rather than left to the arithmetic: ``0 * (0 - total)`` would
        # write -0 instead of the +0 the masked half is supposed to hold.
        dx = tl.where(causal, y * (dy - total[:, None]) * scale, 0.0)
        tl.store(
            dx_ptr + offsets,
            dx.to(dx_ptr.dtype.element_ty),
            mask=store_mask,
        )


@libentry()
@triton.jit
def _scaled_upper_triang_masked_softmax_backward_wide_kernel(
    dy_ptr,
    y_ptr,
    dx_ptr,
    n_rows,
    n_cols,
    scale,
    BLOCK_N: tl.constexpr,
):
    """Chunked causal-prefix gradient for rows wider than the UB tile.

    One program walks rows with stride ``n_programs``; per row it accumulates
    ``sum(dy * y)`` over the kept prefix, then re-reads it to write.
    """
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    col_arange = tl.arange(0, BLOCK_N)
    for row in range(pid, n_rows, n_progs):
        pos = row % n_cols
        base = row.to(tl.int64) * n_cols
        prefix_end = pos + 1
        # Chunks before the diagonal are unpredicated; one crosses it.
        full_end = (prefix_end // BLOCK_N) * BLOCK_N

        total = tl.zeros([], tl.float32)
        for start in range(0, full_end, BLOCK_N):
            cols = start + col_arange
            dy = tl.load(dy_ptr + base + cols).to(tl.float32)
            y = tl.load(y_ptr + base + cols).to(tl.float32)
            total += tl.sum(dy * y, 0)
        if full_end < prefix_end:
            cols = full_end + col_arange
            keep = cols <= pos
            dy = tl.load(dy_ptr + base + cols, mask=keep, other=0.0).to(tl.float32)
            y = tl.load(y_ptr + base + cols, mask=keep, other=0.0).to(tl.float32)
            total += tl.sum(dy * y, 0)

        # The row sum is only known now, so walk the prefix again rather than
        # hold it in registers.
        for start in range(0, full_end, BLOCK_N):
            cols = start + col_arange
            dy = tl.load(dy_ptr + base + cols).to(tl.float32)
            y = tl.load(y_ptr + base + cols).to(tl.float32)
            dx = y * (dy - total) * scale
            tl.store(dx_ptr + base + cols, dx.to(dx_ptr.dtype.element_ty))
        if full_end < prefix_end:
            cols = full_end + col_arange
            keep = cols <= pos
            dy = tl.load(dy_ptr + base + cols, mask=keep, other=0.0).to(tl.float32)
            y = tl.load(y_ptr + base + cols, mask=keep, other=0.0).to(tl.float32)
            dx = tl.where(keep, y * (dy - total) * scale, 0.0)
            tl.store(
                dx_ptr + base + cols,
                dx.to(dx_ptr.dtype.element_ty),
                mask=cols < n_cols,
            )

        # Everything past the prefix is masked: a pure zero fill.
        zero = tl.zeros([BLOCK_N], dx_ptr.dtype.element_ty)
        pad_end = tl.cdiv(prefix_end, BLOCK_N) * BLOCK_N
        for start in range(pad_end, n_cols, BLOCK_N):
            cols = start + col_arange
            tl.store(dx_ptr + base + cols, zero, mask=cols < n_cols)


def _tile_config(n_cols, dtype):
    """Tile shape for rows of ``n_cols`` columns: rows, columns."""
    block_n = triton.next_power_of_2(n_cols)
    block_m = max(1, min(_MAX_BLOCK_M, _TILE_ELEMENTS // block_n))
    # A 2 x 4096 fp32 tile overflows the UB at compile time; one row fits.
    # Narrower rows keep the wider tile.
    if dtype == torch.float32 and block_n >= 4096:
        block_m = 1
    return block_m, block_n


def _wide_backward(grad_output, output, scale):
    """Rows that do not fit one UB tile (wide rows and fp32 rows past 4096)."""
    dy = grad_output.contiguous().view(-1)
    y = output.contiguous().view(-1)
    grad_input = torch.empty_like(dy)
    n_cols = grad_output.shape[2]
    n_rows = dy.numel() // n_cols
    # One row per loop step; the grid is capped below ``coreDim`` and each
    # program strides over the remaining rows.
    grid = (min(n_rows, _MAX_PROGRAMS),)
    with torch_device_fn.device(grad_output.device):
        _scaled_upper_triang_masked_softmax_backward_wide_kernel[grid](
            dy,
            y,
            grad_input,
            n_rows,
            n_cols,
            scale,
            BLOCK_N=_WIDE_BLOCK_N,
        )
    return grad_input.view(grad_output.shape)


def scaled_upper_triang_masked_softmax_backward(grad_output, output, scale):
    """Gradient of ``softmax(scale * x)`` under Megatron's upper-triangular mask.

    Entry ``(r, c)`` of each ``sq x sk`` matrix is masked when ``c > r``, as in
    the forward, and its gradient is zero.

    Args:
        grad_output (Tensor): fp16, bf16 or fp32 ``[attn_batches, sq, sk]``
            gradient of the forward's output, on the active device. A
            non-contiguous input is copied.
        output (Tensor): the forward's output for the same input and scale, in
            the same shape, dtype and device. A non-contiguous input is copied.
        scale (float): the multiplier the forward was given. The forward scales
            by ``scale * log2(e)`` internally; that factor cancels out of the
            derivative and must not be included here.

    Returns:
        Tensor: gradient with respect to the forward's input, in
        ``grad_output``'s dtype. Masked entries are zero.
    """
    logger.debug("TRAIN_ASCEND SCALED_UPPER_TRIANG_MASKED_SOFTMAX_BACKWARD")
    _check_input(grad_output, "grad_output", _SCALED_UPPER_TRIANG_MASKED_SOFTMAX_DTYPES)
    _check_input(output, "output", _SCALED_UPPER_TRIANG_MASKED_SOFTMAX_DTYPES)
    _check_shape(grad_output)
    if grad_output.shape != output.shape:
        raise ValueError(
            f"{_NAME} needs grad_output and output to have the same shape, got "
            f"{tuple(grad_output.shape)} and {tuple(output.shape)}"
        )
    if grad_output.dtype != output.dtype:
        raise ValueError(
            f"{_NAME} needs grad_output and output to share a dtype, got "
            f"{grad_output.dtype} and {output.dtype}"
        )
    if grad_output.device != output.device:
        raise ValueError(
            f"{_NAME} needs grad_output and output on the same device, got "
            f"{grad_output.device} and {output.device}"
        )
    scale = _check_scale(scale)

    if grad_output.numel() == 0:
        # No launch when there is no row to reduce over.
        return torch.empty_like(grad_output)

    n_cols = grad_output.shape[2]
    tile_cols = triton.next_power_of_2(n_cols)
    if tile_cols > _MAX_BLOCK_N or (
        grad_output.dtype == torch.float32 and tile_cols > _MAX_ONE_TILE_FP32
    ):
        # The row does not fit one tile; walk it in chunks.
        return _wide_backward(grad_output, output, scale)

    dy = grad_output.contiguous().view(-1)
    y = output.contiguous().view(-1)
    grad_input = torch.empty_like(dy)
    n_rows = dy.numel() // n_cols
    block_m, block_n = _tile_config(n_cols, grad_output.dtype)
    even_m = n_rows % block_m == 0
    # Grid capped at the core count; the stride loop absorbs the rest.
    grid = (min(triton.cdiv(n_rows, block_m), CORE_NUM),)

    with torch_device_fn.device(grad_output.device):
        _scaled_upper_triang_masked_softmax_backward_kernel[grid](
            dy,
            y,
            grad_input,
            n_rows,
            n_cols,
            scale,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            EVEN_M=even_m,
        )
    return grad_input.view(grad_output.shape)
