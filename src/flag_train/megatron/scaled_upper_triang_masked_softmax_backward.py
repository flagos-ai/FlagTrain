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
from flag_train.utils import libentry
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_NAME = "scaled_upper_triang_masked_softmax_backward"

_SCALED_UPPER_TRIANG_MASKED_SOFTMAX_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)

# Widest row the single-tile kernel serves. Short rows pack many to a program and
# read them as one contiguous span; wider rows run faster on the causal-prefix
# path, whose register footprint does not grow with the row.
_MAX_ONE_TILE = 512

# Elements (not bytes) per program when the row fits one tile: 16 fp32 values per
# thread at 4 warps, which measured faster than a wider tile.
_TILE_ELEMENTS = 2048

# Cap on rows per program, so a short row does not pick a tile that is mostly
# unwritten rows.
_MAX_BLOCK_M = 32

# Column width the causal-prefix path walks per chunk. Rows of 8192 columns and
# up amortize a 1024-wide chunk; narrower rows do not.
_CAUSAL_BLOCK_N = 512
_CAUSAL_WIDE_BLOCK_N = 1024
_CAUSAL_WIDE_MIN_COLS = 8192

# Target elements per thread when choosing num_warps from the tile size.
_ELEMENTS_PER_THREAD = 16


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


def _tile_config(n_cols, n_rows):
    """Launch shape and mask flags for a ``[n_rows, n_cols]`` row set."""
    if n_cols <= _MAX_ONE_TILE:
        block_n = triton.next_power_of_2(n_cols)
        block_m = max(1, min(_MAX_BLOCK_M, _TILE_ELEMENTS // block_n))
        one_tile = True
    else:
        if n_cols < _CAUSAL_WIDE_MIN_COLS:
            block_n = _CAUSAL_BLOCK_N
        else:
            block_n = _CAUSAL_WIDE_BLOCK_N
        block_m = 1
        one_tile = False

    warp = 32 * _ELEMENTS_PER_THREAD
    num_warps = max(1, min(16, (block_m * block_n + warp - 1) // warp))
    even_m = n_rows % block_m == 0
    return block_m, block_n, one_tile, even_m, num_warps


@triton.jit
def _load_masked_pair(dy_ptr, y_ptr, base, cols, pos):
    """``dy`` and ``y`` for a chunk crossing the diagonal, zero on masked lanes.

    Only this chunk compares columns against the row position, so the loads
    elsewhere in the row take no predicate.
    """
    keep = cols <= pos
    dy = tl.load(dy_ptr + base + cols, mask=keep, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + base + cols, mask=keep, other=0.0).to(tl.float32)
    return dy, y


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
    ONE_TILE: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    """``dx = scale * y * (dy - rowsum(dy * y))``, zero above the diagonal.

    Rows are laid out as in the forward kernel, and so is the branch: ``BLOCK_M``
    consecutive rows per program when a row fits one tile, one row per program
    otherwise.
    """
    if ONE_TILE:
        pid = tle.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        row_pos = rows % n_cols
        base = rows.to(tl.int64) * n_cols
        cols = tl.arange(0, BLOCK_N)
        in_range = cols < n_cols
        causal = in_range[None, :] & (cols[None, :] <= row_pos[:, None])
        store_mask = in_range[None, :]
        # No program straddles the end of the tensor, so the row guards can go.
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
    else:
        # One program per (row position, batch), as in the forward: the position
        # is the program id, so the prefix length needs no row guards.
        pos = tle.program_id(0)
        base = (tle.program_id(1) * n_cols + pos) * n_cols
        col_arange = tl.arange(0, BLOCK_N)
        prefix_end = pos + 1
        # At most one chunk crosses the diagonal, and peeling it out keeps both
        # loop bodies branch-free.
        full_end = (prefix_end // BLOCK_N) * BLOCK_N

        total = tl.zeros([], tl.float32)
        for start in range(0, full_end, BLOCK_N):
            cols = start + col_arange
            dy = tl.load(dy_ptr + base + cols).to(tl.float32)
            y = tl.load(y_ptr + base + cols).to(tl.float32)
            total += tl.sum(dy * y, 0)
        if full_end < prefix_end:
            cols = full_end + col_arange
            dy, y = _load_masked_pair(dy_ptr, y_ptr, base, cols, pos)
            total += tl.sum(dy * y, 0)

        # The row sum is only known now, so the prefix is walked a second time
        # rather than held in registers: that is the whole point of this branch.
        for start in range(0, full_end, BLOCK_N):
            cols = start + col_arange
            dy = tl.load(dy_ptr + base + cols).to(tl.float32)
            y = tl.load(y_ptr + base + cols).to(tl.float32)
            dx = y * (dy - total) * scale
            tl.store(dx_ptr + base + cols, dx.to(dx_ptr.dtype.element_ty))
        if full_end < prefix_end:
            cols = full_end + col_arange
            dy, y = _load_masked_pair(dy_ptr, y_ptr, base, cols, pos)
            dx = tl.where(cols <= pos, y * (dy - total) * scale, 0.0)
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
    logger.debug("TRAIN SCALED_UPPER_TRIANG_MASKED_SOFTMAX_BACKWARD")
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

    dy = grad_output.contiguous().view(-1)
    y = output.contiguous().view(-1)
    grad_input = torch.empty_like(dy)
    n_cols = grad_output.shape[2]
    n_rows = dy.numel() // n_cols
    block_m, block_n, one_tile, even_m, num_warps = _tile_config(n_cols, n_rows)

    if one_tile:
        grid = (triton.cdiv(n_rows, block_m),)
    else:
        # One program per (row position, batch).
        grid = (n_cols, n_rows // n_cols)
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
            ONE_TILE=one_tile,
            EVEN_M=even_m,
            num_warps=num_warps,
        )
    return grad_input.view(grad_output.shape)
