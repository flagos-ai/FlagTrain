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

# log2(e), folded into ``scale`` so the kernels run the softmax in base 2, where
# exp2 is a single SFU instruction.
_LOG2E = 1.4426950408889634


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
            "scaled_upper_triang_masked_softmax_forward needs a "
            f"[attn_batches, sq, sk] tensor, got {tuple(x.shape)}"
        )
    if x.shape[1] != x.shape[2]:
        raise ValueError(
            "scaled_upper_triang_masked_softmax_forward needs sq == sk for the "
            f"upper-triangular mask, got sq={x.shape[1]} sk={x.shape[2]}"
        )


@triton.jit
def _load_diagonal_chunk(x_ptr, base, cols, pos, scale):
    """Scaled fp32 chunk crossing the diagonal, ``-inf`` on the masked lanes.

    Only this chunk compares columns against the row position; the masking
    follows the scaling, so a negative ``scale`` cannot produce ``+inf``.
    """
    causal = cols <= pos
    x = tl.load(x_ptr + base + cols, mask=causal, other=0.0).to(tl.float32)
    return tl.where(causal, x * scale, float("-inf"))


@triton.jit
def _online_max_sum(x, m, z):
    """Fold one chunk into the running max and the rescaled exponential sum."""
    m_new = tl.maximum(m, tl.max(x, 0))
    # exp2(-inf - -inf) is NaN; leave z alone while every lane is still masked.
    z = tl.where(
        m_new == float("-inf"),
        z,
        z * tl.math.exp2(m - m_new) + tl.sum(tl.math.exp2(x - m_new), 0),
    )
    return m_new, z


@libentry()
@triton.jit
def _scaled_upper_triang_masked_softmax_forward_kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ONE_TILE: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    """``y = softmax(scale * x)`` with the upper triangle masked out.

    A column is kept when ``col <= row_pos``, where ``row_pos`` is the row's index
    inside its own ``sq x sk`` matrix. Masked entries are stored as zero, as
    Megatron's kernel does. A program owns ``BLOCK_M`` consecutive rows in the
    single-tile branch, and exactly one row in the causal-prefix branch.

    ``scale`` arrives pre-multiplied by log2(e), and every exponentiation below is
    ``exp2``.
    """
    if ONE_TILE:
        pid = tle.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        row_pos = rows % n_cols
        base = rows.to(tl.int64) * n_cols
        cols = tl.arange(0, BLOCK_N)
        in_range = cols < n_cols
        causal = in_range[None, :] & (cols[None, :] <= row_pos[:, None])
        mask = causal
        store_mask = in_range[None, :]
        # No program straddles the end of the tensor, so the row guards can go.
        # The column mask above already covers a tile wider than the row.
        if not EVEN_M:
            row_mask = (rows < n_rows)[:, None]
            mask = mask & row_mask
            store_mask = store_mask & row_mask

        x = tl.load(
            x_ptr + base[:, None] + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        x = tl.where(causal, x * scale, float("-inf"))
        m = tl.max(x, 1)
        e = tl.math.exp2(x - m[:, None])
        z = tl.sum(e, 1)
        # One reciprocal per row instead of a division per element.
        y = e * (1.0 / z)[:, None]
        tl.store(
            y_ptr + base[:, None] + cols[None, :],
            y.to(y_ptr.dtype.element_ty),
            mask=store_mask,
        )
    else:
        # Rows too wide for one register tile walk only the kept prefix, so the
        # masked suffix is zero-filled without being read or exponentiated. The
        # prefix length is a runtime value, so the row cannot live in registers:
        # an online max/sum pass is followed by a write pass that re-reads it.
        # The grid is one program per (row position, batch), which makes the
        # position the program id and leaves no row guards to carry.
        pos = tle.program_id(0)
        base = (tle.program_id(1) * n_cols + pos) * n_cols
        col_arange = tl.arange(0, BLOCK_N)
        prefix_end = pos + 1
        # A chunk ending at or before the row position holds only kept columns and
        # takes no predicate; at most one chunk crosses the diagonal, and peeling
        # that one out keeps both loop bodies branch-free.
        full_end = (prefix_end // BLOCK_N) * BLOCK_N

        m = tl.full([], float("-inf"), tl.float32)
        z = tl.zeros([], tl.float32)
        for start in range(0, full_end, BLOCK_N):
            cols = start + col_arange
            x = tl.load(x_ptr + base + cols).to(tl.float32)
            x = x * scale
            m, z = _online_max_sum(x, m, z)
        if full_end < prefix_end:
            cols = full_end + col_arange
            x = _load_diagonal_chunk(x_ptr, base, cols, pos, scale)
            m, z = _online_max_sum(x, m, z)

        inv_z = 1.0 / z
        for start in range(0, full_end, BLOCK_N):
            cols = start + col_arange
            x = tl.load(x_ptr + base + cols).to(tl.float32)
            x = x * scale
            y = tl.math.exp2(x - m) * inv_z
            tl.store(y_ptr + base + cols, y.to(y_ptr.dtype.element_ty))
        if full_end < prefix_end:
            # Masked lanes come out as exp2(-inf - m) == 0, so this store writes
            # its own zeros.
            cols = full_end + col_arange
            x = _load_diagonal_chunk(x_ptr, base, cols, pos, scale)
            y = tl.math.exp2(x - m) * inv_z
            tl.store(
                y_ptr + base + cols,
                y.to(y_ptr.dtype.element_ty),
                mask=cols < n_cols,
            )

        # Everything past the prefix is masked: a pure zero fill.
        zero = tl.zeros([BLOCK_N], y_ptr.dtype.element_ty)
        pad_end = tl.cdiv(prefix_end, BLOCK_N) * BLOCK_N
        for start in range(pad_end, n_cols, BLOCK_N):
            cols = start + col_arange
            tl.store(y_ptr + base + cols, zero, mask=cols < n_cols)


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


def scaled_upper_triang_masked_softmax_forward(x, scale):
    """``softmax(scale * x)`` under Megatron's upper-triangular mask.

    Entry ``(r, c)`` of each ``sq x sk`` matrix is masked when ``c > r``, the
    same mask Megatron-LM's fused kernel applies for causal attention scores.

    Args:
        x (Tensor): fp16, bf16 or fp32 tensor of shape ``[attn_batches, sq, sk]``
            on the active device, with ``sq == sk``. A non-contiguous input is
            copied. An empty input is returned without a launch.
        scale (float): Multiplier applied to ``x`` before the softmax.

    Returns:
        Tensor: same shape and dtype as ``x``. Masked entries are zero.
    """
    logger.debug("TRAIN SCALED_UPPER_TRIANG_MASKED_SOFTMAX_FORWARD")
    _check_input(
        x,
        "scaled_upper_triang_masked_softmax_forward",
        _SCALED_UPPER_TRIANG_MASKED_SOFTMAX_DTYPES,
    )
    _check_shape(x)
    # Fold log2(e) in, so the kernels can exponentiate in base 2.
    scale = _check_scale(scale) * _LOG2E

    if x.numel() == 0:
        # No launch when there is no row to normalize.
        return torch.empty_like(x)

    flat = x.contiguous().view(-1)
    out = torch.empty_like(flat)
    n_cols = x.shape[2]
    n_rows = flat.numel() // n_cols
    block_m, block_n, one_tile, even_m, num_warps = _tile_config(n_cols, n_rows)

    if one_tile:
        grid = (triton.cdiv(n_rows, block_m),)
    else:
        # One program per (row position, batch).
        grid = (n_cols, n_rows // n_cols)
    with torch_device_fn.device(x.device):
        _scaled_upper_triang_masked_softmax_forward_kernel[grid](
            flat,
            out,
            n_rows,
            n_cols,
            scale,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            ONE_TILE=one_tile,
            EVEN_M=even_m,
            num_warps=num_warps,
        )
    return out.view(x.shape)
