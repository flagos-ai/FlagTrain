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

_SCALED_UPPER_TRIANG_MASKED_SOFTMAX_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)

# Elements (not bytes) in one program's tile.  8192 fp32 lanes are 32 KB for x
# and 32 KB for e; a single wide row fits the 192 KB UB, while a multi-row fp32
# tile at BLOCK_N >= 4096 needs the cap in ``_tile_config``.
_TILE_ELEMENTS = 8192

# Cap rows per program so short rows still spread over the cores.
_MAX_BLOCK_M = 32

# Widest row the single-tile kernel serves; wider rows take the chunked kernel.
_MAX_BLOCK_N = 8192

# ``coreDim`` cap for the chunked wide-row path.
_MAX_PROGRAMS = 65535

# Columns one chunk of the wide-row path walks.
_WIDE_BLOCK_N = 1024

# log2(e), folded into ``scale`` so the kernels run the softmax in base 2, where
# exp2 is a single vector instruction.
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
    EVEN_M: tl.constexpr,
):
    """``y = softmax(scale * x)`` with the upper triangle masked out.

    Kept columns satisfy ``col <= row_pos``; masked entries are stored as zero.
    A program owns ``BLOCK_M`` rows and strides over the row blocks; ``scale``
    is pre-multiplied by log2(e) and every exponentiation is ``exp2``.
    """
    n_blocks = tl.cdiv(n_rows, BLOCK_M)
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    # Negative scale: flip x so the shifted exponent stays <= 0 with a max.
    sgn = tl.where(scale < 0.0, -1.0, 1.0)
    for block in range(pid, n_blocks, n_progs):
        rows = block * BLOCK_M + tl.arange(0, BLOCK_M)
        row_pos = rows % n_cols
        base = rows.to(tl.int64) * n_cols
        cols = tl.arange(0, BLOCK_N)
        in_range = cols < n_cols
        causal = in_range[None, :] & (cols[None, :] <= row_pos[:, None])
        mask = causal
        store_mask = in_range[None, :]
        # The stride walk never starts past n_rows; the guard is only needed
        # when the tile does not divide the row count.
        if not EVEN_M:
            row_mask = (rows < n_rows)[:, None]
            mask = mask & row_mask
            store_mask = store_mask & row_mask

        x = tl.load(
            x_ptr + base[:, None] + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        # Shift before scaling so fp32 rounding follows the row's spread.
        m = tl.max(tl.where(causal, x * sgn, float("-inf")), 1) * sgn
        d = tl.where(causal, (x - m[:, None]) * scale, float("-inf"))
        e = tl.math.exp2(d)
        z = tl.sum(e, 1)
        # One reciprocal per row instead of a division per element.
        y = e * (1.0 / z)[:, None]
        tl.store(
            y_ptr + base[:, None] + cols[None, :],
            y.to(y_ptr.dtype.element_ty),
            mask=store_mask,
        )


@triton.jit
def _load_diagonal_chunk(x_ptr, base, cols, pos, scale):
    """Scaled fp32 chunk crossing the diagonal, ``-inf`` on the masked lanes."""
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
def _scaled_upper_triang_masked_softmax_forward_wide_kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    scale,
    BLOCK_N: tl.constexpr,
):
    """Chunked causal-prefix softmax for rows wider than the UB tile.

    One program walks rows with stride ``n_programs``; per row it keeps a
    running max/sum over the kept prefix, then re-reads it to write.
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
            # Masked lanes give exp2(-inf - m) == 0, so this store writes zeros.
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


def _tile_config(n_cols, dtype):
    """Tile shape for rows of ``n_cols`` columns: rows, columns."""
    block_n = triton.next_power_of_2(n_cols)
    block_m = max(1, min(_MAX_BLOCK_M, _TILE_ELEMENTS // block_n))
    # A 2 x 4096 fp32 tile overflows the UB at compile time; one row fits.
    # Narrower rows keep the wider tile.
    if dtype == torch.float32 and block_n >= 4096:
        block_m = 1
    return block_m, block_n


def _wide_forward(x, scale):
    """Rows whose next power of two exceeds the UB tile."""
    flat = x.contiguous().view(-1)
    out = torch.empty_like(flat)
    n_cols = x.shape[2]
    n_rows = flat.numel() // n_cols
    grid = (min(n_rows, _MAX_PROGRAMS),)
    with torch_device_fn.device(x.device):
        _scaled_upper_triang_masked_softmax_forward_wide_kernel[grid](
            flat,
            out,
            n_rows,
            n_cols,
            scale * _LOG2E,
            BLOCK_N=_WIDE_BLOCK_N,
        )
    return out.view(x.shape)


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
    logger.debug("TRAIN_ASCEND SCALED_UPPER_TRIANG_MASKED_SOFTMAX_FORWARD")
    _check_input(
        x,
        "scaled_upper_triang_masked_softmax_forward",
        _SCALED_UPPER_TRIANG_MASKED_SOFTMAX_DTYPES,
    )
    _check_shape(x)
    scale = _check_scale(scale)

    if x.numel() == 0:
        # No launch when there is no row to normalize.
        return torch.empty_like(x)

    n_cols = x.shape[2]
    if triton.next_power_of_2(n_cols) > _MAX_BLOCK_N:
        # The row does not fit the UB tile; walk it in chunks.
        return _wide_forward(x, scale)

    flat = x.contiguous().view(-1)
    out = torch.empty_like(flat)
    n_rows = flat.numel() // n_cols
    block_m, block_n = _tile_config(n_cols, x.dtype)
    even_m = n_rows % block_m == 0
    # Grid capped at the core count; the stride loop absorbs the rest.
    grid = (min(triton.cdiv(n_rows, block_m), CORE_NUM),)

    with torch_device_fn.device(x.device):
        _scaled_upper_triang_masked_softmax_forward_kernel[grid](
            flat,
            out,
            n_rows,
            n_cols,
            # Fold log2(e) in, so the kernel can exponentiate in base 2.
            scale * _LOG2E,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            EVEN_M=even_m,
        )
    return out.view(x.shape)
