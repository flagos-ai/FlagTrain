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

"""Non-interleaved mRoPE with Torch position conversion on GCU300."""

import torch
import triton
import triton.language as tl

from flag_train.megatron.multi_modal_rotary_embedding import (
    multi_modal_rotary_embedding as _generic,
)
from flag_train.runtime import torch_device_fn

from .. import arch_version


@triton.jit
def _load_position(positions, offset, mask):
    return tl.load(positions + offset, mask=mask, other=0.0)


@triton.jit
def _frequencies(
    positions,
    frequencies,
    output,
    rows,
    scale,
    BATCH: tl.constexpr,
    WIDTH: tl.constexpr,
    CUT0: tl.constexpr,
    CUT1: tl.constexpr,
    STRIDE_MODE: tl.constexpr,
    STRIDE_BATCH: tl.constexpr,
    STRIDE_SEQ: tl.constexpr,
    STRIDE_FREQ: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    column = tl.program_id(1) * COLS + tl.arange(0, COLS)
    seq = row // BATCH
    seq_batch = row - seq * BATCH
    offset = seq_batch * STRIDE_BATCH + seq * STRIDE_SEQ
    p0 = _load_position(positions, offset, row < rows) * scale
    p1 = _load_position(positions, offset + STRIDE_MODE, row < rows) * scale
    p2 = _load_position(positions, offset + 2 * STRIDE_MODE, row < rows) * scale
    frequency = tl.load(
        frequencies + column * STRIDE_FREQ,
        mask=column < WIDTH,
        other=0,
    )
    position = tl.where(
        column[None, :] < CUT0,
        p0[:, None],
        tl.where(column[None, :] < CUT1, p1[:, None], p2[:, None]),
    )
    value = position * frequency[None, :]
    base = output + row[:, None] * (2 * WIDTH) + column[None, :]
    mask = (row[:, None] < rows) & (column[None, :] < WIDTH)
    tl.store(base, value, mask=mask)
    tl.store(base + WIDTH, value, mask=mask)


def _run(position_ids, inv_freq, mrope_section, seq_len_interpolation_factor):
    if position_ids.ndim == 2:
        position_ids = position_ids.unsqueeze(1)
    if position_ids.ndim != 3 or position_ids.shape[0] != 3 or inv_freq.ndim != 1:
        raise ValueError("expected position_ids [3, B, S] and inv_freq [D]")
    if position_ids.device != inv_freq.device:
        raise ValueError("position_ids and inv_freq must be on the same device")
    if isinstance(mrope_section, torch.Tensor):
        mrope_section = mrope_section.reshape(-1).tolist()
    sections = tuple(int(size) for size in mrope_section)
    width = inv_freq.numel()
    if len(sections) != 3 or min(sections) < 0 or sum(sections) != width:
        raise ValueError(
            "mrope_section must contain three widths that sum to inv_freq.numel()"
        )
    batch, length = position_ids.shape[1:]
    output = torch.empty(
        (length, batch, 1, 2 * width), device=inv_freq.device, dtype=torch.float32
    )
    if output.numel() == 0:
        return output
    if position_ids.dtype != torch.int64:
        raise ValueError("position_ids must have dtype torch.int64")
    positions = position_ids.to(dtype=inv_freq.dtype).contiguous()
    mode, batch_stride, seq = positions.stride()
    factor = seq_len_interpolation_factor
    scale = 1.0 if factor is None else 1.0 / float(factor)
    columns = min(triton.next_power_of_2(width), 256)
    rows = 256 if batch * length >= 128 else 1
    grid = (triton.cdiv(batch * length, rows), triton.cdiv(width, columns))
    _frequencies[grid](
        positions,
        inv_freq,
        output,
        batch * length,
        scale,
        batch,
        width,
        sections[0],
        sections[0] + sections[1],
        mode,
        batch_stride,
        seq,
        inv_freq.stride(0),
        ROWS=rows,
        COLS=columns,
        num_warps=4,
    )
    return output


def multi_modal_rotary_embedding(
    position_ids, inv_freq, mrope_section, seq_len_interpolation_factor=None
):
    if (
        arch_version != 300
        or inv_freq.device != position_ids.device
        or inv_freq.dtype != torch.float32
        or position_ids.dtype != torch.int64
        or position_ids.numel() * inv_freq.numel() >= 2**31
        or (inv_freq.ndim == 1 and inv_freq.numel() * inv_freq.stride(0) >= 2**31)
    ):
        return _generic(
            position_ids, inv_freq, mrope_section, seq_len_interpolation_factor
        )
    with torch_device_fn.device(position_ids.device):
        return _run(position_ids, inv_freq, mrope_section, seq_len_interpolation_factor)
