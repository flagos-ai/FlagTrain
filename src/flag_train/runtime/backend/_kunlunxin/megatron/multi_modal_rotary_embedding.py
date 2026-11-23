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

"""Multimodal rotary frequencies with XPU3-specific launch geometry."""

import torch
import triton
import triton.language as tl

from flag_train.megatron.multi_modal_rotary_embedding import (
    multi_modal_rotary_embedding as _generic,
)
from flag_train.runtime import torch_device_fn


@triton.jit
def _frequencies(
    positions,
    frequencies,
    output,
    scale,
    BATCH: tl.constexpr,
    WIDTH: tl.constexpr,
    CUT0: tl.constexpr,
    CUT1: tl.constexpr,
    STRIDE_MODE: tl.constexpr,
    STRIDE_BATCH: tl.constexpr,
    STRIDE_SEQ: tl.constexpr,
    STRIDE_FREQ: tl.constexpr,
    COLS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, COLS)
    batch_index = row % BATCH
    seq_index = row // BATCH
    offset = (
        batch_index.to(tl.int64) * STRIDE_BATCH + seq_index.to(tl.int64) * STRIDE_SEQ
    )

    p0 = tl.load(positions + offset).to(tl.float32)
    p1 = tl.load(positions + offset + STRIDE_MODE).to(tl.float32)
    p2 = tl.load(positions + offset + 2 * STRIDE_MODE).to(tl.float32)
    p = tl.where(cols < CUT0, p0, tl.where(cols < CUT1, p1, p2))
    freq = tl.load(frequencies + cols.to(tl.int64) * STRIDE_FREQ, cols < WIDTH, 0)
    value = (p * scale) * freq
    base = output + row * (2 * WIDTH) + cols
    tl.store(base, value, mask=cols < WIDTH)
    tl.store(base + WIDTH, value, mask=cols < WIDTH)


def _native_frequencies(position_ids, inv_freq, sections, factor):
    # Select each modality before multiplying to avoid building all three
    # modality/column combinations and discarding two thirds of the result.
    p = position_ids.to(dtype=inv_freq.dtype).transpose(1, 2)
    if factor is not None:
        p = p * (1.0 / float(factor))
    parts = [p[i].unsqueeze(-1) * f for i, f in enumerate(inv_freq.split(sections))]
    return torch.cat(parts + parts, dim=-1).unsqueeze(-2)


def _run(position_ids, inv_freq, mrope_section, seq_len_interpolation_factor):
    if position_ids.ndim == 2:
        position_ids = position_ids.unsqueeze(1)
    if position_ids.ndim != 3 or position_ids.shape[0] != 3 or inv_freq.ndim != 1:
        raise ValueError("expected position_ids [3, B, S] and inv_freq [D]")
    if isinstance(mrope_section, torch.Tensor):
        mrope_section = mrope_section.reshape(-1).tolist()
    sections = tuple(int(size) for size in mrope_section)
    width = inv_freq.numel()
    if len(sections) != 3 or min(sections) < 0 or sum(sections) != width:
        raise ValueError(
            "mrope_section must contain three widths that sum to inv_freq.numel()"
        )
    batch, length = position_ids.shape[1:]
    if batch * length >= 128 or width > 128:
        return _native_frequencies(
            position_ids, inv_freq, sections, seq_len_interpolation_factor
        )
    output = torch.empty(
        (length, batch, 1, 2 * width), device=inv_freq.device, dtype=torch.float32
    )
    if not output.numel():
        return output
    mode, batch_stride, seq = position_ids.stride()
    factor = seq_len_interpolation_factor
    scale = 1.0 if factor is None else 1.0 / float(factor)
    columns = triton.next_power_of_2(width)
    grid = (batch * length,)
    _frequencies[grid](
        position_ids,
        inv_freq,
        output,
        scale,
        batch,
        width,
        sections[0],
        sections[0] + sections[1],
        mode,
        batch_stride,
        seq,
        inv_freq.stride(0),
        COLS=columns,
        num_warps=4,
    )
    return output


def multi_modal_rotary_embedding(
    position_ids, inv_freq, mrope_section, seq_len_interpolation_factor=None
):
    if (
        inv_freq.device != position_ids.device
        or inv_freq.dtype != torch.float32
        or position_ids.dtype != torch.int64
    ):
        return _generic(
            position_ids, inv_freq, mrope_section, seq_len_interpolation_factor
        )
    with torch_device_fn.device(position_ids.device):
        if triton.runtime.driver.active.get_current_target().arch != 3:
            return _generic(
                position_ids, inv_freq, mrope_section, seq_len_interpolation_factor
            )
        return _run(position_ids, inv_freq, mrope_section, seq_len_interpolation_factor)
