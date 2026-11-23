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


"""Multimodal (mrope) rotary embedding frequencies -- Triton implementation.

Reference semantics (Megatron-LM, non-interleaved multimodal RoPE):

  position_ids: int64 [3, B, S]; inv_freq: fp32 [D]; mrope_section = (c0, c1, c2)
  with c0 + c1 + c2 == D; out: fresh contiguous fp32 [S, B, 1, 2D]

  jr = j % D
  msel(jr) = 0 if jr < c0 else (1 if jr < c0 + c1 else 2)
  emb[s, b, 0, j] = float32(position_ids[msel(jr), b, s]) * inv_freq[jr] * scale

One fused store-bound kernel writes the whole output; no intermediate tensors.
"""

import torch
import triton
import triton.language as tl

from flag_train.runtime import torch_device_fn


@triton.jit
def _mrope_freqs_kernel(
    pos_ptr,  # *i64  [3, B, S]
    freq_ptr,  # *fp32 [D]
    out_ptr,  # *fp32 [S, B, 1, 2D]
    N_ROWS: tl.constexpr,  # S * B
    B: tl.constexpr,  # batch
    SM: tl.constexpr,  # position_ids stride along modality axis
    SB: tl.constexpr,  # position_ids stride along batch axis
    SS: tl.constexpr,  # position_ids stride along sequence axis
    SF: tl.constexpr,  # inv_freq stride (usually 1)
    FACTOR: tl.constexpr,  # seq_len_interpolation_factor (ignored if !HAS_INTERP)
    D: tl.constexpr,
    M: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    HAS_INTERP: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)  # flat row id = s * B + b
    rmask = rows < N_ROWS

    # decode (s, b) from the flat output row id (sequence-major layout)
    s = rows // B
    b = rows - s * B
    p_off = b.to(tl.int64) * SB + s.to(tl.int64) * SS

    p0 = tl.load(pos_ptr + p_off, mask=rmask, other=0).to(tl.float32)
    p1 = tl.load(pos_ptr + SM + p_off, mask=rmask, other=0).to(tl.float32)
    p2 = tl.load(pos_ptr + 2 * SM + p_off, mask=rmask, other=0).to(tl.float32)
    if HAS_INTERP:
        p0 = p0 / FACTOR
        p1 = p1 / FACTOR
        p2 = p2 / FACTOR

    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cols < M
    jr = tl.where(cols < D, cols, cols - D)  # j % D
    f = tl.load(freq_ptr + jr.to(tl.int64) * SF, mask=cmask, other=0.0)

    sel0 = jr < C0
    sel1 = jr < (C0 + C1)
    pos = tl.where(
        sel0[None, :], p0[:, None], tl.where(sel1[None, :], p1[:, None], p2[:, None])
    )
    val = pos * f[None, :]

    if WIDE:
        roff = rows.to(tl.int64) * M
    else:
        roff = rows * M
    offs = roff[:, None] + cols[None, :]
    tl.store(out_ptr + offs, val, mask=rmask[:, None] & cmask[None, :])


def _as_int_list(value):
    if torch.is_tensor(value):
        return [int(v) for v in value.reshape(-1).tolist()]
    return [int(v) for v in value]


def _make_entry(position_ids, inv_freq, mrope_section, factor_arg):
    S = position_ids.shape[2]
    B = position_ids.shape[1]
    D = inv_freq.shape[0]
    M = 2 * D
    n_rows = S * B

    sec = _as_int_list(mrope_section)
    if len(sec) != 3 or min(sec) < 0 or sum(sec) != D:
        raise ValueError(
            "mrope_section must contain three widths that sum to inv_freq.numel()"
        )
    c0, c1, _ = sec

    has_interp = factor_arg is not None
    factor = float(factor_arg) if has_interp else 1.0

    block_c = triton.next_power_of_2(M) if M <= 1024 else 1024
    rows_target = max(1, 4096 // block_c)
    block_r = 1
    while block_r * 2 <= rows_target and (n_rows // (block_r * 2)) >= 264:
        block_r *= 2

    tile = block_r * block_c
    num_warps = max(1, min(8, tile // 512))
    grid = (triton.cdiv(n_rows, block_r), triton.cdiv(M, block_c))

    stride_m, stride_b, stride_s = position_ids.stride()
    args = (
        n_rows,
        B,
        stride_m,
        stride_b,
        stride_s,
        inv_freq.stride(0),
        factor,
        D,
        M,
        c0,
        c1,
        has_interp,
        n_rows * M >= 2**31,
        block_r,
        block_c,
    )
    out_shape = (S, B, 1, M)
    is_empty = n_rows == 0 or M == 0 or D == 0
    return out_shape, grid, num_warps, args, is_empty


def multi_modal_rotary_embedding(
    position_ids, inv_freq, mrope_section, seq_len_interpolation_factor=None
):
    """Return the non-interleaved rotary frequencies in sequence-major layout."""
    if position_ids.ndim == 2:
        position_ids = position_ids.unsqueeze(1)
    if position_ids.ndim != 3 or position_ids.shape[0] != 3 or inv_freq.ndim != 1:
        raise ValueError("expected position_ids [3, B, S] and inv_freq [D]")
    if position_ids.device != inv_freq.device:
        raise ValueError("position_ids and inv_freq must be on the same device")
    shape, grid, num_warps, args, is_empty = _make_entry(
        position_ids, inv_freq, mrope_section, seq_len_interpolation_factor
    )
    out = torch.empty(shape, dtype=torch.float32, device=position_ids.device)
    if not is_empty:
        with torch_device_fn.device(position_ids.device):
            _mrope_freqs_kernel[grid](
                position_ids, inv_freq, out, *args, num_warps=num_warps
            )
    return out
