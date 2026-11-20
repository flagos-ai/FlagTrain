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
"""MetaX-tuned fused RMSNorm/SiLU gating.

The generic implementation is ``flag_train.megatron.fused_gated_norm`` and is
what every backend without an override runs. The arithmetic here is that
module's, unchanged -- same casts, same rounding to the activation dtype before
the FP32 gating multiply, same checks, and the strided addressing is imported
from it rather than rewritten. Two things differ, both measured on a C550.

**The launch tile is wider, and its optimum depends on the activation dtype.**
The kernel is bandwidth-bound, so a program wants enough contiguous elements
per thread to issue wide loads. At ``(2, 1024, 16, 128)`` the generic tile is
four rows on two warps, which is four fp16 elements per thread (8 B) and
measures 588 GB/s; eight rows on one warp is eight elements per thread (16 B)
and measures 691 GB/s. The whole float16 sweep follows: eight rows on one warp
for a 128-wide head, four for 256, two for 512.

Bfloat16 does not follow it. Every element of the normalised tile is converted
to the activation dtype and back, and the store converts again; on this device
those are explicit ``cvt.f32tobf16`` instructions, so a bfloat16 tile carries
roughly twice the instruction count of its float16 twin and wants a tile wide
enough to amortise the setup instead of the eight-row one. At
``(2, 1024, 16, 128)`` eight rows measure 0.83 of the baseline, four rows 0.78
and thirty-two rows on eight warps 0.86; the ordering holds at 2048 and at 32
heads. Past ``_LARGE_ROWS`` the tensor is bandwidth-bound and the generic
eight-row tile is again the faster one. A float32 gate doubles that stream and
wants the small tile even there: thirty-two rows is 0.62 against 0.80 for four.
Narrower heads keep the generic tile for bfloat16.

A launch with fewer rows than a wave is dominated by its own setup rather than
by the tile, so it gets one row per program: 16 rows over 16 programs takes
14.1 us against 14.3 for two programs of eight.

**SiLU is evaluated on exp2.** ``gv / (1 + exp2(-gv * log2(e)))`` is the same
function as ``gv * sigmoid(gv)`` and rounds to the same output here (checked
against the reference: identical maximum error in both dtypes), but the
base-two exponential lowers to the device's own instruction while the
base-``e`` one takes the libdevice path. Worth 1.5-2% end to end -- small
because the tile is mostly waiting on memory, but free.

What is deliberately *not* changed is the arithmetic: the normalised
activation is still rounded to the activation dtype before the FP32 gating
multiply, which is the contract the correctness suite pins.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_train.megatron.fused_gated_norm import _check_arguments, _row_offset
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry

logger = logging.getLogger(__name__)

# The widest tile the selector asks for, and the row count past which the
# bfloat16 128-wide tile stops helping. See the module docstring.
_LARGE_ROWS = 1 << 17
_SMALL_ROWS = 64

# log2(e): SiLU is evaluated on exp2, as the generic kernel's softmax relatives
# already do, so the conversion rides on the exponent rather than the base.
_LOG2E = tl.constexpr(1.4426950408889634)


@libentry()
@triton.jit
def _gated_norm_fwd_metax(
    x,
    gate,
    weight,
    out,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    SEQUENCE: tl.constexpr,
    X_STRIDES: tl.constexpr,
    GATE_STRIDES: tl.constexpr,
    X_CONTIGUOUS: tl.constexpr,
    GATE_CONTIGUOUS: tl.constexpr,
    EPS: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
    BT: tl.constexpr,
    EVEN_ROWS: tl.constexpr,
    I64: tl.constexpr,
):
    """The generic kernel's body with SiLU evaluated on exp2."""
    if I64:
        row = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    else:
        row = tl.program_id(0) * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    off = row[:, None] * D + d[None, :]
    if X_CONTIGUOUS:
        x_ptr = x + off
    else:
        xrow = _row_offset(
            row, SEQUENCE, HEADS, X_STRIDES[0], X_STRIDES[1], X_STRIDES[2]
        )
        x_ptr = x + xrow[:, None] + d[None, :] * X_STRIDES[3]
    if GATE_CONTIGUOUS:
        gate_ptr = gate + off
    else:
        grow = _row_offset(
            row, SEQUENCE, HEADS, GATE_STRIDES[0], GATE_STRIDES[1], GATE_STRIDES[2]
        )
        gate_ptr = gate + grow[:, None] + d[None, :] * GATE_STRIDES[3]

    # Issue both input streams before the reduction so they overlap in flight.
    if EVEN_ROWS:
        xv = tl.load(x_ptr, eviction_policy="evict_first").to(tl.float32)
        gate_raw = tl.load(gate_ptr, eviction_policy="evict_first")
    else:
        valid = row[:, None] < ROWS
        xv = tl.load(x_ptr, valid, 0, eviction_policy="evict_first").to(tl.float32)
        gate_raw = tl.load(gate_ptr, valid, 0, eviction_policy="evict_first")

    w = tl.load(weight + d).to(tl.float32)
    if ZERO_CENTERED:
        w += 1
    inv = tl.rsqrt(tl.sum(xv * xv, 1) / D + EPS)
    # Materialise RMSNorm in the activation dtype before FP32 gating.
    norm = (xv * inv[:, None] * w[None, :]).to(x.dtype.element_ty).to(tl.float32)
    gv = gate_raw.to(tl.float32)
    # x * sigmoid(x), on exp2: see the module docstring.
    res = norm * (gv / (1.0 + tl.exp2(-gv * _LOG2E)))
    if EVEN_ROWS:
        tl.store(out + off, res)
    else:
        tl.store(out + off, res, valid)


def _launch_config(
    dim: int, rows: int, dtype, gate_element_size: int
) -> tuple[int, int]:
    """Pick ``(rows per program, warps per program)`` for one launch.

    See the module docstring for the measurements behind each branch. The
    activation dtype is ``x``'s for the tile width; a float32 gate is a wider
    stream and wants the smaller tile instead.
    """
    if rows <= _SMALL_ROWS:
        return 1, 1
    if dtype == torch.bfloat16:
        if dim == 128:
            if rows >= _LARGE_ROWS:
                return 8, 1
            # A float32 gate doubles that stream's traffic; the wide tile that
            # wins for a bfloat16 gate loses on it.
            return (4, 1) if gate_element_size == 4 else (32, 8)
        if dim <= 64:
            return 8, 2
        if dim == 256:
            return 2, 1
        return 1, min(8, max(1, dim // 512))
    if dim <= 128:
        return 8, 1
    if dim == 256:
        return 4, 1
    if dim == 512:
        return 2, 1
    # Wider heads: one row per program, about 512 columns per warp, as the
    # generic launcher does.
    return 1, min(8, max(1, dim // 512))


@torch.compiler.disable
def fused_gated_norm(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    zero_centered: bool = False,
) -> torch.Tensor:
    """Run fused RMSNorm/SiLU gating (forward only).

    Same signature, arithmetic and returned tensor as
    ``flag_train.megatron.fused_gated_norm``; only the launch and the SiLU
    evaluation differ -- see this module's docstring.
    """
    logger.debug("TRAIN_METAX FUSED_GATED_NORM")

    dim = _check_arguments(x, gate, weight)

    batch, sequence, heads, _ = x.shape
    rows = batch * sequence * heads
    x_contiguous = x.is_contiguous()
    gate_contiguous = gate.is_contiguous()
    block_rows, num_warps = _launch_config(dim, rows, x.dtype, gate.element_size())
    grid = triton.cdiv(rows, block_rows)
    # 32-bit index arithmetic is only valid when no address in the padded block
    # can pass 2**31 elements; the generic launcher makes the same test.
    padded_elems = grid * block_rows * dim
    index_i64 = padded_elems >= (1 << 31) or not (x_contiguous and gate_contiguous)
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    with torch_device_fn.device(x.device):
        _gated_norm_fwd_metax[(grid,)](
            x,
            gate,
            weight,
            out,
            ROWS=rows,
            HEADS=heads,
            D=dim,
            SEQUENCE=sequence,
            X_STRIDES=x.stride(),
            GATE_STRIDES=gate.stride(),
            X_CONTIGUOUS=x_contiguous,
            GATE_CONTIGUOUS=gate_contiguous,
            EPS=eps,
            ZERO_CENTERED=zero_centered,
            BT=block_rows,
            EVEN_ROWS=rows % block_rows == 0,
            I64=index_i64,
            num_warps=num_warps,
        )
    return out
