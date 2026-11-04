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
"""Ascend-tuned fused scale + mask + softmax forward.

Same contract as the generic implementation: ``softmax(scale * x)``, with the
bool mask's ``True`` lanes excluded, masked keys and fully masked rows
answering exactly zero. It carries its own kernel because the generic launcher
requests ``num_stages=4`` (at most two are accepted here) and because a program
costs ~73 ns whatever it carries, so the grid is pinned to the vector-core
count and each program walks a contiguous run of row tiles. Per-launch values
are ``constexpr`` (a second runtime scalar costs ~40%); a key length that is
not a power of two takes a 1-D row path, as a 2-D column mask that shortens a
row is miscompiled here. Mask tiles load as one flat run and reshape; the
denominator guard ``1 / (z + 1e-30)`` is exact and free, where a ``where``
costs 7-9%.
"""

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

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# ``FusedScaleMaskSoftmax.is_kernel_available`` rejects ``sk > 4096``.
_MAX_KEY_LEN = 4096

# 32-bit indices unless the addressed span can pass 2**31.
_I64_THRESHOLD = 1 << 31

_LOG2E = 1.4426950408889634

# The measured lanes-per-program plateau.
_TILE_LANES = 8192

_MAX_ROWS = 64


def _check_input(inputs):
    if inputs.ndim != 4:
        raise ValueError(
            f"scaled masked softmax needs a 4-D tensor, got {inputs.ndim}-D"
        )
    if inputs.dtype not in _SUPPORTED_DTYPES:
        wanted = " / ".join(str(dtype) for dtype in _SUPPORTED_DTYPES)
        raise ValueError(
            f"scaled masked softmax needs a {wanted} tensor, got {inputs.dtype}"
        )
    if inputs.device.type == "cpu":
        raise ValueError(
            "scaled masked softmax needs a device tensor, not one on the CPU"
        )
    if inputs.device.type != runtime_device.name:
        raise ValueError(
            "scaled masked softmax needs a "
            f"{runtime_device.name} tensor, got {inputs.device.type}"
        )


def _check_mask(mask, inputs):
    b, _, sq, sk = inputs.shape
    if mask.ndim != 4:
        raise ValueError(
            f"scaled masked softmax mask needs a 4-D tensor, got {mask.ndim}-D"
        )
    if mask.dtype != torch.bool:
        raise ValueError(
            f"scaled masked softmax mask needs a bool tensor, got {mask.dtype}"
        )
    if mask.shape[0] not in (1, b) or mask.shape[1] != 1:
        raise ValueError(
            "scaled masked softmax mask must be [b or 1, 1, sq, sk], got "
            f"{tuple(mask.shape)}"
        )
    if mask.shape[2] != sq or mask.shape[3] != sk:
        raise ValueError(
            "scaled masked softmax mask must match the scores on the query and "
            f"key axes, got {tuple(mask.shape[2:])} for scores {sq} x {sk}"
        )
    if mask.device != inputs.device:
        raise ValueError(
            "scaled masked softmax needs the scores and the mask on one device"
        )


def _check_scale(scale):
    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            raise ValueError(
                f"fused softmax scale needs one element, got {tuple(scale.shape)}"
            )
        scale = scale.item()
    if not isinstance(scale, numbers.Real):
        raise TypeError(f"scale needs a real number, got {type(scale).__name__}")
    return float(scale)


@libentry()
@triton.jit
def _scaled_masked_softmax_forward_kernel(
    x_ptr,
    mask_ptr,
    y_ptr,
    BLOCK_N: tl.constexpr,
    ROWS: tl.constexpr,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    MASK_SHARED: tl.constexpr,
    MB_DIV: tl.constexpr,
    MASK_B_STRIDE: tl.constexpr,
    SCALE: tl.constexpr,
    N_ROWS: tl.constexpr,
    NBLK: tl.constexpr,
    NPROG: tl.constexpr,
    EVEN_ROWS: tl.constexpr,
    SCALAR_ROW: tl.constexpr,
    FLAT_MASK: tl.constexpr,
    I64: tl.constexpr,
    EXP2: tl.constexpr,
):
    """Scale, mask and softmax one row tile per walk step.

    ``SCALAR_ROW`` is the 1-D path; ``FLAT_MASK`` loads the mask tile as one
    contiguous run. Programs take exact contiguous runs of tiles: a strided
    walk overshoots ``NBLK`` and an unpredicated store corrupts the output.
    """
    pid = tl.program_id(0)
    per: tl.constexpr = NBLK // NPROG
    extra: tl.constexpr = NBLK % NPROG
    first = pid * per + tl.minimum(pid, extra)
    stop = first + per + tl.where(pid < extra, 1, 0)

    if SCALAR_ROW:
        col = tl.arange(0, BLOCK_N)
        valid = col < SK
        for idx in tl.range(first, stop):
            if I64:
                idx = idx.to(tl.int64)
            base = idx * SK
            q = idx % SQ
            mb = 0 if MASK_SHARED else idx // MB_DIV
            mo = mb * MASK_B_STRIDE + q * SK
            x = tl.load(x_ptr + base + col, mask=valid, other=0.0).to(tl.float32)
            attended = tl.load(mask_ptr + mo + col, mask=valid, other=1) == 0

            scores = tl.where(attended, x * SCALE, float("-inf"))
            m = tl.max(scores, 0)
            # An all-masked row is ``-inf`` across; pivot at zero so it
            # exponentiates to zeros instead of NaNs.
            m = tl.where(m == float("-inf"), 0.0, m)
            if EXP2:
                e = tl.exp2(scores - m)
            else:
                e = tl.exp(scores - m)
            # ``z + 1e-30`` keeps the reciprocal finite on an all-masked row
            # (its exact zeros then stay zero) and is free, where a ``where``
            # costs 7-9%.
            z = tl.sum(e, 0)
            y = e * (1.0 / (z + 1e-30))
            tl.store(y_ptr + base + col, y.to(y_ptr.dtype.element_ty), mask=valid)
    else:
        col = tl.arange(0, BLOCK_N)
        row_ids = tl.arange(0, ROWS)
        flat = tl.arange(0, ROWS * BLOCK_N)
        for idx in tl.range(first, stop):
            if I64:
                idx = idx.to(tl.int64)
            rows = idx * ROWS + row_ids
            row_ok = rows < N_ROWS
            q = rows % SQ
            mb = 0 if MASK_SHARED else rows // MB_DIV
            base = rows * SK
            mo = mb * MASK_B_STRIDE + q * SK
            offs = base[:, None] + col[None, :]

            if EVEN_ROWS:
                x = tl.load(x_ptr + offs).to(tl.float32)
            else:
                rm = row_ok[:, None]
                x = tl.load(x_ptr + offs, mask=rm, other=0.0).to(tl.float32)
            if FLAT_MASK:
                mo0 = ((idx * ROWS) % SQ) * SK
                if not MASK_SHARED:
                    mo0 = mo0 + ((idx * ROWS) // MB_DIV) * MASK_B_STRIDE
                mk = tl.load(mask_ptr + mo0 + flat)
                attended = tl.reshape(mk, (ROWS, BLOCK_N)) == 0
            elif EVEN_ROWS:
                attended = tl.load(mask_ptr + mo[:, None] + col[None, :]) == 0
            else:
                attended = (
                    tl.load(mask_ptr + mo[:, None] + col[None, :], mask=rm, other=1)
                    == 0
                )

            scores = tl.where(attended, x * SCALE, float("-inf"))
            m = tl.max(scores, 1)
            m = tl.where(m == float("-inf"), 0.0, m)
            if EXP2:
                e = tl.exp2(scores - m[:, None])
            else:
                e = tl.exp(scores - m[:, None])
            # Same guard as the 1-D path.
            z = tl.sum(e, 1)
            y = e * (1.0 / (z + 1e-30))[:, None]
            out = y.to(y_ptr.dtype.element_ty)
            if EVEN_ROWS:
                tl.store(y_ptr + offs, out)
            else:
                tl.store(y_ptr + offs, out, mask=row_ok[:, None])


@torch.compiler.disable
def scaled_masked_softmax_cuda_forward(inputs, mask, scale):
    """Fused scale + mask + softmax, Megatron's masked-attention forward.

    Args:
        inputs (Tensor): fp16, bf16 or fp32 scores of shape ``[b, np, sq, sk]``.
            A non-contiguous tensor is copied.
        mask (Tensor): bool mask of shape ``[b or 1, 1, sq, sk]`` where ``True``
            marks a key that must not be attended to.
        scale (float | Tensor): multiplier applied before the softmax.

    Returns:
        Tensor: contiguous softmax results with the input's shape and dtype.
        Masked keys and fully masked rows are exactly zero.
    """
    logger.debug("TRAIN_ASCEND SCALED_MASKED_SOFTMAX_CUDA_FORWARD")
    _check_input(inputs)
    _check_mask(mask, inputs)
    b, np_, sq, sk = inputs.shape
    if sk > _MAX_KEY_LEN:
        raise ValueError(
            f"scaled masked softmax: key length {sk} exceeds Megatron's "
            f"fused-softmax limit of {_MAX_KEY_LEN}"
        )
    if sk == 0:
        raise ValueError("scaled masked softmax: key length must be positive")
    scale = _check_scale(scale)
    if inputs.numel() == 0:
        return torch.empty_like(inputs, memory_format=torch.contiguous_format)

    # 16-bit kernels fold log2(e) into the scale and use exp2; fp32 keeps exp,
    # whose rounding tolerance the folded argument would exceed.
    exp2 = inputs.dtype != torch.float32
    if exp2:
        scale *= _LOG2E

    x = inputs.contiguous()
    m = mask.contiguous()
    out = torch.empty_like(x)
    n_rows = b * np_ * sq

    # Lane floor of two: a one-lane vector aborts bishengir.
    block_n = max(2, triton.next_power_of_2(sk))
    even_col = sk == block_n
    rows = max(1, min(_TILE_LANES // block_n, _MAX_ROWS)) if even_col else 1
    nblk = triton.cdiv(n_rows, rows)
    grid = min(nblk, CORE_NUM)
    # A flat mask run needs full tiles that cannot cross a query or batch
    # boundary.
    flat_mask = even_col and n_rows % rows == 0 and sq % rows == 0

    mask_shared = m.shape[0] == 1
    with torch_device_fn.device(inputs.device):
        _scaled_masked_softmax_forward_kernel[(grid,)](
            x,
            m.view(torch.uint8),
            out,
            BLOCK_N=block_n,
            ROWS=rows,
            SQ=sq,
            SK=sk,
            MASK_SHARED=mask_shared,
            MB_DIV=np_ * sq,
            MASK_B_STRIDE=0 if mask_shared else m.stride(0),
            SCALE=scale,
            N_ROWS=n_rows,
            NBLK=nblk,
            NPROG=grid,
            EVEN_ROWS=n_rows % rows == 0,
            SCALAR_ROW=not even_col,
            FLAT_MASK=flat_mask,
            I64=n_rows * sk >= _I64_THRESHOLD,
            EXP2=exp2,
            num_stages=1,
        )
    return out
