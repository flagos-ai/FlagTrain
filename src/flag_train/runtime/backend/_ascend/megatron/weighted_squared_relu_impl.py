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
import math
import numbers

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_WEIGHTED_SQUARED_RELU_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)

# Lanes per program.  This is the measured plateau, and also the largest tile
# that fits the clamp's live buffers in UB.
_TILE_LANES = 8192

# Column tile bounds for the weighted kernel.  The floor keeps every constant
# shape at least two lanes wide (a one-lane vector aborts bishengir); the cap
# keeps a row within a handful of column blocks.
_TILE_COLS_MIN = 64
_TILE_COLS_MAX = 1024

# Fewest programs that make the column walk pay for itself (see module doc).
_WALK_MIN_PROGRAMS = 32

# Flat block for the unweighted path, the floor used for tiny tensors, and the
# program count the persistent grid is pinned to when the input is large enough
# to loop over.
_FLAT_BLOCK = 8192
_FLAT_BLOCK_MIN = 128
_FLAT_WALK_PROGRAMS = 512


def _check_input(tensor, name):
    """Validate the dtype and device of the Ascend row-weight kernel.

    Other dtypes/devices are delegated to the generic Triton path before this
    check. Defined here rather than imported so importing this module does not
    re-enter ``SpecOpRegistrar`` mid-scan.
    """
    if tensor.dtype not in _WEIGHTED_SQUARED_RELU_DTYPES:
        wanted = " / ".join(str(dtype) for dtype in _WEIGHTED_SQUARED_RELU_DTYPES)
        raise ValueError(f"{name} needs a {wanted} tensor, got {tensor.dtype}")
    if tensor.device.type == "cpu":
        raise ValueError(f"{name} needs a device tensor, not one on the CPU")
    if tensor.device.type != runtime_device.name:
        raise ValueError(
            f"{name} needs a {runtime_device.name} tensor, got {tensor.device.type}"
        )


def _check_weights(weights):
    """Weights may be any real numeric dtype; the kernel casts to fp32."""
    if weights.dtype.is_complex:
        raise ValueError("weighted_squared_relu_impl weights must be real-valued")
    if weights.device.type == "cpu":
        raise ValueError(
            "weighted_squared_relu_impl weights needs a device tensor, not one on the CPU"
        )
    if weights.device.type != runtime_device.name:
        raise ValueError(
            "weighted_squared_relu_impl weights needs a "
            f"{runtime_device.name} tensor, got {weights.device.type}"
        )


# Weight dtypes torch stores but this device's ``mul`` and kernels reject.
_UNSUPPORTED_WEIGHT_DTYPES = (torch.uint16, torch.uint32, torch.uint64)


def _widen_unsigned_weights(weights, input):
    """Spell out torch's promotion for unsigned weights.

    torch multiplies an integer weight by promoting it to the activation's
    dtype, but this device has neither an ATen ``mul`` nor a kernel load for
    unsigned 16/32/64-bit tensors, so make that promotion explicit.
    """
    if (
        isinstance(weights, torch.Tensor)
        and weights.dtype in _UNSUPPORTED_WEIGHT_DTYPES
        and input.dtype.is_floating_point
    ):
        return weights.to(input.dtype)
    return weights


@triton.jit
def _clamped_square(value, SCALE: tl.constexpr, NEG2: tl.constexpr):
    """``(scale * tanh(relu(value) / scale)) ** 2`` for a non-negative scale.

    The generic formula with the operands baked in as ``constexpr``: ``NEG2``
    is ``-2 / scale``, so the division becomes a multiply, and the exponent
    underflows to 0 for large magnitudes. The caller passes ``|clamp_scale|``.
    """
    e = tl.exp(tl.maximum(value, 0.0) * NEG2)
    c = SCALE * ((1.0 - e) / (1.0 + e))
    return c * c


@libentry()
@triton.jit
def _weighted_forward_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    hidden,
    SCALE: tl.constexpr,
    NEG2: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    EVEN_T: tl.constexpr,
    EVEN_H: tl.constexpr,
    WALK: tl.constexpr,
    NT: tl.constexpr,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``y = squared_relu(x) * weight[row]``, optionally tanh soft-clamped.

    One program owns an ``R`` x ``BLOCK`` tile that never spans a row boundary,
    so the weights are one short vector load broadcast across the tile. With
    ``WALK`` it loops over its rows' column blocks (1-D grid); otherwise each
    program handles one block (2-D grid). Only one branch is compiled.
    """
    rows = tle.program_id(0) * R + tl.arange(0, R)
    row_off = rows[:, None] * hidden
    if EVEN_T:
        weight = tl.load(w_ptr + rows).to(tl.float32)
    else:
        weight = tl.load(w_ptr + rows, mask=rows < NT, other=0.0).to(tl.float32)
    wv = weight[:, None]

    if WALK:
        n_full = hidden // BLOCK
        for c in range(0, n_full):
            offs = row_off + c * BLOCK + tl.arange(0, BLOCK)[None, :]
            if EVEN_T:
                x = tl.load(x_ptr + offs).to(tl.float32)
            else:
                x = tl.load(x_ptr + offs, mask=rows[:, None] < NT, other=0.0).to(
                    tl.float32
                )
            if HAS_CLAMP:
                act = _clamped_square(x, SCALE, NEG2)
            else:
                r = tl.maximum(x, 0.0)
                act = r * r
            out = (act * wv).to(y_ptr.dtype.element_ty)
            if EVEN_T:
                tl.store(y_ptr + offs, out)
            else:
                tl.store(y_ptr + offs, out, mask=rows[:, None] < NT)
        if not EVEN_H:
            columns = n_full * BLOCK + tl.arange(0, BLOCK)
            offs = row_off + columns[None, :]
            mask = columns[None, :] < hidden
            if not EVEN_T:
                mask = mask & (rows[:, None] < NT)
            x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            if HAS_CLAMP:
                act = _clamped_square(x, SCALE, NEG2)
            else:
                r = tl.maximum(x, 0.0)
                act = r * r
            tl.store(y_ptr + offs, (act * wv).to(y_ptr.dtype.element_ty), mask=mask)
    else:
        columns = tle.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        offs = row_off + columns[None, :]
        if EVEN_T and EVEN_H:
            x = tl.load(x_ptr + offs).to(tl.float32)
        else:
            mask = (rows[:, None] < NT) & (columns[None, :] < hidden)
            x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_CLAMP:
            act = _clamped_square(x, SCALE, NEG2)
        else:
            r = tl.maximum(x, 0.0)
            act = r * r
        out = (act * wv).to(y_ptr.dtype.element_ty)
        if EVEN_T and EVEN_H:
            tl.store(y_ptr + offs, out)
        else:
            tl.store(y_ptr + offs, out, mask=mask)


@libentry()
@triton.jit
def _clamped_forward_kernel(
    x_ptr,
    y_ptr,
    n,
    SCALE: tl.constexpr,
    NEG2: tl.constexpr,
    EVEN: tl.constexpr,
    WALK: tl.constexpr,
    BLOCK: tl.constexpr,
    STRIDE: tl.constexpr,
):
    """``y = (clamp_scale * tanh(relu(x) / clamp_scale)) ** 2`` without weights.

    A flat grid of ``BLOCK``-lane blocks. With ``WALK`` each program strides
    through the input (grid pinned to ``_FLAT_WALK_PROGRAMS``, ``STRIDE`` is
    ``grid * BLOCK``), and the runtime ``n`` keeps the loop un-unrolled.
    """
    if WALK:
        for start in range(tle.program_id(0) * BLOCK, n, STRIDE):
            offs = start + tl.arange(0, BLOCK)
            if EVEN:
                x = tl.load(x_ptr + offs).to(tl.float32)
            else:
                mask = offs < n
                x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            act = _clamped_square(x, SCALE, NEG2)
            if EVEN:
                tl.store(y_ptr + offs, act.to(y_ptr.dtype.element_ty))
            else:
                tl.store(y_ptr + offs, act.to(y_ptr.dtype.element_ty), mask=mask)
    else:
        offs = tle.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(x_ptr + offs).to(tl.float32)
        else:
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        act = _clamped_square(x, SCALE, NEG2)
        if EVEN:
            tl.store(y_ptr + offs, act.to(y_ptr.dtype.element_ty))
        else:
            tl.store(y_ptr + offs, act.to(y_ptr.dtype.element_ty), mask=mask)


def _clamp_operands(clamp_scale):
    """``(|scale|, -2 / |scale|)`` for the folded tanh soft clamp."""
    abs_scale = abs(clamp_scale)
    # scale == 0 behaves like the original division: exp(-inf) is 0, so
    # tanh -> 1 and the result is 0 for relu(x) > 0 (nan at relu(x) == 0).
    neg_two_over_scale = float("-inf") if abs_scale == 0.0 else -2.0 / abs_scale
    return abs_scale, neg_two_over_scale


def _tile_shape(hidden):
    """``(rows, columns)`` per program for the plain 2-D grid.

    ``_TILE_LANES`` lanes split across the row, with the column tile a power of
    two in ``_TILE_COLS_MIN``..``_TILE_COLS_MAX``; a narrow tile leaves a masked
    tail, which is correct for any width.
    """
    columns = min(
        max(triton.next_power_of_2(max(hidden, 1)), _TILE_COLS_MIN), _TILE_COLS_MAX
    )
    return max(1, _TILE_LANES // columns), columns


def _walk_tile_shape(hidden):
    """``(rows, columns)`` per program for the walking grid.

    About eight column blocks per row (the largest power of two at most
    ``hidden / 8``, clamped to ``_TILE_COLS_MIN``..``_TILE_COLS_MAX``), which
    measured fastest for every width tried.
    """
    columns = _TILE_COLS_MIN
    while columns * 16 <= hidden and columns < _TILE_COLS_MAX:
        columns *= 2
    return max(1, _TILE_LANES // columns), columns


def _needs_generic(input, weights, clamp_scale):
    """Whether the Ascend row-weight kernel cannot express this call."""
    if input.dtype not in _WEIGHTED_SQUARED_RELU_DTYPES:
        return True
    if input.device.type != runtime_device.name:
        return True
    if input.numel() == 0:
        # The generic path owns Megatron's empty/3-D epilogue edge cases.
        return True

    hidden = input.shape[-1]
    num_tokens = math.prod(input.shape[:-1])
    if weights is not None:
        if not isinstance(weights, torch.Tensor):
            return True
        if weights.device != input.device or weights.is_complex():
            return True
        if weights.dtype not in _WEIGHTED_SQUARED_RELU_DTYPES:
            return True
        try:
            broadcast_shape = torch.broadcast_shapes(
                tuple(weights.shape), (num_tokens, hidden)
            )
        except RuntimeError:
            return True
        if broadcast_shape != (num_tokens, hidden):
            return True
        if torch.broadcast_to(weights, (num_tokens, hidden)).stride(1) != 0:
            return True

    if clamp_scale is not None:
        if isinstance(clamp_scale, torch.Tensor):
            if clamp_scale.is_complex() or clamp_scale.numel() != 1:
                return True
            if clamp_scale.device != input.device and clamp_scale.dim() != 0:
                return True
        elif not isinstance(clamp_scale, numbers.Real):
            return True
    return False


def _generic_forward(input, weights, clamp_scale):
    """Use the generic Triton implementation, never a torch computation."""
    from flag_train.megatron.weighted_squared_relu_impl import (
        weighted_squared_relu_impl as generic_impl,
    )

    return generic_impl(input, weights, clamp_scale)


def _launch_weighted(x, weights, out, hidden, num_tokens, clamp_scale):
    rows, columns = _tile_shape(hidden)
    walk_rows, walk_columns = _walk_tile_shape(hidden)
    walk = triton.cdiv(num_tokens, walk_rows) >= _WALK_MIN_PROGRAMS
    if walk:
        rows, columns = walk_rows, walk_columns
        grid = (triton.cdiv(num_tokens, rows),)
    else:
        grid = (triton.cdiv(num_tokens, rows), triton.cdiv(hidden, columns))
    if clamp_scale is None:
        abs_scale, neg_two_over_scale = 0.0, 0.0
    else:
        abs_scale, neg_two_over_scale = _clamp_operands(clamp_scale)
    # An 8192-lane fp32 tile needs single buffering (see module docstring).
    stages = {"num_stages": 1} if walk and x.dtype == torch.float32 else {}
    with torch_device_fn.device(x.device):
        _weighted_forward_kernel[grid](
            x,
            weights,
            out,
            hidden,
            SCALE=abs_scale,
            NEG2=neg_two_over_scale,
            HAS_CLAMP=clamp_scale is not None,
            EVEN_T=num_tokens % rows == 0,
            EVEN_H=hidden % columns == 0,
            WALK=walk,
            NT=num_tokens,
            R=rows,
            BLOCK=columns,
            **stages,
        )


def _launch_clamped(x, out, n, clamp_scale):
    abs_scale, neg_two_over_scale = _clamp_operands(clamp_scale)
    block = (
        _FLAT_BLOCK
        if n >= _FLAT_BLOCK
        else max(triton.next_power_of_2(n), _FLAT_BLOCK_MIN)
    )
    blocks = triton.cdiv(n, block)
    walk = blocks > _FLAT_WALK_PROGRAMS
    grid = (_FLAT_WALK_PROGRAMS,)
    if not walk:
        grid = (blocks,)
    stages = {} if x.dtype != torch.float32 else {"num_stages": 1}
    with torch_device_fn.device(x.device):
        _clamped_forward_kernel[grid](
            x,
            out,
            n,
            SCALE=abs_scale,
            NEG2=neg_two_over_scale,
            EVEN=n % block == 0,
            WALK=walk,
            BLOCK=block,
            STRIDE=grid[0] * block,
            **stages,
        )


def weighted_squared_relu_impl(input, weights=None, clamp_scale=None):
    """Squared-ReLU with optional per-token weights and optional tanh soft clamp.

    Ascend implementation of ``megatron.core.fusions.fused_weighted_squared_relu
    .weighted_squared_relu_impl``; the contract is the generic operator's and the
    activation formula is unchanged. See the module docstring for what differs.
    """
    logger.debug("TRAIN_ASCEND WEIGHTED_SQUARED_RELU_IMPL")
    assert weights is not None or clamp_scale is not None, (
        "weighted_squared_relu_impl needs weights and/or clamp_scale; "
        "use squared_relu directly otherwise."
    )
    assert input.dim() in (
        2,
        3,
    ), f"weighted_squared_relu_impl expects a 2-D or 3-D input, got {input.dim()}-D"
    weights = _widen_unsigned_weights(weights, input)
    if input.device.type == "cpu":
        # Megatron accepts CPU tensors: run on the accelerator, return on CPU.
        device = torch.device(runtime_device.name)
        if isinstance(weights, torch.Tensor):
            weights = weights.to(device)
        if isinstance(clamp_scale, torch.Tensor):
            clamp_scale = clamp_scale.to(device)
        out = weighted_squared_relu_impl(input.to(device), weights, clamp_scale)
        return out.to(input.device)
    if _needs_generic(input, weights, clamp_scale):
        return _generic_forward(input, weights, clamp_scale)
    _check_input(input, "weighted_squared_relu_impl")

    hidden = input.shape[-1]
    num_tokens = math.prod(input.shape[:-1])

    if weights is not None:
        _check_weights(weights)
        if weights.device != input.device:
            raise ValueError(
                "weighted_squared_relu_impl weights must be on the same device as the "
                f"input, got {weights.device} and {input.device}"
            )
        # Megatron broadcasts the weights against the flattened activation.
        # The Ascend kernel below is specialised for one weight per row; other
        # broadcast forms were already delegated to the generic Triton path.
        try:
            broadcast_shape = torch.broadcast_shapes(
                tuple(weights.shape), (num_tokens, hidden)
            )
        except RuntimeError as exc:
            raise ValueError(
                "weighted_squared_relu_impl weights must be broadcastable to "
                f"{(num_tokens, hidden)} (the flattened activation shape), got "
                f"{tuple(weights.shape)}"
            ) from exc
        if broadcast_shape != (num_tokens, hidden):
            raise ValueError(
                "weighted_squared_relu_impl weights must broadcast to exactly "
                f"{(num_tokens, hidden)} (the flattened activation shape), got "
                f"{tuple(weights.shape)} -> {tuple(broadcast_shape)}"
            )
        weights = torch.broadcast_to(weights, (num_tokens, hidden))

    if input.numel() == 0:
        # No launch when there is nothing to activate.
        return torch.empty_like(input)

    if clamp_scale is not None:
        clamp_scale = float(clamp_scale)

    # Match Megatron's ``input.view(-1, hidden)``: layouts it cannot flatten
    # fail here too. A viewable but non-contiguous layout is copied for the
    # kernel, which is a layout change rather than a different computation.
    flat = input.view(-1, hidden)
    if not flat.is_contiguous():
        flat = flat.contiguous()

    out = torch.empty_like(flat)
    if weights is None:
        _launch_clamped(flat, out, flat.numel(), clamp_scale)
    else:
        # A broadcast dimension has stride 0; materialise the per-row scalar.
        _launch_weighted(
            flat,
            weights[:, 0].contiguous(),
            out,
            hidden,
            num_tokens,
            clamp_scale,
        )
    return out.view(input.shape)
