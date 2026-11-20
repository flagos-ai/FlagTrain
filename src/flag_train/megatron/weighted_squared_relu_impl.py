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

_NUM_WARPS = 4

# 16-bit inputs move half the bytes per element, so H20 needs a wider block.
_BLOCK_SIZE = 1024
_BLOCK_SIZE_16BIT = 2048

_WEIGHTED_SQUARED_RELU_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)

# Megatron leaves the arithmetic to torch, which also accepts integer
# activations; Triton can load and square those. Bool is deliberately absent:
# Megatron's `relu` rejects it.
_INT_INPUT_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)

# Weight dtypes the Triton kernels can load. Complex weights are rejected:
# Megatron's arithmetic would keep a complex product until the final cast, but
# this operator only supports real-valued weights.
_WEIGHT_DTYPES = (
    _WEIGHTED_SQUARED_RELU_DTYPES
    + _INT_INPUT_DTYPES
    + (
        torch.bool,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    )
)

_CLAMP_VECTOR_DTYPES = _WEIGHT_DTYPES


def _fast_weights(weights, shape):
    """A broadcast view of ``weights`` when the fast kernel can read it."""
    if weights is None:
        return None
    try:
        broadcast_shape = tuple(torch.broadcast_shapes(tuple(weights.shape), shape))
    except RuntimeError:
        return None
    if broadcast_shape != tuple(shape):
        return None
    try:
        return torch.broadcast_to(weights, shape)
    except (RuntimeError, NotImplementedError):
        return None


def _normalize_weights(weights, input):
    """Validate/normalize weights; raise where Megatron's torch multiply does."""
    if weights is None:
        return None
    if isinstance(weights, torch.Tensor):
        if weights.is_complex():
            raise ValueError(
                "weighted_squared_relu_impl weights must be real-valued, "
                f"got {weights.dtype}"
            )
        if weights.device != input.device:
            if weights.dim() != 0 or weights.device.type != "cpu":
                raise RuntimeError(
                    "Expected all tensors to be on the same device, but found at "
                    f"least two devices, {input.device} and {weights.device}!"
                )
            # torch allows a 0-dim CPU scalar next to a device tensor, but not
            # a device scalar next to a CPU tensor.
            weights = weights.to(input.device)
        if weights.dtype not in _WEIGHT_DTYPES:
            raise ValueError(
                "weighted_squared_relu_impl weights needs a numeric tensor, "
                f"got {weights.dtype}"
            )
        if weights.dtype == torch.bool:
            weights = weights.to(torch.float32)
        return weights
    if isinstance(weights, numbers.Complex) and not isinstance(weights, numbers.Real):
        raise ValueError(
            "weighted_squared_relu_impl weights must be real-valued, got complex"
        )
    if isinstance(weights, numbers.Number):
        weights = torch.as_tensor(weights, device=input.device)
        if weights.is_complex():
            raise ValueError(
                "weighted_squared_relu_impl weights must be real-valued, "
                f"got {weights.dtype}"
            )
        if weights.dtype not in _WEIGHT_DTYPES:
            raise ValueError(
                "weighted_squared_relu_impl weights needs a numeric tensor, "
                f"got {weights.dtype}"
            )
        if weights.dtype == torch.bool:
            weights = weights.to(torch.float32)
        return weights
    raise TypeError(
        "weighted_squared_relu_impl weights needs a tensor or a number, "
        f"got {type(weights).__name__}"
    )


def _normalize_clamp(clamp_scale, input):
    """Return ``(scalar, vector)`` for Megatron's scalar/vector clamp forms."""
    if clamp_scale is None:
        return None, None
    if isinstance(clamp_scale, torch.Tensor):
        if clamp_scale.is_complex():
            raise ValueError(
                "weighted_squared_relu_impl clamp_scale must be real-valued, "
                f"got {clamp_scale.dtype}"
            )
        if clamp_scale.numel() == 1:
            if clamp_scale.device != input.device and not (
                clamp_scale.dim() == 0
                and clamp_scale.device.type == "cpu"
                and input.device.type != "cpu"
            ):
                raise RuntimeError(
                    "Expected all tensors to be on the same device, but found at "
                    f"least two devices, {input.device} and {clamp_scale.device}!"
                )
            return clamp_scale.item(), None
        if clamp_scale.device != input.device:
            raise RuntimeError(
                "Expected all tensors to be on the same device, but found at "
                f"least two devices, {input.device} and {clamp_scale.device}!"
            )
        if clamp_scale.dtype not in _CLAMP_VECTOR_DTYPES:
            raise ValueError(
                "weighted_squared_relu_impl clamp_scale needs a numeric tensor, "
                f"got {clamp_scale.dtype}"
            )
        return None, clamp_scale.to(torch.float32)
    if isinstance(clamp_scale, numbers.Real):
        return float(clamp_scale), None
    if isinstance(clamp_scale, numbers.Complex):
        raise ValueError(
            "weighted_squared_relu_impl clamp_scale must be real-valued, got complex"
        )
    raise TypeError(
        "weighted_squared_relu_impl clamp_scale needs a number or tensor, "
        f"got {type(clamp_scale).__name__}"
    )


def _validate_input(input):
    if input.dtype not in _WEIGHTED_SQUARED_RELU_DTYPES + _INT_INPUT_DTYPES:
        raise ValueError(
            "weighted_squared_relu_impl input needs a float or integer tensor, "
            f"got {input.dtype}"
        )
    if input.device.type not in ("cpu", runtime_device.name):
        raise ValueError(
            f"weighted_squared_relu_impl needs a CPU or {runtime_device.name} "
            f"tensor, got {input.device.type}"
        )


def _accelerator_device():
    """Move CPU inputs here so the computation still runs in Triton."""
    name = runtime_device.name
    if name == "cpu":
        raise RuntimeError(
            "weighted_squared_relu_impl cannot run CPU inputs without an "
            "accelerator: no device Triton backend is available"
        )
    is_available = getattr(torch_device_fn, "is_available", None)
    if callable(is_available) and not is_available():
        raise RuntimeError(
            f"weighted_squared_relu_impl cannot run CPU inputs: {name} is unavailable"
        )
    return name


@triton.jit
def _clamped_square(value, scale, neg_two_over_scale):
    """``(scale * tanh(relu(value) / scale)) ** 2`` for a non-negative scale.

    Folding ``-2 / scale`` into the exponent replaces the per-element division
    with a multiply, and ``exp(-2 * relu / scale)`` underflows to 0, so large
    magnitudes yield exactly ``scale ** 2``. The caller passes ``|clamp_scale|``,
    which leaves the squared result unchanged for a negative scale.
    """
    e = tl.exp(tl.maximum(value, 0.0) * neg_two_over_scale)
    c = scale * ((1.0 - e) / (1.0 + e))
    return c * c


@libentry()
@triton.jit
def _weighted_forward_kernel(
    x_ptr,
    weight_ptr,
    y_ptr,
    hidden,
    weight_stride_row,
    weight_stride_col,
    abs_scale,
    neg_two_over_scale,
    HAS_CLAMP: tl.constexpr,
    EVEN_HIDDEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """``y = squared_relu(x) * weight[row, col]``, optionally clamped.

    One program handles a ``BLOCK_SIZE`` slice of one row.  ``weight`` is a
    broadcast view of shape ``(num_tokens, hidden)``: a broadcast dimension has
    stride 0, so the same load covers per-token, per-column and scalar weights.
    The activation is computed in fp32, matching the reference; the bounds check
    is dropped when ``hidden`` is block-aligned.
    """
    row = tle.program_id(0)
    base = row * hidden
    columns = tle.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if EVEN_HIDDEN:
        x = tl.load(x_ptr + base + columns).to(tl.float32)
        weight = tl.load(
            weight_ptr + row * weight_stride_row + columns * weight_stride_col
        ).to(tl.float32)
    else:
        mask = columns < hidden
        x = tl.load(x_ptr + base + columns, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(
            weight_ptr + row * weight_stride_row + columns * weight_stride_col,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

    if HAS_CLAMP:
        act = _clamped_square(x, abs_scale, neg_two_over_scale)
    else:
        r = tl.maximum(x, 0.0)
        act = r * r

    if EVEN_HIDDEN:
        tl.store(y_ptr + base + columns, (act * weight).to(y_ptr.dtype.element_ty))
    else:
        tl.store(
            y_ptr + base + columns,
            (act * weight).to(y_ptr.dtype.element_ty),
            mask=mask,
        )


@libentry()
@triton.jit
def _clamped_forward_kernel(
    x_ptr,
    y_ptr,
    n,
    abs_scale,
    neg_two_over_scale,
    EVEN_N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """``y = (clamp_scale * tanh(relu(x) / clamp_scale)) ** 2`` without weights."""
    offsets = tle.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if EVEN_N:
        x = tl.load(x_ptr + offsets).to(tl.float32)
    else:
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    act = _clamped_square(x, abs_scale, neg_two_over_scale)

    if EVEN_N:
        tl.store(y_ptr + offsets, act.to(y_ptr.dtype.element_ty))
    else:
        tl.store(y_ptr + offsets, act.to(y_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def _general_forward_kernel(
    x_ptr,
    weight_ptr,
    clamp_ptr,
    y_ptr,
    in_rows,
    in_hidden,
    out_hidden,
    abs_scale,
    neg_two_over_scale,
    HAS_WEIGHT: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    CLAMP_VECTOR: tl.constexpr,
    INT_INPUT: tl.constexpr,
    UINT8_OUTPUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One output row/column tile for the broadcast and integer paths.

    The output shape is ``broadcast(flat_input, weights, clamp)``. Input rows
    and columns therefore repeat by modulo; weights and a vector clamp are
    materialised to the output shape before launch.
    """
    out_row = tle.program_id(0)
    columns = tle.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = columns < out_hidden

    in_row = out_row % in_rows
    in_columns = columns % in_hidden
    x = tl.load(x_ptr + in_row * in_hidden + in_columns, mask=mask, other=0)

    if HAS_CLAMP:
        xf = x.to(tl.float32)
        if CLAMP_VECTOR:
            c = tl.load(
                clamp_ptr + out_row * out_hidden + columns, mask=mask, other=0.0
            ).to(tl.float32)
            abs_c = tl.abs(c)
            neg2 = tl.where(abs_c == 0.0, float("-inf"), -2.0 / abs_c)
            e = tl.exp(tl.maximum(xf, 0.0) * neg2)
            c = abs_c * ((1.0 - e) / (1.0 + e))
            act = c * c
        else:
            act = _clamped_square(xf, abs_scale, neg_two_over_scale)
    elif INT_INPUT:
        r = tl.maximum(x, 0)
        act = (r * r).to(tl.float32)
    else:
        xf = x.to(tl.float32)
        r = tl.maximum(xf, 0.0)
        act = r * r

    if HAS_WEIGHT:
        weight = tl.load(
            weight_ptr + out_row * out_hidden + columns, mask=mask, other=0.0
        ).to(tl.float32)
        out = act * weight
    else:
        out = act

    if UINT8_OUTPUT:
        # torch's float -> uint8 cast truncates toward zero and wraps modulo
        # 256; Triton's direct cast saturates negatives to zero.
        out = (out.to(tl.int32) & 255).to(tl.uint8)
    else:
        out = out.to(y_ptr.dtype.element_ty)
    tl.store(
        y_ptr + out_row * out_hidden + columns,
        out,
        mask=mask,
    )


def _launch_general(
    x,
    weight,
    clamp,
    out,
    in_rows,
    in_hidden,
    out_hidden,
    clamp_scalar,
):
    block_size, num_warps = _launch_config(x.dtype)
    if clamp_scalar is None:
        abs_scale, neg_two_over_scale = 0.0, 0.0
    else:
        abs_scale, neg_two_over_scale = _clamp_operands(clamp_scalar)
    grid = (out.numel() // out_hidden, triton.cdiv(out_hidden, block_size))
    with torch_device_fn.device(x.device):
        _general_forward_kernel[grid](
            x,
            x if weight is None else weight,
            x if clamp is None else clamp,
            out,
            in_rows,
            in_hidden,
            out_hidden,
            abs_scale,
            neg_two_over_scale,
            HAS_WEIGHT=weight is not None,
            HAS_CLAMP=clamp_scalar is not None or clamp is not None,
            CLAMP_VECTOR=clamp is not None,
            INT_INPUT=x.dtype in _INT_INPUT_DTYPES,
            UINT8_OUTPUT=out.dtype == torch.uint8,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )


def _finish_output(out, ori_shape):
    if len(ori_shape) == 2:
        return out
    return out.view(ori_shape[0], ori_shape[1], -1)


def _launch_config(dtype):
    """Block/warp pair measured fastest for ``dtype`` on this device."""
    if runtime_device.vendor_name == "nvidia" and dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        return _BLOCK_SIZE_16BIT, _NUM_WARPS
    return _BLOCK_SIZE, _NUM_WARPS


def _clamp_operands(clamp_scale):
    """``(|scale|, -2 / |scale|)`` for the folded tanh soft clamp."""
    abs_scale = abs(clamp_scale)
    # scale == 0 behaves like the original division: exp(-inf) is 0, so
    # tanh -> 1 and the result is 0 for relu(x) > 0 (nan at relu(x) == 0).
    neg_two_over_scale = float("-inf") if abs_scale == 0.0 else -2.0 / abs_scale
    return abs_scale, neg_two_over_scale


def _launch_weighted(
    x,
    weights,
    out,
    hidden,
    num_tokens,
    weight_stride_row,
    weight_stride_col,
    clamp_scale,
):
    block_size, num_warps = _launch_config(x.dtype)
    if clamp_scale is None:
        abs_scale, neg_two_over_scale = 0.0, 0.0
    else:
        abs_scale, neg_two_over_scale = _clamp_operands(clamp_scale)
    grid = (num_tokens, triton.cdiv(hidden, block_size))
    with torch_device_fn.device(x.device):
        _weighted_forward_kernel[grid](
            x,
            weights,
            out,
            hidden,
            weight_stride_row,
            weight_stride_col,
            abs_scale,
            neg_two_over_scale,
            HAS_CLAMP=clamp_scale is not None,
            EVEN_HIDDEN=hidden % block_size == 0,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )


def _launch_clamped(x, out, n, clamp_scale):
    block_size, num_warps = _launch_config(x.dtype)
    abs_scale, neg_two_over_scale = _clamp_operands(clamp_scale)
    grid = (triton.cdiv(n, block_size),)
    with torch_device_fn.device(x.device):
        _clamped_forward_kernel[grid](
            x,
            out,
            n,
            abs_scale,
            neg_two_over_scale,
            EVEN_N=n % block_size == 0,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )


def weighted_squared_relu_impl(input, weights=None, clamp_scale=None):
    """Squared-ReLU with optional weights and optional tanh soft clamp.

    Forward-only port of ``megatron.core.fusions.fused_weighted_squared_relu
    .weighted_squared_relu_impl``.  The public contract is Megatron's: the input
    is viewed as ``(num_tokens, hidden)`` with ``view`` (so layouts Megatron
    cannot flatten fail here too), ``weights`` is ordinary torch broadcasting,
    and the 3-D epilogue reshapes the result back to ``(S, B, hidden)``.
    Float and integer tensors run Triton; broadcast weights (including extra
    leading dimensions), vector clamps and non-contiguous inputs use the general
    Triton kernel. CPU inputs are staged to the active accelerator, computed in
    Triton, and copied back to CPU. Bool inputs, complex weights and complex
    clamp scales are rejected rather than emulated with torch.

    Args:
        input (Tensor): 2-D or 3-D tensor.  Supported dtypes are fp16, bf16,
            fp32, fp64 and the integer types torch's ``relu`` accepts.
        weights (Tensor or number, optional): real-valued tensor or number
            broadcastable to the flattened activation, with Megatron's ordinary
            broadcasting rules.
        clamp_scale (float or Tensor, optional): real scalar or broadcastable
            tanh soft-clamp scale.

    Returns:
        Tensor: the forward result, with the same shape/dtype rules as Megatron.

    Raises:
        AssertionError: when both ``weights`` and ``clamp_scale`` are ``None``,
            or when the input is not 2-D/3-D.
        RuntimeError: when Megatron's ``view``/broadcast operations reject the
            inputs, or when a CPU input has no accelerator to stage it through.
        ValueError: for bool inputs, complex weights or clamp scales, or other
            dtypes Triton cannot execute.
    """
    logger.debug("TRAIN WEIGHTED_SQUARED_RELU_IMPL")
    assert weights is not None or clamp_scale is not None, (
        "weighted_squared_relu_impl needs weights and/or clamp_scale; "
        "use squared_relu directly otherwise."
    )
    ori_shape = input.shape
    assert len(ori_shape) in (
        2,
        3,
    ), f"weighted_squared_relu_impl expects a 2-D or 3-D input, got {input.dim()}-D"
    _validate_input(input)

    # Match Megatron's `input.view(-1, hidden)`: layouts that cannot be viewed
    # raise the same error before either implementation does any arithmetic.
    flat = input.view(-1, ori_shape[-1])
    weight_tensor = _normalize_weights(weights, input)
    clamp_scalar, clamp_vector = _normalize_clamp(clamp_scale, input)

    out_shape = tuple(
        torch.broadcast_shapes(
            tuple(flat.shape),
            () if weight_tensor is None else tuple(weight_tensor.shape),
            () if clamp_vector is None else tuple(clamp_vector.shape),
        )
    )
    if flat.numel() == 0 or math.prod(out_shape) == 0:
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
        return _finish_output(out, ori_shape)

    compute_device = input.device
    if input.device.type == "cpu":
        # Megatron accepts CPU tensors. Keep its device rules during
        # normalization, then run the arithmetic in Triton on the accelerator
        # and copy the result back.
        compute_device = _accelerator_device()
        flat = flat.contiguous().to(compute_device)
        if weight_tensor is not None:
            weight_tensor = weight_tensor.to(compute_device)
        if clamp_vector is not None:
            clamp_vector = clamp_vector.to(compute_device)

    flat_kernel = flat if flat.is_contiguous() else flat.contiguous()
    fast_weights = (
        _fast_weights(weight_tensor, flat.shape) if weight_tensor is not None else None
    )
    use_fast = (
        flat_kernel.dtype in _WEIGHTED_SQUARED_RELU_DTYPES
        and clamp_vector is None
        and tuple(out_shape) == tuple(flat.shape)
        and (weight_tensor is None or fast_weights is not None)
    )

    if use_fast:
        out = torch.empty_like(flat_kernel)
        if weight_tensor is None:
            _launch_clamped(flat_kernel, out, flat_kernel.numel(), clamp_scalar)
        else:
            _launch_weighted(
                flat_kernel,
                fast_weights,
                out,
                flat.shape[1],
                flat.shape[0],
                fast_weights.stride(0),
                fast_weights.stride(1),
                clamp_scalar,
            )
        out = _finish_output(out, ori_shape)
        return out.to(input.device) if input.device.type == "cpu" else out

    out = torch.empty(out_shape, dtype=input.dtype, device=compute_device)
    out_hidden = out_shape[-1]
    weight = (
        None
        if weight_tensor is None
        else torch.broadcast_to(weight_tensor, out_shape).contiguous()
    )
    clamp = (
        None
        if clamp_vector is None
        else torch.broadcast_to(clamp_vector, out_shape).to(torch.float32).contiguous()
    )
    _launch_general(
        flat_kernel,
        weight,
        clamp,
        out,
        flat.shape[0],
        flat.shape[1],
        out_hidden,
        clamp_scalar,
    )
    out = _finish_output(out, ori_shape)
    return out.to(input.device) if input.device.type == "cpu" else out
