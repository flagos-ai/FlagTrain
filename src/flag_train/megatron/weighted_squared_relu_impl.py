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

import torch
import torch.nn.functional as F
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


def _scalar_clamp(clamp_scale):
    """Return ``clamp_scale`` as a float when the Triton path can use it."""
    if isinstance(clamp_scale, (str, bytes)):
        return None
    if isinstance(clamp_scale, torch.Tensor):
        if clamp_scale.numel() != 1:
            return None
        try:
            return float(clamp_scale.item())
        except (TypeError, ValueError):
            return None
    try:
        return float(clamp_scale)
    except (TypeError, ValueError):
        return None


def _fast_weights(weights, shape, input):
    """A broadcast view of ``weights`` when the Triton kernel can read it."""
    if weights is None:
        return None
    if not isinstance(weights, torch.Tensor):
        return None
    if weights.device != input.device:
        return None
    if weights.is_complex() or weights.dtype not in _WEIGHTED_SQUARED_RELU_DTYPES:
        return None
    try:
        broadcast_shape = torch.broadcast_shapes(tuple(weights.shape), shape)
    except RuntimeError:
        return None
    if broadcast_shape != shape:
        return None
    try:
        return torch.broadcast_to(weights, shape)
    except (RuntimeError, NotImplementedError):
        return None


def _torch_relu(x):
    """ReLU for the fallback, including Megatron's compiled bool behaviour."""
    # Eager torch.relu rejects bool, but Megatron's @jit_fuser path accepts it
    # (bool values are already non-negative, so ReLU is the identity).
    return x if x.dtype == torch.bool else F.relu(x)


def _torch_forward(flat, weights, clamp_scale):
    """Megatron's forward composition, for inputs the Triton path cannot take."""
    if weights is None:
        t = torch.tanh(F.relu(flat.float()) / clamp_scale)
        return torch.pow(clamp_scale * t, 2).to(flat.dtype)
    if clamp_scale is None:
        return (torch.pow(_torch_relu(flat), 2) * weights).to(flat.dtype)
    t = torch.tanh(F.relu(flat.float()) / clamp_scale)
    return (torch.pow(clamp_scale * t, 2) * weights).to(flat.dtype)


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
    Contiguous float tensors on the active device take the Triton path; every
    other valid input (CPU, integer/bool dtypes, complex weights, broadcast
    weights that add leading dimensions, non-contiguous views) uses the torch
    composition so the externally visible behaviour matches Megatron.

    Args:
        input (Tensor): 2-D or 3-D tensor.  The Triton path handles
            fp16/bf16/fp32/fp64 on the active device; other dtypes and CPU
            inputs use the torch fallback.
        weights (Tensor, optional): any tensor broadcastable to the flattened
            activation, with Megatron's ordinary broadcasting rules.
        clamp_scale (float, optional): the tanh soft-clamp scale.

    Returns:
        Tensor: the forward result, with the same shape/dtype rules as Megatron.

    Raises:
        AssertionError: when both ``weights`` and ``clamp_scale`` are ``None``,
            or when the input is not 2-D/3-D.
        RuntimeError: when Megatron's ``view``/broadcast operations reject the
            inputs; those errors are left to the corresponding torch call.
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

    # Match Megatron's `input.view(-1, hidden)`: layouts that cannot be viewed
    # raise the same error before either implementation does any arithmetic.
    flat = input.view(-1, ori_shape[-1])

    clamp_value = None if clamp_scale is None else _scalar_clamp(clamp_scale)
    if isinstance(clamp_scale, torch.Tensor) and clamp_scale.device != input.device:
        # Let the torch fallback reproduce Megatron's device rules for a
        # tensor-valued clamp scale instead of silently treating it as a host
        # float on the Triton path.
        clamp_value = None
    fast_weights = _fast_weights(weights, flat.shape, input)
    use_triton = (
        flat.is_contiguous()
        and flat.dtype in _WEIGHTED_SQUARED_RELU_DTYPES
        and flat.device.type == runtime_device.name
        and (weights is None or fast_weights is not None)
        and (clamp_scale is None or clamp_value is not None)
    )

    if not use_triton:
        output = _torch_forward(flat, weights, clamp_scale)
        if len(ori_shape) == 2:
            return output
        return output.view(ori_shape[0], ori_shape[1], -1)

    num_tokens, hidden = flat.shape
    if flat.numel() == 0:
        out = torch.empty_like(flat)
        if len(ori_shape) == 2:
            return out
        # Megatron still reaches its 3-D epilogue reshape, which fails when
        # the flattened result has zero elements.
        return out.view(ori_shape[0], ori_shape[1], -1)

    out = torch.empty_like(flat)
    if weights is None:
        _launch_clamped(flat, out, flat.numel(), clamp_value)
    else:
        _launch_weighted(
            flat,
            fast_weights,
            out,
            hidden,
            num_tokens,
            fast_weights.stride(0),
            fast_weights.stride(1),
            clamp_value,
        )
    return out.view(ori_shape)
