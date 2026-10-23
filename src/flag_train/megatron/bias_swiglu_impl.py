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
"""Fused bias + SwiGLU, including the clamped, SiTU-GLU and weighted variants.

One token is one row. The last dimension is concatenated ``[gate | linear]``
and the output keeps only the gated half. Bias is added inside the kernel.
The bias gradient is the pre-activation gradient reduced to the bias shape,
which is what ``x + bias`` produces. Megatron-LM's
``BiasSwiGLUFunction.backward`` returns that gradient unreduced for the bias
slot and relies on autograd's implicit broadcast reduction; this
implementation reduces it in fp32 explicitly, so the result does not depend on
that implicit step.
"""

import logging
import math
import numbers
from functools import cache

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry, tl_extra_shim
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_NUM_WARPS = 4
_BLOCK_SIZE = 1024
_MIN_BLOCK_SIZE = 16
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Both helpers are provided by the platform layer. ``tl.sigmoid`` is a core
# Triton builtin; ``tl_extra_shim.tanh`` maps to libdevice and falls back to an
# exp-based form only on backends without it. Neither needs the extra argument
# clamp this file used to carry: exp overflow in the sigmoid denominator lands
# on inf, and ``1 / (1 + inf)`` is exactly 0 rather than NaN.
_sigmoid = tl.sigmoid
_tanh = tl_extra_shim.tanh

_MODE_PLAIN = 0
_MODE_CLAMPED = 1
_MODE_SITU = 2

_BIAS_NONE = 0
_BIAS_VECTOR = 1
_BIAS_FULL = 2

_WEIGHT_NONE = 0
_WEIGHT_SCALAR = 1
_WEIGHT_ROW = 2
_WEIGHT_ELEMENT = 3


def _check_input(tensor, name):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} needs a tensor, got {type(tensor).__name__}")
    if tensor.dtype not in _DTYPES:
        wanted = " / ".join(str(dtype) for dtype in _DTYPES)
        raise ValueError(f"{name} needs a {wanted} tensor, got {tensor.dtype}")
    if tensor.device.type == "cpu":
        raise ValueError(f"{name} needs a device tensor, not one on the CPU")
    if tensor.device.type != runtime_device.name:
        raise ValueError(
            f"{name} needs a {runtime_device.name} tensor, got {tensor.device.type}"
        )


def _check_same_device(tensor, name, device):
    if tensor.device != device:
        raise ValueError(f"{name} is on {tensor.device}, expected {device}")


def _positive_scale(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} needs a finite float, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero, got {value}")
    return value


def _resolve_variant(clamp_value, gate_clamp_scale, linear_clamp_scale):
    """Pick the activation and validate the mutually exclusive clamps.

    A non-positive ``clamp_value`` is rejected. Megatron-LM's kernel treats
    that as the unclamped SwiGLU; the config layer already forbids it, and
    silently ignoring it would hide a bad call.
    """
    if gate_clamp_scale is not None and clamp_value is not None:
        raise ValueError("gate_clamp_scale and clamp_value cannot both be set")
    if linear_clamp_scale is not None and gate_clamp_scale is None:
        raise ValueError("linear_clamp_scale requires gate_clamp_scale")

    has_linear_clamp = False
    if gate_clamp_scale is not None:
        gate_clamp_scale = _positive_scale(gate_clamp_scale, "gate_clamp_scale")
        if linear_clamp_scale is not None:
            linear_clamp_scale = _positive_scale(
                linear_clamp_scale, "linear_clamp_scale"
            )
            has_linear_clamp = True
        return _MODE_SITU, gate_clamp_scale, linear_clamp_scale, has_linear_clamp

    if clamp_value is not None:
        clamp_value = _positive_scale(clamp_value, "clamp_value")
        return _MODE_CLAMPED, clamp_value, None, False

    return _MODE_PLAIN, None, None, False


def _can_broadcast_to(tensor, shape):
    try:
        torch.broadcast_to(tensor, shape)
    except RuntimeError:
        return False
    return True


def _sum_to(tensor, shape):
    """Sum ``tensor`` down to ``shape``, the broadcast rule for a gradient."""
    shape = tuple(shape)
    if tuple(tensor.shape) == shape:
        return tensor
    if len(shape) == 0:
        return tensor.sum()
    leading = tensor.ndim - len(shape)
    if leading < 0:
        raise ValueError(f"cannot reduce shape {tuple(tensor.shape)} to {shape}")
    dims = list(range(leading))
    for index, size in enumerate(shape):
        src = tensor.shape[index + leading]
        if size == 1 and src != 1:
            dims.append(index + leading)
        elif size != src:
            raise ValueError(f"cannot reduce shape {tuple(tensor.shape)} to {shape}")
    if dims:
        tensor = tensor.sum(dim=dims, keepdim=True)
    return tensor.reshape(shape)


def _prepare_bias(bias, n_rows, hidden):
    """Return a kernel bias tensor and a mode.

    Mode vector is one row of ``hidden`` values shared by every token, which
    is the ``bias`` produced by ``skip_bias_add``. Mode full is a different
    bias per element and is only used when the tensor really depends on the
    token index.
    """
    if bias is None:
        return None, _BIAS_NONE
    _check_input(bias, "bias_swiglu bias")
    flat_shape = (n_rows, hidden)
    if not _can_broadcast_to(bias, flat_shape):
        raise ValueError(
            "bias_swiglu bias must broadcast with the flattened input "
            f"{flat_shape}, got {tuple(bias.shape)}"
        )
    if hidden == 0 or n_rows == 0:
        return bias, _BIAS_VECTOR
    expanded = torch.broadcast_to(bias, flat_shape)
    if expanded.stride(0) == 0:
        # Same values on every token. Copy one row, not the whole broadcast.
        return expanded[0].contiguous(), _BIAS_VECTOR
    return expanded.contiguous(), _BIAS_FULL


def _prepare_weight(weights, out_shape):
    """Return a compact weight tensor and how the kernel should index it.

    The mode is chosen from the **shape**, not from strides. A full-shape
    tensor that happens to have zero strides (``expand``) still owns one
    gradient per element in PyTorch, so it must take the elementwise path.
    Scalar and per-row weights stay one value per token when their shape says
    so. An elementwise weight is copied to ``[token, half]`` so the kernel can
    load it with the output.
    """
    if weights is None:
        return None, _WEIGHT_NONE
    _check_input(weights, "bias_swiglu weights")
    n_rows = 1
    for size in out_shape[:-1]:
        n_rows *= size
    n_cols = out_shape[-1]
    flat_shape = (n_rows, n_cols)
    if _can_broadcast_to(weights, flat_shape):
        expanded = torch.broadcast_to(weights, flat_shape)
    elif _can_broadcast_to(weights, out_shape):
        expanded = torch.broadcast_to(weights, out_shape).reshape(flat_shape)
    else:
        raise ValueError(
            "bias_swiglu weights must broadcast with the output "
            f"{tuple(out_shape)} (or the flattened {flat_shape}), "
            f"got {tuple(weights.shape)}"
        )
    if n_rows == 0 or n_cols == 0:
        return weights, _WEIGHT_ROW
    if weights.numel() == 1:
        # One value shared by every output element. ``reshape(1)`` rejects the
        # broadcast view because it still reports ``n_rows * n_cols``
        # elements, so keep the single value explicitly.
        return expanded.reshape(-1)[:1].contiguous(), _WEIGHT_SCALAR
    if tuple(weights.shape) in (flat_shape, tuple(out_shape)):
        # One weight per output element. This includes zero-stride views such
        # as ``scalar.expand(*out_shape)``: PyTorch stores a per-element
        # gradient for that leaf, so the kernel must not reduce it to a
        # per-token value.
        return expanded.contiguous(), _WEIGHT_ELEMENT
    if weights.shape[-1] == 1:
        # One weight per token: ``[token, 1]`` or the unflattened
        # ``[*batch, 1]`` used by MoE.
        return expanded[:, 0].contiguous(), _WEIGHT_ROW
    # A partial broadcast such as ``[half]`` or ``[1, half]``. Keep one value
    # per output element; the host reduction restores ``weights.shape``.
    return expanded.contiguous(), _WEIGHT_ELEMENT


def _weight_grad(reduced, weights, out_shape, summed_columns):
    """Map a kernel reduction back onto ``weights``.

    ``summed_columns`` means the half-dimension is already summed and
    ``reduced`` has a trailing 1 standing in for that dimension.
    """
    n_rows = 1
    for size in out_shape[:-1]:
        n_rows *= size
    n_cols = out_shape[-1]
    flat_shape = (n_rows, n_cols)
    if summed_columns:
        flat_reduced = (n_rows, 1)
        if _can_broadcast_to(weights, flat_reduced) or _can_broadcast_to(
            weights, flat_shape
        ):
            return _sum_to(reduced, weights.shape).to(weights.dtype)
        return _sum_to(reduced.reshape(*out_shape[:-1], 1), weights.shape).to(
            weights.dtype
        )
    if _can_broadcast_to(weights, flat_shape):
        return _sum_to(reduced, weights.shape).to(weights.dtype)
    return _sum_to(reduced.reshape(out_shape), weights.shape).to(weights.dtype)


@cache
def _probe_clamp_boundaries():
    """Whether ``clamp`` keeps a unit gradient on an exact boundary.

    PyTorch has shipped both conventions. The mask in the clamped backward
    follows whichever one this process uses, so the kernel matches
    ``Tensor.clamp`` rather than a hardcoded ``<`` or ``<=``.
    """
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    with unset_fake_temporarily(), torch.inference_mode(False), torch.enable_grad():
        gate = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
        linear = torch.tensor([-1.0, 1.0], dtype=torch.float32, requires_grad=True)
        gate_grad, linear_grad = torch.autograd.grad(
            gate.clamp(max=1.0) + linear.clamp(min=-1.0, max=1.0).sum(),
            (gate, linear),
        )
        gradients = [gate_grad.item(), *linear_grad.tolist()]
    if gradients not in ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]):
        raise RuntimeError(f"Unsupported scalar-clamp boundary gradients: {gradients}")
    return gradients[0] == 1.0


@torch.compiler.assume_constant_result
def _clamp_includes_boundaries():
    return _probe_clamp_boundaries()


@triton.jit
def _activate(
    gate,
    linear,
    clamp_value,
    gate_scale,
    linear_scale,
    MODE: tl.constexpr,
    HAS_LINEAR_CLAMP: tl.constexpr,
):
    """Forward value of one gate/linear pair, in fp32."""
    if MODE == 0:
        return gate * _sigmoid(gate) * linear
    if MODE == 1:
        gate_c = tl.minimum(gate, clamp_value)
        linear_c = tl.minimum(tl.maximum(linear, -clamp_value), clamp_value)
        return gate_c * _sigmoid(gate_c) * linear_c
    t = _tanh(gate / gate_scale)
    gate_v = (gate_scale * t) * _sigmoid(gate)
    if HAS_LINEAR_CLAMP:
        lin = linear_scale * _tanh(linear / linear_scale)
    else:
        lin = linear
    return gate_v * lin


@triton.jit
def _activate_backward(
    gate,
    linear,
    dy,
    clamp_value,
    gate_scale,
    linear_scale,
    MODE: tl.constexpr,
    HAS_LINEAR_CLAMP: tl.constexpr,
    INCLUSIVE: tl.constexpr,
):
    """Input gradients and the forward value of one pair.

    ``dy`` already includes the per-token weight. The returned forward value
    does not, so the weight gradient can be ``h * grad_output``.
    """
    if MODE == 0:
        sig = _sigmoid(gate)
        h = gate * sig * linear
        dgate = dy * sig * (1.0 + gate * (1.0 - sig)) * linear
        dlinear = dy * gate * sig
        return dgate, dlinear, h
    if MODE == 1:
        gate_c = tl.minimum(gate, clamp_value)
        linear_c = tl.minimum(tl.maximum(linear, -clamp_value), clamp_value)
        sig = _sigmoid(gate_c)
        silu = gate_c * sig
        h = silu * linear_c
        dgate = dy * sig * (1.0 + gate_c * (1.0 - sig)) * linear_c
        dlinear = dy * silu
        if INCLUSIVE:
            gate_ok = gate <= clamp_value
            linear_ok = (linear >= -clamp_value) & (linear <= clamp_value)
        else:
            gate_ok = gate < clamp_value
            linear_ok = (linear > -clamp_value) & (linear < clamp_value)
        dgate = tl.where(gate_ok, dgate, 0.0)
        dlinear = tl.where(linear_ok, dlinear, 0.0)
        return dgate, dlinear, h
    t = _tanh(gate / gate_scale)
    c = gate_scale * t
    sech2 = 1.0 - t * t
    sig = _sigmoid(gate)
    gate_v = c * sig
    if HAS_LINEAR_CLAMP:
        t_lin = _tanh(linear / linear_scale)
        lin = linear_scale * t_lin
        dlinear = dy * (1.0 - t_lin * t_lin) * gate_v
    else:
        lin = linear
        dlinear = dy * gate_v
    dgate = dy * (sech2 + c * (1.0 - sig)) * sig * lin
    return dgate, dlinear, gate_v * lin


@libentry()
@triton.jit
def _bias_swiglu_forward_kernel(
    x_ptr,
    bias_ptr,
    weight_ptr,
    y_ptr,
    n_cols,
    clamp_value,
    gate_scale,
    linear_scale,
    BIAS_MODE: tl.constexpr,
    WEIGHT_MODE: tl.constexpr,
    MODE: tl.constexpr,
    HAS_LINEAR_CLAMP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tle.program_id(0)
    hidden = n_cols * 2
    x_row = x_ptr + row * hidden
    y_row = y_ptr + row * n_cols
    bias_row = bias_ptr + row * hidden
    weight = 0.0
    if WEIGHT_MODE == 1:
        weight = tl.load(weight_ptr).to(tl.float32)
    elif WEIGHT_MODE == 2:
        weight = tl.load(weight_ptr + row).to(tl.float32)

    for start in tl.range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        gate = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        linear = tl.load(x_row + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        if BIAS_MODE == 1:
            gate += tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            linear += tl.load(bias_ptr + n_cols + cols, mask=mask, other=0.0).to(
                tl.float32
            )
        elif BIAS_MODE == 2:
            gate += tl.load(bias_row + cols, mask=mask, other=0.0).to(tl.float32)
            linear += tl.load(bias_row + n_cols + cols, mask=mask, other=0.0).to(
                tl.float32
            )
        out = _activate(
            gate,
            linear,
            clamp_value,
            gate_scale,
            linear_scale,
            MODE,
            HAS_LINEAR_CLAMP,
        )
        if WEIGHT_MODE == 3:
            out *= tl.load(weight_ptr + row * n_cols + cols, mask=mask, other=0.0).to(
                tl.float32
            )
        elif WEIGHT_MODE != 0:
            out *= weight
        tl.store(y_row + cols, out.to(y_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def _bias_swiglu_backward_kernel(
    x_ptr,
    bias_ptr,
    weight_ptr,
    dy_ptr,
    dx_ptr,
    dg_ptr,
    dw_ptr,
    n_cols,
    clamp_value,
    gate_scale,
    linear_scale,
    BIAS_MODE: tl.constexpr,
    WEIGHT_MODE: tl.constexpr,
    MODE: tl.constexpr,
    HAS_LINEAR_CLAMP: tl.constexpr,
    INCLUSIVE: tl.constexpr,
    COMPUTE_BGRAD: tl.constexpr,
    COMPUTE_WGRAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tle.program_id(0)
    hidden = n_cols * 2
    x_row = x_ptr + row * hidden
    dx_row = dx_ptr + row * hidden
    dg_row = dg_ptr + row * hidden
    dy_row = dy_ptr + row * n_cols
    bias_row = bias_ptr + row * hidden
    weight = 0.0
    row_weight_grad = 0.0
    if WEIGHT_MODE == 1:
        weight = tl.load(weight_ptr).to(tl.float32)
    elif WEIGHT_MODE == 2:
        weight = tl.load(weight_ptr + row).to(tl.float32)

    for start in tl.range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        gate = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        linear = tl.load(x_row + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        if BIAS_MODE == 1:
            gate += tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            linear += tl.load(bias_ptr + n_cols + cols, mask=mask, other=0.0).to(
                tl.float32
            )
        elif BIAS_MODE == 2:
            gate += tl.load(bias_row + cols, mask=mask, other=0.0).to(tl.float32)
            linear += tl.load(bias_row + n_cols + cols, mask=mask, other=0.0).to(
                tl.float32
            )
        dy = tl.load(dy_row + cols, mask=mask, other=0.0).to(tl.float32)
        if WEIGHT_MODE == 3:
            w = tl.load(weight_ptr + row * n_cols + cols, mask=mask, other=0.0).to(
                tl.float32
            )
        elif WEIGHT_MODE != 0:
            w = weight
        else:
            w = 1.0
        dgate, dlinear, activated = _activate_backward(
            gate,
            linear,
            dy * w,
            clamp_value,
            gate_scale,
            linear_scale,
            MODE,
            HAS_LINEAR_CLAMP,
            INCLUSIVE,
        )
        tl.store(dx_row + cols, dgate.to(dx_ptr.dtype.element_ty), mask=mask)
        tl.store(
            dx_row + n_cols + cols,
            dlinear.to(dx_ptr.dtype.element_ty),
            mask=mask,
        )
        if COMPUTE_BGRAD:
            # Keep the fp32 values for the bias reduction; rounding the
            # per-token gradient to half before summing over many tokens
            # loses more than one ulp of the final bias gradient.
            tl.store(dg_row + cols, dgate, mask=mask)
            tl.store(dg_row + n_cols + cols, dlinear, mask=mask)
        if COMPUTE_WGRAD:
            if WEIGHT_MODE == 3:
                tl.store(
                    dw_ptr + row * n_cols + cols,
                    (activated * dy).to(dw_ptr.dtype.element_ty),
                    mask=mask,
                )
            elif WEIGHT_MODE != 0:
                row_weight_grad += tl.sum(tl.where(mask, activated * dy, 0.0), axis=0)
    if COMPUTE_WGRAD and (WEIGHT_MODE == 1 or WEIGHT_MODE == 2):
        tl.store(dw_ptr + row, row_weight_grad.to(dw_ptr.dtype.element_ty))


def _block_size(n_cols):
    """Power-of-two column tile, capped at the tuned maximum."""
    return max(_MIN_BLOCK_SIZE, min(_BLOCK_SIZE, triton.next_power_of_2(n_cols)))


def _num_warps(block):
    """Do not launch threads that have no columns to read on a narrow row."""
    if block <= 64:
        return 1
    if block <= 256:
        return 2
    return _NUM_WARPS


def _launch_forward(
    x,
    bias,
    bias_mode,
    weight,
    weight_mode,
    y,
    variant,
    clamp_value,
    gate_scale,
    linear_scale,
):
    mode, has_linear_clamp = variant
    n_rows, hidden = x.shape
    n_cols = hidden // 2
    grid = (n_rows,)
    block = _block_size(n_cols)
    with torch_device_fn.device(x.device):
        _bias_swiglu_forward_kernel[grid](
            x,
            # The constexpr mode stops the kernel from reading this pointer,
            # but Triton still type-checks the argument, so reuse ``x``.
            x if bias is None else bias,
            x if weight is None else weight,
            y,
            n_cols,
            0.0 if clamp_value is None else clamp_value,
            1.0 if gate_scale is None else gate_scale,
            1.0 if linear_scale is None else linear_scale,
            BIAS_MODE=bias_mode,
            WEIGHT_MODE=weight_mode,
            MODE=mode,
            HAS_LINEAR_CLAMP=has_linear_clamp,
            BLOCK=block,
            num_warps=_num_warps(block),
        )


def _launch_backward(
    x,
    bias,
    bias_mode,
    weight,
    weight_mode,
    grad_output,
    dx,
    dg,
    dw,
    variant,
    clamp_value,
    gate_scale,
    linear_scale,
    inclusive,
    compute_bgrad,
    compute_wgrad,
):
    mode, has_linear_clamp = variant
    n_rows, hidden = x.shape
    n_cols = hidden // 2
    grid = (n_rows,)
    block = _block_size(n_cols)
    with torch_device_fn.device(x.device):
        _bias_swiglu_backward_kernel[grid](
            x,
            x if bias is None else bias,
            x if weight is None else weight,
            grad_output,
            dx,
            dx if dg is None else dg,
            dx if dw is None else dw,
            n_cols,
            0.0 if clamp_value is None else clamp_value,
            1.0 if gate_scale is None else gate_scale,
            1.0 if linear_scale is None else linear_scale,
            BIAS_MODE=bias_mode,
            WEIGHT_MODE=weight_mode,
            MODE=mode,
            HAS_LINEAR_CLAMP=has_linear_clamp,
            INCLUSIVE=inclusive,
            COMPUTE_BGRAD=compute_bgrad,
            COMPUTE_WGRAD=compute_wgrad,
            BLOCK=block,
            num_warps=_num_warps(block),
        )


def _flatten_input(x):
    _check_input(x, "bias_swiglu")
    if x.ndim not in (2, 3):
        raise ValueError(
            f"bias_swiglu input must have rank 2 or 3, got rank {x.ndim} "
            f"shape {tuple(x.shape)}"
        )
    hidden = x.shape[-1]
    if hidden % 2 != 0:
        raise ValueError(
            f"bias_swiglu input's last dimension must be even, got {hidden}"
        )
    flat = x.contiguous().view(-1, hidden)
    out_shape = (*x.shape[:-1], hidden // 2)
    return flat, out_shape


def _empty_output(x, out_shape):
    return x.new_empty(out_shape)


def _forward_from_flat(
    flat, bias, weights, out_shape, clamp_value, gate_scale, linear_scale, variant
):
    n_rows, hidden = flat.shape
    n_cols = hidden // 2
    # Validate and normalize the optional operands before the empty-input
    # shortcut, so an empty call rejects a bad bias exactly like a non-empty one.
    bias_tensor, bias_mode = _prepare_bias(bias, n_rows, hidden)
    if bias is not None:
        _check_same_device(bias, "bias_swiglu bias", flat.device)
        if bias.dtype != flat.dtype:
            raise ValueError(
                "bias_swiglu bias must have the input dtype, got "
                f"{bias.dtype} for {flat.dtype}"
            )
    weight_tensor, weight_mode = _prepare_weight(weights, out_shape)
    if weights is not None:
        _check_same_device(weights, "bias_swiglu weights", flat.device)

    if n_rows == 0 or n_cols == 0:
        return _empty_output(flat, (n_rows, n_cols)).view(out_shape)

    y = flat.new_empty((n_rows, n_cols))
    _launch_forward(
        flat,
        bias_tensor,
        bias_mode,
        weight_tensor,
        weight_mode,
        y,
        variant,
        clamp_value,
        gate_scale,
        linear_scale,
    )
    return y.view(out_shape)


def _backward_from_flat(
    grad_output,
    flat,
    bias,
    weights,
    out_shape,
    clamp_value,
    gate_scale,
    linear_scale,
    variant,
    compute_bias_grad,
    compute_weight_grad,
):
    n_rows, hidden = flat.shape
    n_cols = hidden // 2
    if grad_output.shape != out_shape or grad_output.dtype != flat.dtype:
        raise ValueError(
            "bias_swiglu grad needs the output shape and dtype, got "
            f"{tuple(grad_output.shape)} {grad_output.dtype} for "
            f"{tuple(out_shape)} {flat.dtype}"
        )
    _check_input(grad_output, "bias_swiglu grad")
    _check_same_device(grad_output, "bias_swiglu grad", flat.device)

    grad_flat = grad_output.contiguous().view(n_rows, n_cols)
    bias_tensor, bias_mode = _prepare_bias(bias, n_rows, hidden)
    weight_tensor, weight_mode = _prepare_weight(weights, out_shape)
    if bias is not None:
        _check_same_device(bias, "bias_swiglu bias", flat.device)
    if weights is not None:
        _check_same_device(weights, "bias_swiglu weights", flat.device)

    if n_rows == 0 or n_cols == 0:
        grad_x = torch.zeros_like(flat)
        grad_bias = (
            torch.zeros_like(bias) if compute_bias_grad and bias is not None else None
        )
        grad_weight = (
            torch.zeros_like(weights)
            if compute_weight_grad and weights is not None
            else None
        )
        return grad_x, grad_bias, grad_weight

    # The input gradient is written in the input dtype. When the bias needs a
    # gradient, the kernel also writes the unrounded fp32 values to ``dg`` so
    # the reduction over tokens does not accumulate per-token rounding error.
    dx = torch.empty((n_rows, hidden), dtype=flat.dtype, device=flat.device)
    dg = None
    if compute_bias_grad:
        # For an fp32 input the input gradient already has the precision the
        # bias reduction needs, so both pointers can share one buffer.
        dg = (
            dx
            if flat.dtype == torch.float32
            else torch.empty((n_rows, hidden), dtype=torch.float32, device=flat.device)
        )
    dw = None
    if compute_weight_grad:
        if weight_mode in (_WEIGHT_SCALAR, _WEIGHT_ROW):
            dw = torch.empty((n_rows,), dtype=torch.float32, device=flat.device)
        elif weight_mode == _WEIGHT_ELEMENT:
            dw = torch.empty((n_rows, n_cols), dtype=weights.dtype, device=flat.device)

    inclusive = True
    if variant[0] == _MODE_CLAMPED:
        inclusive = _clamp_includes_boundaries()
    _launch_backward(
        flat,
        bias_tensor,
        bias_mode,
        weight_tensor,
        weight_mode,
        grad_flat,
        dx,
        dg,
        dw,
        variant,
        clamp_value,
        gate_scale,
        linear_scale,
        inclusive,
        compute_bias_grad,
        compute_weight_grad,
    )
    grad_x = dx
    grad_bias = None
    if compute_bias_grad and bias is not None:
        grad_bias = _sum_to(dg, bias.shape).to(bias.dtype)
    grad_weight = None
    if compute_weight_grad and weights is not None:
        summed = weight_mode != _WEIGHT_ELEMENT
        reduced = dw.reshape(n_rows, 1) if summed else dw
        grad_weight = _weight_grad(reduced, weights, out_shape, summed)
    return grad_x, grad_bias, grad_weight


def _mark_offload(tensor, enabled):
    if enabled and tensor is not None:
        tensor.activation_offloading = True


class _SwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        flat,
        bias,
        weights,
        fp8_input_store,
        cpu_offload_input,
        clamp_value,
        gate_scale,
        linear_scale,
        out_shape,
    ):
        out_shape = tuple(out_shape)
        variant = _resolve_variant(clamp_value, gate_scale, linear_scale)
        mode, _, resolved_linear, has_linear_clamp = variant
        # ``_resolve_variant`` may replace the scales with plain floats.
        # Re-read them so the kernel and the backward see the validated values.
        if mode == _MODE_SITU:
            gate_scale = variant[1]
            linear_scale = resolved_linear
        elif mode == _MODE_CLAMPED:
            clamp_value = variant[1]
        saved_input = flat.to(torch.float8_e4m3fn) if fp8_input_store else flat
        _mark_offload(saved_input, cpu_offload_input)
        _mark_offload(bias, cpu_offload_input)
        if bias is None and weights is None:
            ctx.save_for_backward(saved_input)
        elif weights is None:
            ctx.save_for_backward(saved_input, bias)
        elif bias is None:
            ctx.save_for_backward(saved_input, weights)
        else:
            ctx.save_for_backward(saved_input, bias, weights)
        ctx.fp8_input_store = fp8_input_store
        ctx.input_dtype = flat.dtype
        ctx.clamp_value = clamp_value if mode == _MODE_CLAMPED else None
        ctx.gate_scale = gate_scale if mode == _MODE_SITU else None
        ctx.linear_scale = linear_scale if has_linear_clamp else None
        ctx.out_shape = out_shape
        ctx.has_bias = bias is not None
        ctx.has_weights = weights is not None
        return _forward_from_flat(
            flat,
            bias,
            weights,
            out_shape,
            ctx.clamp_value,
            ctx.gate_scale,
            ctx.linear_scale,
            (mode, has_linear_clamp),
        )

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("TRAIN BIAS_SWIGLU_BACKWARD")
        saved = ctx.saved_tensors
        flat = saved[0]
        if ctx.fp8_input_store:
            flat = flat.to(ctx.input_dtype)
        bias = None
        weights = None
        next_index = 1
        if ctx.has_bias:
            bias = saved[next_index]
            next_index += 1
        if ctx.has_weights:
            weights = saved[next_index]
        mode = _MODE_PLAIN
        has_linear_clamp = ctx.linear_scale is not None
        if ctx.gate_scale is not None:
            mode = _MODE_SITU
        elif ctx.clamp_value is not None:
            mode = _MODE_CLAMPED
        # Inputs are (flat, bias, weights, ...); skip work whose gradient the
        # caller does not consume.
        grad_x, grad_bias, grad_weight = _backward_from_flat(
            grad_output,
            flat,
            bias,
            weights,
            ctx.out_shape,
            ctx.clamp_value,
            ctx.gate_scale,
            ctx.linear_scale,
            (mode, has_linear_clamp),
            ctx.needs_input_grad[1] and ctx.has_bias,
            ctx.needs_input_grad[2] and ctx.has_weights,
        )
        return (grad_x, grad_bias, grad_weight, None, None, None, None, None, None)


def _apply(
    x,
    bias,
    weights,
    fp8_input_store,
    cpu_offload_input,
    clamp_value,
    gate_clamp_scale,
    linear_clamp_scale,
):
    flat, out_shape = _flatten_input(x)
    # Bias and weights are validated and normalized in ``_forward_from_flat``,
    # which also runs for empty inputs.
    return _SwiGLUFn.apply(
        flat,
        bias,
        weights,
        bool(fp8_input_store),
        bool(cpu_offload_input),
        clamp_value,
        gate_clamp_scale,
        linear_clamp_scale,
        out_shape,
    )


def bias_swiglu_impl(
    input,
    bias,
    fp8_input_store=False,
    cpu_offload_input=False,
    clamp_value=None,
    gate_clamp_scale=None,
    linear_clamp_scale=None,
):
    """Bias plus SwiGLU, or SiTU-GLU when ``gate_clamp_scale`` is set.

    Args:
        input (Tensor): rank-2 or rank-3 fp16, bf16 or fp32 tensor on the
            active device. The last dimension is ``[gate | linear]`` and must
            be even. A non-contiguous input is copied.
        bias (Tensor or None): added to ``input`` before the activation.
            ``None`` skips the add. A bias of shape ``[hidden]`` is the
            ``skip_bias_add`` case. It must have the same dtype as ``input``.
            The backward reduces the bias gradient to ``bias.shape``.
        fp8_input_store (bool): save the pre-activation as ``float8_e4m3fn``
            and cast it back for the backward. The forward result is unchanged.
        cpu_offload_input (bool): mark the saved input and bias with
            ``activation_offloading`` for a caller that offloads activations.
        clamp_value (float or None): hard-clamp the gate at this upper bound
            and the linear half to ``[-clamp_value, clamp_value]``.
        gate_clamp_scale (float or None): select SiTU-GLU. The gate's linear
            factor becomes ``scale * tanh(x / scale)``.
        linear_clamp_scale (float or None): also soft-clamp the linear half.
            Requires ``gate_clamp_scale``.

    Returns:
        Tensor: same rank and dtype as ``input``, with the last dimension halved.
    """
    logger.debug("TRAIN BIAS_SWIGLU")
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError(
            f"bias_swiglu bias needs a tensor or None, got {type(bias).__name__}"
        )
    if cpu_offload_input:
        # The autograd save marks the flattened tensor. Also mark the tensors
        # the caller still holds; a contiguous rank-2 input is that same tensor
        # only when ``view`` returns it, which is not guaranteed.
        input.activation_offloading = True
        if bias is not None:
            bias.activation_offloading = True
    return _apply(
        input,
        bias,
        None,
        fp8_input_store,
        cpu_offload_input,
        clamp_value,
        gate_clamp_scale,
        linear_clamp_scale,
    )


def weighted_bias_swiglu_impl(
    input,
    bias,
    weights,
    fp8_input_store=False,
    clamp_value=None,
    gate_clamp_scale=None,
    linear_clamp_scale=None,
):
    """Per-token weighted SwiGLU or SiTU-GLU.

    ``bias`` is accepted so the signature matches Megatron-LM. A non-``None``
    bias raises ``NotImplementedError``, as it does there. ``weights``
    broadcasts with the output; the usual MoE shape is ``[token, 1]`` or the
    unflattened ``[*batch, 1]``.
    """
    logger.debug("TRAIN WEIGHTED_BIAS_SWIGLU")
    if bias is not None:
        raise NotImplementedError("Bias is not supported for weighted swiglu fusion")
    if not isinstance(weights, torch.Tensor):
        raise TypeError(
            f"bias_swiglu weights needs a tensor, got {type(weights).__name__}"
        )
    return _apply(
        input,
        None,
        weights,
        fp8_input_store,
        False,
        clamp_value,
        gate_clamp_scale,
        linear_clamp_scale,
    )
