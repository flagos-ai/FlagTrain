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

"""Fused bias + GeGLU (forward).

Forward only. One token is one row; the last dimension is the concatenated
``[gate | linear]`` and the output keeps the gated half.

All activation arithmetic runs in the Triton kernel, including the bool and
integer addition rules of Megatron's expression. The host only decides torch's
promotion (``torch.promote_types``), validates operands, computes broadcast
shapes, materializes expanded/contiguous copies and stages CPU tensors through
the active accelerator; it never evaluates the activation with torch.

The numeric contract is the high-precision one: floating operands are widened
to FP32 (FP64 for a float64 promotion), the bias add and the activation both
run there, and the result is rounded once when it is stored. That tracks the
upcast reference the correctness tests compare against; it is not eager's
operand-dtype round trip (Megatron computes its forward in, say, BF16
throughout). Integer and bool operands follow torch's promoted-width wrap and
OR rules, because those are exact.

Supported real dtypes are the ones Triton can load and store here:
float16/bfloat16/float32/float64, int8/16/32/64, uint8 and bool. Complex and
the 16/32/64-bit unsigned types are rejected. CPU inputs are staged through
the accelerator and copied back; without an accelerator they cannot run
because there is no CPU Triton backend.

Autograd/backward is out of scope: no state is saved and no
``torch.autograd.Function`` is defined, so inputs requiring grad do not build
a graph.
"""

import logging
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry, tl_extra_shim
from flag_train.utils import triton_lang_extension as tle

__all__ = ["bias_geglu_impl"]

logger = logging.getLogger(__name__)

_NUM_WARPS = 4
_BLOCK_SIZE = 1024
_MIN_BLOCK_SIZE = 16

# Real dtypes the Triton kernel can load and store on this stack.
_SUPPORTED_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
)
# ADD_KIND codes for the promoted integer addition; 0 is floating add and 1
# is the bool OR that torch uses for bool + bool.
_INT_ADD_KIND = {
    torch.int8: 2,
    torch.int16: 3,
    torch.int32: 4,
    torch.int64: 5,
    torch.uint8: 6,
}

# ``tl_extra_shim.tanh`` maps to libdevice and falls back to an exp-based
# form on backends without it.
_tanh = tl_extra_shim.tanh


_BIAS_NONE = 0
_BIAS_VECTOR = 1
_BIAS_FULL = 2

# sqrt(2/pi) and the cubic coefficient of the tanh GELU approximation.
_GELU_SQRT_2_OVER_PI = tl.constexpr(0.79788456)
_GELU_COEFF = tl.constexpr(0.044715)


def _accelerator_device():
    """The active accelerator used to stage CPU operands."""
    name = runtime_device.name
    if name == "cpu":
        raise RuntimeError(
            "bias_geglu cannot run CPU inputs without an accelerator: "
            "no device Triton backend is available"
        )
    is_available = getattr(torch_device_fn, "is_available", None)
    if callable(is_available) and not is_available():
        raise RuntimeError(f"bias_geglu cannot run CPU inputs: {name} is unavailable")
    return name


def _check_dtype(tensor, name):
    if tensor.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"bias_geglu {name} has an unsupported dtype {tensor.dtype}")


def _flatten_input(x):
    """Megatron's rank check and view, decided with upstream semantics.

    ``torch.ops.aten.view`` is used rather than ``x.view`` because Megatron-LM-FL
    patches ``torch.Tensor.view`` on NPU to copy a non-contiguous tensor first
    (``plugin/platform/platform_npu.py``); going through the patched method
    would silently accept a layout upstream Megatron rejects, while the aten op
    keeps the same accept/reject set on every backend.
    """
    ori_shape = x.shape
    assert len(ori_shape) in [2, 3]
    try:
        return torch.ops.aten.view(x, (-1, ori_shape[-1])), ori_shape
    except RuntimeError as exc:
        raise ValueError(
            "bias_geglu input must be view-compatible with "
            f"[tokens, {ori_shape[-1]}], which Megatron's view rejects for this "
            f"layout: {exc}"
        ) from None


def _is_row_vector_bias(bias, y_shape):
    """Whether the broadcast bias is one ``[hidden]`` vector shared by all rows."""
    hidden = y_shape[-1]
    if bias.numel() == 1:
        return True
    if bias.numel() != hidden or bias.shape[-1] != hidden:
        return False
    return all(dim == 1 for dim in bias.shape[:-1])


@triton.jit
def _activate(gate, linear):
    """Forward value of one gate/linear pair, in fp32 or fp64."""
    # tanh approximation of GELU:
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2)))
    t = _tanh(_GELU_SQRT_2_OVER_PI * gate * (1.0 + _GELU_COEFF * gate * gate))
    return 0.5 * gate * (1.0 + t) * linear


@triton.jit
def _wrapping_add(x, b, WIDTH: tl.constexpr, COMPUTE_DTYPE: tl.constexpr):
    """torch's integer add: wraps in ``WIDTH``, then widens for the activation.

    Triton promotes a sub-word add to int32, and Ascend executes it there, so
    the sum is narrowed back to ``WIDTH`` explicitly to restore torch's wrap.
    The int32 hop after the narrowing cast avoids Ascend's direct narrow-int to
    float conversion, which sign-extends a uint8 arithmetic result; on vendors
    that already wrap at ``WIDTH`` both casts are identities.
    """
    return (x.to(tl.int32) + b.to(tl.int32)).to(WIDTH).to(tl.int32).to(COMPUTE_DTYPE)


@triton.jit
def _bias_geglu_add(
    xg,
    xl,
    bg,
    bl,
    ADD_KIND: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """Bias add for one gate/linear tile pair; returns ``(gate, linear)``.

    ``ADD_KIND`` follows Megatron's expression: 0 adds in the compute dtype,
    1 is bool OR, and 2..6 add in int8/int16/int32/int64/uint8 respectively,
    casting the sum back so the promoted integer width wraps.
    """
    if ADD_KIND == 0:
        gate = xg.to(COMPUTE_DTYPE) + bg.to(COMPUTE_DTYPE)
        linear = xl.to(COMPUTE_DTYPE) + bl.to(COMPUTE_DTYPE)
    elif ADD_KIND == 1:
        gate = (xg.to(tl.int8) | bg.to(tl.int8)).to(COMPUTE_DTYPE)
        linear = (xl.to(tl.int8) | bl.to(tl.int8)).to(COMPUTE_DTYPE)
    elif ADD_KIND == 2:
        gate = _wrapping_add(xg, bg, tl.int8, COMPUTE_DTYPE)
        linear = _wrapping_add(xl, bl, tl.int8, COMPUTE_DTYPE)
    elif ADD_KIND == 3:
        gate = _wrapping_add(xg, bg, tl.int16, COMPUTE_DTYPE)
        linear = _wrapping_add(xl, bl, tl.int16, COMPUTE_DTYPE)
    elif ADD_KIND == 4:
        gate = _wrapping_add(xg, bg, tl.int32, COMPUTE_DTYPE)
        linear = _wrapping_add(xl, bl, tl.int32, COMPUTE_DTYPE)
    elif ADD_KIND == 5:
        # int64 keeps its own width: int32 would truncate the sum.
        gate = (xg.to(tl.int64) + bg.to(tl.int64)).to(COMPUTE_DTYPE)
        linear = (xl.to(tl.int64) + bl.to(tl.int64)).to(COMPUTE_DTYPE)
    else:
        gate = _wrapping_add(xg, bg, tl.uint8, COMPUTE_DTYPE)
        linear = _wrapping_add(xl, bl, tl.uint8, COMPUTE_DTYPE)
    return gate, linear


@triton.jit
def _bias_geglu_row(
    x_ptr,
    bias_ptr,
    y_ptr,
    row,
    n_cols,
    BIAS_MODE: tl.constexpr,
    ADD_KIND: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Bias add followed by GEGLU for one ``[2 * n_cols]`` row.

    Floating operands add in the compute dtype (FP32, FP64 for float64) and
    round once when the result is stored; ``_bias_geglu_add`` owns the
    promoted-width integer and bool rules.
    """
    COMPUTE_DTYPE: tl.constexpr = tl.float64 if IS_FP64 else tl.float32
    hidden = n_cols * 2
    x_row = x_ptr + row * hidden
    y_row = y_ptr + row * n_cols
    bias_row = bias_ptr + row * hidden

    for start in tl.range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        xg = tl.load(x_row + cols, mask=mask, other=0)
        xl = tl.load(x_row + n_cols + cols, mask=mask, other=0)
        if BIAS_MODE == 0:
            gate = xg.to(COMPUTE_DTYPE)
            linear = xl.to(COMPUTE_DTYPE)
        else:
            if BIAS_MODE == 1:
                bg = tl.load(bias_ptr + cols, mask=mask, other=0)
                bl = tl.load(bias_ptr + n_cols + cols, mask=mask, other=0)
            else:
                bg = tl.load(bias_row + cols, mask=mask, other=0)
                bl = tl.load(bias_row + n_cols + cols, mask=mask, other=0)
            gate, linear = _bias_geglu_add(xg, xl, bg, bl, ADD_KIND, COMPUTE_DTYPE)
        out = _activate(gate, linear)
        tl.store(y_row + cols, out.to(y_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def _bias_geglu_forward_kernel(
    x_ptr,
    bias_ptr,
    y_ptr,
    n_cols,
    BIAS_MODE: tl.constexpr,
    ADD_KIND: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Portable launch: one program per row."""
    _bias_geglu_row(
        x_ptr,
        bias_ptr,
        y_ptr,
        tle.program_id(0),
        n_cols,
        BIAS_MODE,
        ADD_KIND,
        IS_FP64,
        BLOCK,
    )


def _launch_config(n_cols):
    """Power-of-two column tile and the warps needed to cover it."""
    block = max(_MIN_BLOCK_SIZE, min(_BLOCK_SIZE, triton.next_power_of_2(n_cols)))
    if block <= 64:
        return block, 1
    if block <= 256:
        return block, 2
    return block, _NUM_WARPS


class _Prepared(NamedTuple):
    """Operands and launch parameters after validation, staging and copies."""

    x_flat: torch.Tensor
    bias_arg: torch.Tensor
    out: torch.Tensor
    out_flat: torch.Tensor
    rows: int
    half: int
    bias_mode: int
    add_kind: int
    is_fp64: bool
    staged: bool
    device: torch.device
    ori_shape: torch.Size


def _prepare(x, bias):
    """Validate/normalize operands and return what a launch needs.

    The host work is shared by the portable implementation and the Ascend
    override; only the launch geometry and kernel differ.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"bias_geglu expects a tensor, got {type(x).__name__}")
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError("bias_geglu bias must be a Tensor or None")

    flat, ori_shape = _flatten_input(x)
    _check_dtype(flat, "input")
    if bias is not None:
        _check_dtype(bias, "bias")

    hidden = flat.shape[-1]
    if hidden % 2 != 0:
        raise RuntimeError("bias_geglu input's last dimension must be even")

    staged = flat.device.type == "cpu"
    if staged:
        # Upstream rejects a CPU input paired with an accelerator bias.
        if bias is not None and bias.device.type != "cpu":
            raise RuntimeError("bias_geglu bias and input must share a device")
        compute_device = _accelerator_device()
        flat = flat.to(compute_device)
        if bias is not None:
            bias = bias.to(compute_device)
    else:
        if flat.device.type != runtime_device.name:
            raise ValueError(
                f"bias_geglu supports CPU or {runtime_device.name} tensors, "
                f"got {flat.device.type}"
            )
        if bias is not None:
            if bias.device != flat.device:
                # torch treats a CPU 0-dimensional tensor as a scalar.
                if bias.device.type == "cpu" and bias.dim() == 0:
                    bias = bias.to(flat.device)
                else:
                    raise RuntimeError("bias_geglu bias and input must share a device")

    if bias is None:
        y_dtype = flat.dtype
        add_kind = 0
    else:
        y_dtype = torch.promote_types(flat.dtype, bias.dtype)
        if y_dtype.is_floating_point:
            add_kind = 0
        elif y_dtype is torch.bool:
            add_kind = 1
        else:
            add_kind = _INT_ADD_KIND.get(y_dtype)
            if add_kind is None:
                raise ValueError(
                    f"bias_geglu does not support promoted dtype {y_dtype}"
                )
    out_dtype = y_dtype if y_dtype.is_floating_point else torch.get_default_dtype()
    if out_dtype is torch.float64 and not runtime_device.support_fp64:
        raise ValueError("bias_geglu needs a device with FP64 support for float64")

    y_shape = (
        flat.shape if bias is None else torch.broadcast_shapes(flat.shape, bias.shape)
    )
    hidden = y_shape[-1]
    half = hidden // 2
    rows = 1
    for dim in y_shape[:-1]:
        rows *= dim
    out = torch.empty(y_shape[:-1] + (half,), dtype=out_dtype, device=flat.device)
    is_fp64 = out_dtype is torch.float64

    if y_shape == flat.shape:
        x_flat = flat.contiguous().view(rows, hidden)
    else:
        x_flat = torch.broadcast_to(flat, y_shape).contiguous().view(rows, hidden)

    if bias is None:
        bias_mode = _BIAS_NONE
        bias_arg = x_flat  # never dereferenced under BIAS_MODE == 0
    elif _is_row_vector_bias(bias, y_shape):
        bias_arg = torch.broadcast_to(bias.reshape(-1), (hidden,)).contiguous()
        bias_mode = _BIAS_VECTOR
    else:
        bias_arg = torch.broadcast_to(bias, y_shape).contiguous().view(rows, hidden)
        bias_mode = _BIAS_FULL

    return _Prepared(
        x_flat,
        bias_arg,
        out,
        out.view(rows, half),
        rows,
        half,
        bias_mode,
        add_kind,
        is_fp64,
        staged,
        flat.device,
        ori_shape,
    )


def _launch(prep):
    """Launch the portable one-program-per-row kernel."""
    if prep.rows == 0 or prep.half == 0:
        return
    block, num_warps = _launch_config(prep.half)
    with torch_device_fn.device(prep.device):
        _bias_geglu_forward_kernel[(prep.rows,)](
            prep.x_flat,
            prep.bias_arg,
            prep.out_flat,
            prep.half,
            BIAS_MODE=prep.bias_mode,
            ADD_KIND=prep.add_kind,
            IS_FP64=prep.is_fp64,
            BLOCK=block,
            num_warps=num_warps,
        )


def _finish(prep):
    """The operator's return value: a staged result comes back on the CPU."""
    out = prep.out
    result = (
        out
        if len(prep.ori_shape) == 2
        else out.view(prep.ori_shape[0], prep.ori_shape[1], -1)
    )
    return result.cpu() if prep.staged else result


def bias_geglu_impl(input, bias):
    """Apply Megatron-compatible bias + GEGLU forward.

    The forward contract follows Megatron's expression for real dtypes: rank
    2 or 3, last dimension split into gate/linear, optional bias with torch
    broadcasting and dtype promotion, and the promoted output dtype (floating
    dtype, or the default floating dtype for a promoted integer/bool). The
    Triton kernel performs the bias addition and activation. Complex and
    16/32/64-bit unsigned operands are rejected.

    Floating operands are evaluated in FP32 (FP64 for a float64 promotion) and
    rounded once when the result is stored; integer and bool operands follow
    torch's promoted-width wrap and OR rules.

    CPU operands are staged through the active accelerator and the result is
    copied back; this needs a usable accelerator. Non-contiguous input layouts
    that Megatron's ``view`` accepts and broadcast-expanded biases are
    materialized as contiguous copies before the kernel. This is forward only:
    inputs requiring grad do not build an autograd graph.
    """
    logger.debug("TRAIN BIAS_GEGLU_IMPL")
    prep = _prepare(input, bias)
    _launch(prep)
    return _finish(prep)
