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
"""Megatron-LM Tanh Soft Clamp activation, implemented entirely in Triton.

The arithmetic is Megatron's ``(scale * tanh(x.float() / scale)).to(x.dtype)``:
operands widen to FP32, the tanh runs there, and the result is cast back to the
input dtype. Every supported tensor is processed by a Triton kernel; the host
only validates, normalizes/broadcasts operands (copies), and stages CPU inputs
through the accelerator.

Supported dtypes are those Triton can load and store: float16/32/64, bfloat16,
int8/16/32/64, uint8 and bool. float8 and the 16/32/64-bit unsigned types raise
``ValueError`` (this backend cannot represent them in a kernel), and float64
needs a device that reports FP64 support. Complex inputs are rejected.
"""

import logging
import numbers
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry, tl_extra_shim
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_INTEGRAL_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
)
_KERNEL_DTYPES = _FLOAT_DTYPES + _INTEGRAL_DTYPES
_BLOCK_SIZE = 256
_NUM_WARPS = 4
_IS_ASCEND = runtime_device.vendor_name == "ascend"
# Ascend's measured optimum: one 16384-lane tile per walk step, one program per
# vector core.
_ASCEND_BLOCK = 16384
if _IS_ASCEND:
    from flag_train.runtime.backend._ascend.utils import CORE_NUM


@triton.jit
def _tanh_soft_clamp_tile(
    X,
    S,
    Y,
    N,
    tile,
    SCALE: tl.constexpr,
    SCALE_MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask, other=0).to(tl.float32)
    if SCALE_MODE == 0:
        # Megatron's expression divides; dividing here too keeps the same
        # rounding, which the integer casts below can otherwise expose.
        y = SCALE * tl_extra_shim.tanh(x / SCALE)
    elif SCALE_MODE == 1:
        # A one-element scale tensor: loaded once per tile, not per lane.
        s = tl.load(S).to(tl.float32)
        y = s * tl_extra_shim.tanh(x / s)
    else:
        s = tl.load(S + offsets, mask, other=0).to(tl.float32)
        y = s * tl_extra_shim.tanh(x / s)
    tl.store(Y + offsets, y.to(Y.dtype.element_ty), mask)


@libentry()
@triton.jit
def _tanh_soft_clamp_kernel(
    X,
    S,
    Y,
    N,
    SCALE: tl.constexpr,
    SCALE_MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    _tanh_soft_clamp_tile(
        X, S, Y, N, tle.program_id(0), SCALE, SCALE_MODE, BLOCK
    )


@libentry()
@triton.jit
def _tanh_soft_clamp_walk_kernel(
    X,
    S,
    Y,
    N,
    SCALE: tl.constexpr,
    SCALE_MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program per vector core walks its tiles; the runtime trip count keeps
    # the loop from unrolling.
    tiles = tl.cdiv(N, BLOCK)
    for tile in range(tle.program_id(0), tiles, tl.num_programs(0)):
        _tanh_soft_clamp_tile(
            X, S, Y, N, tile, SCALE, SCALE_MODE, BLOCK
        )


def _accelerator_device():
    name = runtime_device.name
    if name == "cpu":
        raise RuntimeError(
            "tanh_soft_clamp cannot run CPU inputs without an accelerator: "
            "no device Triton backend is available"
        )
    is_available = getattr(torch_device_fn, "is_available", None)
    if callable(is_available) and not is_available():
        raise RuntimeError(
            f"tanh_soft_clamp cannot run CPU inputs: {name} is unavailable"
        )
    return name


def _check_dtype(tensor, name):
    if tensor.is_complex():
        raise TypeError(
            f"tanh_soft_clamp does not support complex {name}, got {tensor.dtype}"
        )
    if tensor.dtype not in _KERNEL_DTYPES:
        raise ValueError(f"tanh_soft_clamp does not support dtype {tensor.dtype}")
    if tensor.dtype is torch.float64 and not runtime_device.support_fp64:
        raise ValueError("tanh_soft_clamp needs a device with FP64 support for float64")


def _validate(x, scale):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"tanh_soft_clamp expects a tensor, got {type(x).__name__}")
    _check_dtype(x, "input")
    if x.device.type not in ("cpu", runtime_device.name):
        raise ValueError(
            f"tanh_soft_clamp supports CPU or {runtime_device.name} tensors, "
            f"got {x.device.type}"
        )
    if isinstance(scale, torch.Tensor):
        _check_dtype(scale, "scale")
        return
    if isinstance(scale, numbers.Real):
        return
    raise TypeError(
        "tanh_soft_clamp scale must be a real number or a tensor, got "
        f"{type(scale).__name__}"
    )


def _dense_storage(tensor):
    """Whether ``tensor`` covers its whole storage in a permuted (dense) layout."""
    return (
        tensor.storage_offset() == 0
        and tensor.numel() * tensor.element_size() == tensor.untyped_storage().size()
    )


def _flat_storage(tensor):
    """1-D view over a dense tensor's storage (no copy)."""
    return torch.as_strided(tensor, (tensor.numel(),), (1,))


def _copy_back(tensor):
    """Move a result to the CPU keeping its strides (``.to`` flattens them)."""
    cpu = torch.empty_strided(
        tensor.shape, tensor.stride(), dtype=tensor.dtype, device="cpu"
    )
    cpu.copy_(tensor)
    return cpu


def tanh_soft_clamp(x: torch.Tensor, scale) -> torch.Tensor:
    """Return Megatron's ``(scale * tanh(x.float() / scale)).to(x.dtype)``.

    ``scale`` is a real Python number or a real tensor broadcastable against
    ``x``. The output has ``x``'s dtype and the broadcast shape of ``x`` and a
    tensor ``scale``. CPU operands are staged through the accelerator and the
    result is copied back. No backward operator is provided.
    """
    logger.debug("TRAIN TANH_SOFT_CLAMP")
    _validate(x, scale)
    scale_is_tensor = isinstance(scale, torch.Tensor)

    result_device = x.device
    staged = result_device.type == "cpu"
    compute_device = _accelerator_device() if staged else result_device
    if staged:
        x = x.to(compute_device)
        if scale_is_tensor:
            scale = scale.to(compute_device)

    if scale_is_tensor:
        if scale.dim() == 0 and scale.device.type == "cpu":
            # torch's 0-dimensional CPU-scalar rule, as Megatron's expression
            # relies on. (A CPU input was staged together with the scale above,
            # so reaching this point means the input is on the accelerator.)
            scale = scale.to(compute_device)
        if scale.device.type != x.device.type:
            raise ValueError("tanh_soft_clamp scale and input must share a device")
        out_shape = torch.broadcast_shapes(x.shape, scale.shape)
    else:
        out_shape = x.shape

    if scale_is_tensor:
        if scale.numel() == 1:
            mode = 1
            scale_arg = scale.reshape(1).contiguous()
        elif out_shape != x.shape:
            # A scale that expands the input: materialize both operands at the
            # broadcast shape (copies, no arithmetic on the host).
            x = torch.broadcast_to(x, out_shape).contiguous()
            scale_arg = torch.broadcast_to(scale, out_shape).contiguous().view(-1)
            mode = 2
        elif scale.numel() == x.numel() and scale.is_contiguous():
            mode = 2
            scale_arg = scale.view(-1)
        else:
            mode = 2
            scale_arg = torch.broadcast_to(scale, x.shape).contiguous().view(-1)
    else:
        mode = 0
        scale_arg = None

    out_dtype = x.dtype
    if out_shape == x.shape:
        dense = x.is_contiguous() or _dense_storage(x)
        x_flat = x.view(-1) if x.is_contiguous() else (
            _flat_storage(x) if dense else x.contiguous().view(-1)
        )
        if dense:
            # Same layout as the input, so the strides survive a transposed or
            # permuted view -- which is what Megatron's eager expression does.
            out = torch.empty_like(x, memory_format=torch.preserve_format)
        else:
            out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
        out_flat = _flat_storage(out) if _dense_storage(out) else out.view(-1)
    else:
        out = torch.empty(out_shape, dtype=out_dtype, device=x.device)
        x_flat = x.view(-1)
        out_flat = out.view(-1)

    n = x_flat.numel()
    if not n:
        return _copy_back(out) if staged else out

    if mode == 0:
        scalar = float(scale)
        s_arg = x_flat  # never dereferenced under SCALE_MODE == 0
        if scalar == 0.0:
            # A compile-time division by zero is not IEEE; hand the kernel a
            # one-element zero instead so the division happens at runtime.
            mode = 1
            s_arg = torch.zeros(1, dtype=torch.float32, device=x.device)
    else:
        scalar = 0.0
        s_arg = scale_arg

    if _IS_ASCEND:
        if n <= 16384:
            # Launch-bound sizes: a handful of wide programs beats one program
            # per core (measured; 4096 elements: 1.33 us at 4 programs against
            # 2.18 us at 32).
            target = max(1, n // 4)
            block = min(2048, max(64, 1 << (target - 1).bit_length()))
            grid = (triton.cdiv(n, block),)
            kernel = _tanh_soft_clamp_kernel
        else:
            lanes = (n + CORE_NUM - 1) // CORE_NUM
            block = min(_ASCEND_BLOCK, max(128, 1 << (lanes - 1).bit_length()))
            tiles = (n + block - 1) // block
            grid = (min(tiles, CORE_NUM),)
            # The walk only pays when one program owns several tiles.
            kernel = (
                _tanh_soft_clamp_kernel
                if tiles <= CORE_NUM
                else _tanh_soft_clamp_walk_kernel
            )
    else:
        block = _BLOCK_SIZE
        if n >= 65536:
            block = 1024 if out_dtype in (torch.float32, torch.float64) else 2048
        grid = (triton.cdiv(n, block),)
        kernel = _tanh_soft_clamp_kernel

    no_switch = (
        result_device.type != "cpu"
        and torch_device_fn.current_device() == result_device.index
    )
    context = nullcontext() if no_switch else torch_device_fn.device(compute_device)
    with context:
        kernel[grid](
            x_flat,
            s_arg,
            out_flat,
            n,
            SCALE=scalar,
            SCALE_MODE=mode,
            BLOCK=block,
            num_warps=_NUM_WARPS,
            **({"num_stages": 1} if _IS_ASCEND else {}),
        )
    return _copy_back(out) if staged else out
