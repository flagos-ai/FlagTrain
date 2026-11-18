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
"""Forward-only bias + tanh GELU using the common Triton/FlagTree backend.

Bias addition rounds to the operand dtype before FP32 activation arithmetic.
The output is a plain tensor: this operator does not propagate gradients.
"""

import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry, tl_extra_shim
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_BLOCK_SIZE = 256
_NUM_WARPS = 4
_IS_ASCEND = runtime_device.vendor_name == "ascend"
_GPU_PATH = tl.constexpr(not _IS_ASCEND)
# Only this instruction capability differs; GPU scheduling and tiling are shared.
_HAS_NATIVE_TANH = tl.constexpr(runtime_device.vendor_name == "nvidia")
if _IS_ASCEND:
    from flag_train.runtime.backend._ascend.utils import CORE_NUM


@triton.jit
def _gpu_gelu(z, argument, NATIVE_TANH: tl.constexpr):
    if NATIVE_TANH:
        t = tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;",
            constraints="=f,f",
            args=[argument],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        return 0.5 * z * (1.0 + t)
    # Portable sigmoid identity: no CUDA libdevice or foreign inline assembly.
    return z * (1.0 / (1.0 + tl.exp(-2.0 * argument)))


@triton.jit
def _bias_gelu_tile(
    X,
    B,
    Y,
    N,
    H: tl.constexpr,
    BIAS_MODE: tl.constexpr,
    BLOCK: tl.constexpr,
    tile,
    ROWWISE: tl.constexpr,
    EVEN: tl.constexpr,
):
    if (ROWWISE and BIAS_MODE == 1) and not (H <= BLOCK and H & (H - 1) == 0):
        # Explicitly contiguous bias columns avoid Ascend's scalar gather
        # lowering of a vector remainder. Row tails have their own mask.
        columns = tile % tl.cdiv(H, BLOCK) * BLOCK + tl.arange(0, BLOCK)
        offsets = tile // tl.cdiv(H, BLOCK) * H + columns
        mask = (offsets < N) & (columns < H)
    else:
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
    if EVEN:
        mask = True
    x = tl.load(X + offsets, mask, other=0)
    if BIAS_MODE == 0:
        b = tl.load(B)
    elif BIAS_MODE == 1:
        if (ROWWISE and H <= BLOCK) and H & (H - 1) == 0:
            row = tl.load(B + tl.arange(0, H))
            b = tl.broadcast_to(row[None, :], (BLOCK // H, H)).reshape((BLOCK,))
        elif ROWWISE:
            b = tl.load(B + columns, mask, other=0)
        else:
            b = tl.load(B + offsets % H, mask, other=0)
    else:
        b = tl.load(B + offsets, mask, other=0)
    if not ROWWISE or X.dtype.element_ty != tl.float16:
        # Only FP16 native addition preserves the tested rounding contract.
        # Ascend BF16 legalization can retain an unrounded FP32 intermediate;
        # keep its explicit FP32 -> BF16 -> FP32 rounding barrier.
        # The established non-Ascend path, including tiny rows, is unchanged.
        x = x.to(tl.float32)
        b = b.to(tl.float32)
    # Match the standalone dtype-rounded add, rather than fusing an FP32 add.
    z = (x + b).to(X.dtype.element_ty).to(tl.float32)
    if ROWWISE and X.dtype.element_ty != tl.float32:
        # Same cubic, factored to remove one FP32 vector multiplication.
        argument = (0.79788456 + (0.79788456 * 0.044715) * z * z) * z
    else:
        argument = 0.79788456 * z * (1.0 + 0.044715 * z * z)
    if ROWWISE:
        # 0.5*z*(1+tanh(u)) == z/(1+exp(-2*u)). Fuse the activation
        # epilogue, retaining the FP32 calculation and the dtype rounding above.
        # Multiply by the reciprocal to preserve the original tiny-value path.
        result = z * (1.0 / (1.0 + tl.exp(-2.0 * argument)))
    elif _GPU_PATH:
        result = _gpu_gelu(z, argument, _HAS_NATIVE_TANH)
    else:
        # Ascend small rows retain their original non-rowwise tanh path.
        t = tl_extra_shim.tanh(argument)
        result = 0.5 * z * (1.0 + t)
    tl.store(Y + offsets, result, mask)


@libentry()
@triton.jit
def _bias_gelu_forward_kernel(
    X,
    B,
    Y,
    N,
    H: tl.constexpr,
    BIAS_MODE: tl.constexpr,
    BLOCK: tl.constexpr,
    PERSISTENT: tl.constexpr = False,
    EVEN: tl.constexpr = False,
):
    if PERSISTENT:
        # Bound physical programs to vector cores; stride over arbitrarily many
        # logical tiles. The runtime trip count avoids compile-time unrolling.
        if (BIAS_MODE == 1 and H >= 128) and not (H <= BLOCK and H & (H - 1) == 0):
            tiles = (N // H) * tl.cdiv(H, BLOCK)
        else:
            tiles = tl.cdiv(N, BLOCK)
        for tile in range(tle.program_id(0), tiles, tl.num_programs(0)):
            _bias_gelu_tile(X, B, Y, N, H, BIAS_MODE, BLOCK, tile, H >= 128, EVEN)
    else:
        _bias_gelu_tile(X, B, Y, N, H, BIAS_MODE, BLOCK, tle.program_id(0), False, EVEN)


def bias_gelu_impl(input, bias):
    """Return tanh GELU(input + bias), without gradient propagation.

    Operands must share the active accelerator and an FP16/BF16/FP32 dtype.
    PyTorch broadcasting, noncontiguous inputs and empty tensors are supported.
    The result is contiguous, with the broadcast shape and operand dtype.
    Invalid types raise TypeError; invalid dtype, device or shapes raise ValueError.
    """
    logger.debug("TRAIN BIAS_GELU_IMPL")
    metadata = []
    for name, tensor in (("input", input), ("bias", bias)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        dtype, device = tensor.dtype, tensor.device
        if dtype not in _DTYPES:
            raise ValueError(f"{name} must have FP16, BF16 or FP32 dtype")
        if device.type != runtime_device.name or device.type == "cpu":
            raise ValueError(f"{name} must be on the active accelerator")
        metadata.append((dtype, device))
    (input_dtype, input_device), (bias_dtype, bias_device) = metadata
    if input_dtype != bias_dtype or input_device != bias_device:
        raise ValueError("input and bias must have the same dtype and device")

    vector_bias = input.ndim and bias.ndim == 1 and bias.numel() == input.shape[-1]
    if vector_bias:
        x, b = input, bias
    else:
        try:
            x, b = torch.broadcast_tensors(input, bias)
        except RuntimeError:
            raise ValueError(
                "input and bias must have broadcast-compatible shapes"
            ) from None
    out = (
        input.new_empty(x.shape)
        if _IS_ASCEND
        else torch.empty(x.shape, dtype=input_dtype, device=input_device)
    )
    n = x.numel()
    if not n:
        return out

    h = x.shape[-1] if x.ndim else 1
    if vector_bias:
        b, mode = bias.contiguous(), 1
    elif bias.numel() == 1:
        b, mode = bias.contiguous(), 0
    elif b.stride(-1) == 1 and all(s == 0 for s in b.stride()[:-1]):
        # Preserve a shared bias row instead of copying its full broadcast.
        b, mode = b.reshape(-1, h)[0].contiguous(), 1
    else:
        b, mode = b.contiguous(), 2
    x = x.contiguous()
    if _IS_ASCEND:
        # FlagTree wraps its integer helpers in JIT dispatch; these launch
        # dimensions are host integers and need only ordinary Python arithmetic.
        lanes = (n + CORE_NUM - 1) // CORE_NUM
        block = min(16384, max(128, 1 << (lanes - 1).bit_length()))
        # Low-precision non-power-of-two rows can waste nearly half a large
        # tile (e.g. H=11008). A 4096-lane tile reduces masked UB work.
        if (
            input_dtype != torch.float32
            and mode == 1
            and h >= 128
            and block > 4096
            and not (h <= block and h & (h - 1) == 0)
            and ((h + block - 1) // block) * block * 8 > h * 9
        ):
            block = 4096
        if mode == 1 and h >= 128 and not (h <= block and h & (h - 1) == 0):
            tiles = (n // h) * ((h + block - 1) // block)
        else:
            tiles = (n + block - 1) // block
        grid = (min(tiles, CORE_NUM),)
    else:
        block = _BLOCK_SIZE
        if n >= 65536:
            # Share GPU tiling across vendors; FP32 favors a smaller tile.
            block = 1024 if input_dtype == torch.float32 else 2048
        grid = (triton.cdiv(n, block),)
    # A tensor on the current NPU needs no device exchange. Keep the guard for
    # cross-device calls and leave every non-Ascend launch on its original path.
    context = (
        nullcontext()
        if _IS_ASCEND and torch_device_fn.current_device() == input_device.index
        else torch_device_fn.device(input_device)
    )
    with context:
        if _IS_ASCEND:
            # Same LibEntry.run path as kernel[grid], without its temporary
            # closure/dict merge. warmup=False is required on cold JIT misses.
            _bias_gelu_forward_kernel.run(
                x,
                b,
                out,
                n,
                h,
                mode,
                grid=grid,
                warmup=False,
                BLOCK=block,
                PERSISTENT=True,
                num_warps=_NUM_WARPS,
                num_stages=1,
                EVEN=n % block == 0 and (mode != 1 or h % block == 0),
            )
        else:
            _bias_gelu_forward_kernel[grid](
                x,
                b,
                out,
                n,
                h,
                mode,
                BLOCK=block,
                PERSISTENT=_IS_ASCEND,
                num_warps=_NUM_WARPS,
                EVEN=n % block == 0,
                **({"num_stages": 1} if _IS_ASCEND else {}),
            )
    return out
