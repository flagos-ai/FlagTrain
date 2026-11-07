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
"""Ascend-tuned scaled masked softmax backward.

Same contract as the generic implementation: ``dx = scale * y * (dy - sum(dy *
y))`` over the key axis, accumulated in fp32. The mask is not an operand --
masked positions have ``y == 0`` and therefore zero gradient -- so the kernel
moves three streams. As in the forward, the grid is pinned to the vector-core
count and each program walks its share of row tiles, per-launch values are
``constexpr``, and a non-power-of-two key length takes the 1-D row path. The
kernel is vector-pass bound rather than bandwidth bound (a three-stream copy
runs in 3.7 ms against 5.5 ms on the largest shape); two pipeline stages hide
the next tile's loads.
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

# The measured lanes-per-program plateau (see the forward module).
_TILE_LANES = 8192

_MAX_ROWS = 64


def _check_tensor(tensor, name):
    if tensor.ndim != 4:
        raise ValueError(f"{name} needs a 4-D tensor, got {tensor.ndim}-D")
    if tensor.dtype not in _SUPPORTED_DTYPES:
        wanted = " / ".join(str(dtype) for dtype in _SUPPORTED_DTYPES)
        raise ValueError(f"{name} needs a {wanted} tensor, got {tensor.dtype}")
    if tensor.device.type == "cpu":
        raise ValueError(f"{name} needs a device tensor, not one on the CPU")
    if tensor.device.type != runtime_device.name:
        raise ValueError(
            f"{name} needs a {runtime_device.name} tensor, got {tensor.device.type}"
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
def _scaled_masked_softmax_backward_kernel(
    dy_ptr,
    y_ptr,
    dx_ptr,
    BLOCK_N: tl.constexpr,
    ROWS: tl.constexpr,
    SK: tl.constexpr,
    SCALE: tl.constexpr,
    N_ROWS: tl.constexpr,
    NBLK: tl.constexpr,
    NPROG: tl.constexpr,
    EVEN_ROWS: tl.constexpr,
    SCALAR_ROW: tl.constexpr,
    I64: tl.constexpr,
):
    """``dx = SCALE * y * (dy - sum(dy * y))``, one row tile per walk step.

    ``SCALAR_ROW`` is the 1-D path. Programs take exact contiguous runs of
    tiles: a strided walk overshoots ``NBLK`` and would write past the tensor.
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
            dy = tl.load(dy_ptr + base + col, mask=valid, other=0.0).to(tl.float32)
            y = tl.load(y_ptr + base + col, mask=valid, other=0.0).to(tl.float32)
            dot = tl.sum(dy * y, 0)
            dx = SCALE * y * (dy - dot)
            tl.store(dx_ptr + base + col, dx.to(dx_ptr.dtype.element_ty), mask=valid)
    else:
        col = tl.arange(0, BLOCK_N)
        row_ids = tl.arange(0, ROWS)
        for idx in tl.range(first, stop):
            if I64:
                idx = idx.to(tl.int64)
            rows = idx * ROWS + row_ids
            row_ok = rows < N_ROWS
            offs = rows[:, None] * SK + col[None, :]
            if EVEN_ROWS:
                dy = tl.load(dy_ptr + offs).to(tl.float32)
                y = tl.load(y_ptr + offs).to(tl.float32)
            else:
                rm = row_ok[:, None]
                dy = tl.load(dy_ptr + offs, mask=rm, other=0.0).to(tl.float32)
                y = tl.load(y_ptr + offs, mask=rm, other=0.0).to(tl.float32)
            dot = tl.sum(dy * y, 1)
            dx = SCALE * y * (dy - dot[:, None])
            out = dx.to(dx_ptr.dtype.element_ty)
            if EVEN_ROWS:
                tl.store(dx_ptr + offs, out)
            else:
                tl.store(dx_ptr + offs, out, mask=row_ok[:, None])


@torch.compiler.disable
def scaled_masked_softmax_cuda_backward(output_grads, softmax_results, scale):
    """Fused scaled-softmax gradient, Megatron's masked-attention backward.

    Args:
        output_grads (Tensor): fp16, bf16 or fp32 upstream gradient of shape
            ``[b, np, sq, sk]``. A non-contiguous tensor is copied.
        softmax_results (Tensor): forward output, same shape and dtype.
        scale (float | Tensor): the forward scaling factor.

    Returns:
        Tensor: contiguous input gradient with the input's shape and dtype.
    """
    logger.debug("TRAIN_ASCEND SCALED_MASKED_SOFTMAX_CUDA_BACKWARD")
    _check_tensor(output_grads, "scaled masked softmax backward grad")
    _check_tensor(softmax_results, "scaled masked softmax backward results")
    if output_grads.shape != softmax_results.shape:
        raise ValueError(
            "scaled masked softmax backward needs matching shapes, got "
            f"{tuple(output_grads.shape)} and {tuple(softmax_results.shape)}"
        )
    if output_grads.dtype != softmax_results.dtype:
        raise ValueError(
            "scaled masked softmax backward needs matching dtypes, got "
            f"{output_grads.dtype} and {softmax_results.dtype}"
        )
    if output_grads.device != softmax_results.device:
        raise ValueError(
            "scaled masked softmax backward needs both operands on one device"
        )
    sk = output_grads.shape[-1]
    if sk > _MAX_KEY_LEN:
        raise ValueError(
            f"scaled masked softmax backward: key length {sk} exceeds "
            f"Megatron's fused-softmax limit of {_MAX_KEY_LEN}"
        )
    if sk == 0:
        raise ValueError("scaled masked softmax backward: key length must be positive")
    scale = _check_scale(scale)
    if output_grads.numel() == 0:
        return torch.empty_like(output_grads, memory_format=torch.contiguous_format)

    dy = output_grads.contiguous()
    y = softmax_results.contiguous()
    dx = torch.empty_like(dy)
    n_rows = dy.numel() // sk

    block_n = max(2, triton.next_power_of_2(sk))
    even_col = sk == block_n
    rows = max(1, min(_TILE_LANES // block_n, _MAX_ROWS)) if even_col else 1
    nblk = triton.cdiv(n_rows, rows)
    grid = min(nblk, CORE_NUM)

    with torch_device_fn.device(output_grads.device):
        _scaled_masked_softmax_backward_kernel[(grid,)](
            dy,
            y,
            dx,
            BLOCK_N=block_n,
            ROWS=rows,
            SK=sk,
            SCALE=scale,
            N_ROWS=n_rows,
            NBLK=nblk,
            NPROG=grid,
            EVEN_ROWS=n_rows % rows == 0,
            SCALAR_ROW=not even_col,
            I64=n_rows * sk >= _I64_THRESHOLD,
            # Two stages hide the next tile's loads (128/256 keys: 12.0 ->
            # 11.2 us and 36.1 -> 33.6 us); the large shapes are unchanged.
            num_stages=2,
        )
    return dx
