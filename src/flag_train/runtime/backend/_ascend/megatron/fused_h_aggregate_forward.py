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
"""Ascend-specialised ``fused_h_aggregate_forward``.

The generic tile is ``[4, 512]``, 1 KiB per stream, which leaves 910B4 at ~13% of
its bandwidth.  Here ``BLOCK_C`` covers the whole row, so each stream becomes one
contiguous transfer, and ``@libentry()`` takes the FlagGems launch fast path.

``(256, 32, 8, 4096)`` fp16 on 910B4 (torch 2.6.0 / torch_npu 2.6.0rc1):
15.7 ms without ``libentry``, 2.25 ms with it on the generic tile, 0.87 ms here;
eager is 1.53 ms.

The contract is unchanged: same shapes, dtype promotion, empty-axis behaviour,
inputs read-only.
"""

import functools

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import triton_lang_extension as tle

try:  # FlagGems' Ascend launch fast path; plain Triton if the image lacks it.
    from flag_gems.utils import libentry
except ImportError:  # pragma: no cover - only on images without FlagGems

    def libentry():
        def _identity(fn):
            return fn

        return _identity


_SUPPORTED = (torch.float16, torch.bfloat16, torch.float32)

# 8192 fp32 lanes is a 32 KiB accumulator; wider spills.
_MAX_BLOCK_C = 8192
_MIN_BLOCK_C = 32
# Two rows per program measured best on 910B4; _TILE_ELEMENTS bounds it further.
_TILE_ELEMENTS = 8192
_MAX_BLOCK_M = 2


@functools.lru_cache(maxsize=128)
def _config(C, sb):
    """Tile geometry for one launch: the whole row if it fits, never more rows
    than ``sb`` (``tl.arange`` needs a power of two)."""
    block_c = min(_MAX_BLOCK_C, max(_MIN_BLOCK_C, triton.next_power_of_2(max(C, 1))))
    rows = 1 << (max(sb, 1).bit_length() - 1)
    block_m = max(1, min(_MAX_BLOCK_M, _TILE_ELEMENTS // block_c, rows))
    return block_m, block_c, 4


@libentry()
@triton.jit
def _h_aggregate_forward_kernel(
    x_ptr,
    h_ptr,
    out_ptr,
    sb,
    C,
    stride_x_s,
    stride_x_n,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """One ``[BLOCK_M, BLOCK_C]`` tile of ``out``, reducing over the streams."""
    pid_m = tle.program_id(0)
    pid_c = tle.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_m = offs_m < sb
    mask = mask_m[:, None] & (offs_c[None, :] < C)

    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)
    for i in tl.static_range(N):
        x_i = tl.load(
            x_ptr + offs_m[:, None] * stride_x_s + i * stride_x_n + offs_c[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        h_i = tl.load(h_ptr + offs_m * N + i, mask=mask_m, other=0.0).to(tl.float32)
        acc += h_i[:, None] * x_i

    tl.store(
        out_ptr + offs_m[:, None] * C + offs_c[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def _check_input(tensor, name, ndim):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dim() != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {tuple(tensor.shape)}")
    if tensor.dtype not in _SUPPORTED:
        raise ValueError(f"{name} dtype {tensor.dtype} is not supported")
    if tensor.device.type == "cpu":
        raise ValueError(f"{name} must be a device tensor, not a CPU tensor")
    if tensor.device.type != runtime_device.name:
        raise ValueError(
            f"{name} must be a {runtime_device.name} tensor, got {tensor.device.type}"
        )


def fused_h_aggregate_forward(
    x: torch.Tensor,
    h_pre: torch.Tensor,
) -> torch.Tensor:
    """Weighted sum of ``n`` streams.

    Args:
        x (Tensor): ``[s, b, n, C]`` activations on the active device;
            a non-contiguous input is copied.
        h_pre (Tensor): ``[s, b, n]`` stream weights, any supported dtype.

    Returns:
        Tensor: ``[s, b, C]`` in ``torch.promote_types(x.dtype, h_pre.dtype)``.
    """
    _check_input(x, "x", 4)
    _check_input(h_pre, "h_pre", 3)
    if x.shape[:3] != h_pre.shape:
        raise ValueError(
            "x and h_pre must agree on the (s, b, n) axes, got "
            f"{tuple(x.shape)} and {tuple(h_pre.shape)}"
        )

    s, b, n, C = x.shape
    out_dtype = torch.promote_types(x.dtype, h_pre.dtype)
    out = x.new_empty((s, b, C), dtype=out_dtype)
    if s == 0 or b == 0 or C == 0:
        return out
    if n == 0:
        out.zero_()
        return out

    if not x.is_contiguous():
        x = x.contiguous()
    if not h_pre.is_contiguous():
        h_pre = h_pre.contiguous()

    sb = s * b
    block_m, block_c, num_warps = _config(C, sb)
    grid = (triton.cdiv(sb, block_m), triton.cdiv(C, block_c))

    device_index = x.device.index
    if device_index is None:
        device_index = torch_device_fn.current_device()
    guard = (
        None
        if torch_device_fn.current_device() == device_index
        else torch_device_fn.device(x.device)
    )
    if guard is None:
        _h_aggregate_forward_kernel[grid](
            x,
            h_pre,
            out,
            sb,
            C,
            # ``s`` and ``b`` are flattened away, so the row stride is the one
            # between adjacent ``b``, not between adjacent ``s``.
            x.stride(1),
            x.stride(2),
            N=n,
            BLOCK_M=block_m,
            BLOCK_C=block_c,
            num_warps=num_warps,
        )
    else:
        with guard:
            _h_aggregate_forward_kernel[grid](
                x,
                h_pre,
                out,
                sb,
                C,
                # ``s`` and ``b`` are flattened away, so the row stride is the
                # one between adjacent ``b``, not between adjacent ``s``.
                x.stride(1),
                x.stride(2),
                N=n,
                BLOCK_M=block_m,
                BLOCK_C=block_c,
                num_warps=num_warps,
            )
    return out
