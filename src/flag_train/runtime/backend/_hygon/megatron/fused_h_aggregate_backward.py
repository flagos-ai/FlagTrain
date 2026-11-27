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
"""Hygon fused n-stream weighted aggregation (backward).

The generic implementation is ``flag_train.megatron.fused_h_aggregate_backward``
and is what every backend without an override runs. The arithmetic, the tile
geometry and the validation are that module's, unchanged -- ``_check_input`` and
``_config`` are imported from it rather than rewritten. One thing differs, and
it is a code-generation workaround, not a tuning choice:

**The fp32 product of ``grad_x`` is pinned before the narrowing store.** The
generic kernel computes

    grad_x[s, b, i, c] = grad_out[s, b, c] * h_pre[s, b, i]

in fp32 and stores it into ``x``'s dtype. The reference -- Megatron's autograd
derivative, and the generic kernel's documented arithmetic -- rounds the product
to fp32 and then converts that value. On the HCU backend LLVM contracts
``fptrunc(fmul)`` into one mixed-precision FMA (``v_fma_mixlo_f16``), which
rounds the *exact* product straight to the output dtype. The two agree except on
the rare products that land on an fp16/bf16 midpoint after the fp32 rounding,
where they differ by one ULP; the mixed-dtype test, which compares the mixed
call bit for bit against the promoted-dtype call, catches it (measured: 815 of
16.7M elements at ``(32, 32, 4, 4096)``, fp16 activations over fp32 weights).

The kernel below multiplies the product by a runtime ``1.0`` before the store.
That multiply is exact, so no value changes; it only forces the fp32 product to
be materialized (a ``v_mul_f32``) instead of being folded into the conversion,
so the conversion rounds the value the reference rounds. The ``1.0`` is a kernel
argument rather than a literal so the compiler cannot fold it away, and only the
narrow-output branch takes it -- a float32 store has no conversion to fuse.
"""

import logging
from contextlib import nullcontext
from typing import Tuple

import torch
import triton
import triton.language as tl

from flag_train.megatron.fused_h_aggregate_backward import _check_input, _config
from flag_train.runtime import torch_device_fn
from flag_train.utils import triton_lang_extension as tle

__all__ = ["fused_h_aggregate_backward"]

logger = logging.getLogger(__name__)


@triton.jit
def _h_aggregate_backward_kernel(
    grad_out_ptr,
    x_ptr,
    h_ptr,
    grad_x_ptr,
    grad_h_ptr,
    sb,
    C,
    stride_x_s,
    stride_x_n,
    one,
    N: tl.constexpr,
    N_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """One row block for all of ``C``: writes ``grad_x`` and ``grad_h``.

    ``N_PAD`` is ``N`` rounded up to a power of two, which is what ``tl.arange``
    requires; the lanes above ``N`` are masked out of every load and store, so
    the padding costs nothing but the registers it occupies. ``stride_x_s`` is
    the row stride of the flattened ``[s*b, n, C]`` view and ``stride_x_n`` the
    distance between streams.

    ``one`` is a runtime ``1.0`` and pins the rounding of ``grad_x``: see the
    module docstring. Multiplying by it is exact, so the value is unchanged; the
    multiply only keeps ``fptrunc(fmul(go, h))`` from being contracted into a
    single-rounding mixed-precision FMA by forcing the fp32 product to be
    materialized first.

    ``grad_h`` is reduced per iteration and folded into a ``[BLOCK_M, N_PAD]``
    accumulator, so the loop carries no ``[BLOCK_M, N_PAD, BLOCK_C]`` fp32 tile.
    Holding one and reducing once at the end instead removes the cross-lane
    reduction from the loop's critical path -- and is what a small shape wants,
    where the serial chain *is* the kernel's cost -- but it costs 16 KB of
    registers per program and measured 2-9% slower at every shape large enough
    to be bandwidth-bound. Those are the shapes a training step runs.
    """
    pid_m = tle.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < sb
    offs_n = tl.arange(0, N_PAD)
    mask_n = offs_n < N

    h = tl.load(
        h_ptr + offs_m[:, None] * N + offs_n[None, :],
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)

    grad_h = tl.zeros((BLOCK_M, N_PAD), dtype=tl.float32)
    for c_start in range(0, C, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        mask_mc = mask_m[:, None] & mask_c[None, :]
        grad_out = tl.load(
            grad_out_ptr + offs_m[:, None] * C + offs_c[None, :],
            mask=mask_mc,
            other=0.0,
        ).to(tl.float32)

        mask_x = mask_m[:, None, None] & mask_n[None, :, None] & mask_c[None, None, :]
        x = tl.load(
            x_ptr
            + offs_m[:, None, None] * stride_x_s
            + offs_n[None, :, None] * stride_x_n
            + offs_c[None, None, :],
            mask=mask_x,
            other=0.0,
        ).to(tl.float32)

        grad_x = grad_out[:, None, :] * h[:, :, None]
        if grad_x_ptr.dtype.element_ty != tl.float32:
            # See ``one``: keep the fp32 product live so the narrowing store
            # rounds it, not the exact product.
            grad_x = grad_x * one
        tl.store(
            grad_x_ptr
            + offs_m[:, None, None] * stride_x_s
            + offs_n[None, :, None] * stride_x_n
            + offs_c[None, None, :],
            grad_x.to(grad_x_ptr.dtype.element_ty),
            mask=mask_x,
        )
        grad_h += tl.sum(grad_out[:, None, :] * x, axis=2)

    tl.store(
        grad_h_ptr + offs_m[:, None] * N + offs_n[None, :],
        grad_h.to(grad_h_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def fused_h_aggregate_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    h_pre: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gradients of ``fused_h_aggregate_forward`` with respect to ``x`` and ``h_pre``.

    Args:
        grad_output (Tensor): ``[s, b, C]`` gradient of the aggregated output.
        x (Tensor): ``[s, b, n, C]`` stream activations the forward read.
        h_pre (Tensor): ``[s, b, n]`` stream weights the forward read.

    Returns:
        Tuple[Tensor, Tensor]: ``grad_x`` of shape ``[s, b, n, C]`` in ``x``'s
        dtype and ``grad_h`` of shape ``[s, b, n]`` in ``h_pre``'s dtype.
    """
    logger.debug("TRAIN FUSED_H_AGGREGATE_BACKWARD")
    _check_input(grad_output, "grad_output", 3)
    _check_input(x, "x", 4)
    _check_input(h_pre, "h_pre", 3)
    if x.shape[:3] != h_pre.shape:
        raise ValueError(
            "x and h_pre must agree on the (s, b, n) axes, got "
            f"{tuple(x.shape)} and {tuple(h_pre.shape)}"
        )

    s, b, n, C = x.shape
    if grad_output.shape != (s, b, C):
        raise ValueError(
            f"grad_output must have shape {(s, b, C)}, got {tuple(grad_output.shape)}"
        )

    grad_x = x.new_empty((s, b, n, C))
    grad_h = h_pre.new_empty((s, b, n))
    if s == 0 or b == 0 or n == 0:
        return grad_x, grad_h
    if C == 0:
        # The reduction is over an empty axis: zero, not whatever the allocator
        # handed back.
        grad_h.zero_()
        return grad_x, grad_h

    if not x.is_contiguous():
        x = x.contiguous()
    if not h_pre.is_contiguous():
        h_pre = h_pre.contiguous()
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()

    sb = s * b
    n_pad = triton.next_power_of_2(n)
    block_m, block_c, num_warps, num_stages = _config(C, n_pad, sb)
    grid = (triton.cdiv(sb, block_m),)

    device_index = x.device.index
    if device_index is None:
        device_index = torch_device_fn.current_device()
    guard = (
        nullcontext()
        if torch_device_fn.current_device() == device_index
        else torch_device_fn.device(x.device)
    )

    with guard:
        _h_aggregate_backward_kernel[grid](
            grad_output,
            x,
            h_pre,
            grad_x,
            grad_h,
            sb,
            C,
            # ``s`` and ``b`` are flattened away, so the row stride is the one
            # between adjacent ``b``, not between adjacent ``s``.
            x.stride(1),
            x.stride(2),
            # A runtime ``1.0``; see the module docstring. It must be an
            # argument rather than a constant so the compiler cannot fold the
            # multiply away and recreate the single-rounding fusion.
            1.0,
            N=n,
            N_PAD=n_pad,
            BLOCK_M=block_m,
            BLOCK_C=block_c,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return grad_x, grad_h
