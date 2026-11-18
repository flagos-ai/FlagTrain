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


"""Ascend Triton Sinkhorn projection and first-order backward."""

import torch
import triton
import triton.language as tl

from flag_train.megatron.fused_sinkhorn import (
    _fits_triton_tile,
    _prepare_sinkhorn,
    _TorchSinkhorn,
)
from flag_train.runtime import torch_device_fn

_BACKWARD_WORKSPACE_BYTES = 32 * 1024**2


@triton.jit
def _rounded(value, DTYPE: tl.constexpr):
    return value.to(DTYPE).to(tl.float32)


@triton.jit
def _forward(
    LOGITS,
    OUT,
    SOFTMAX,
    STATES,
    DIVISORS,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
    ITERATIONS: tl.constexpr,
    EPS: tl.constexpr,
    SAVE: tl.constexpr,
):
    matrix_id = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, TILE)
    mask = (i[:, None] < WIDTH) & (i[None, :] < WIDTH)
    elem = matrix_id * WIDTH * WIDTH + i[:, None] * WIDTH + i[None, :]
    dt: tl.constexpr = LOGITS.dtype.element_ty
    logits = tl.load(LOGITS + elem, mask, float("-inf")).to(tl.float32)
    logits = tl.where(i[:, None] < WIDTH, logits, 0.0)
    centered = logits - tl.max(logits, 1, keep_dims=True)
    if dt == tl.float16:
        # Ascend Torch softmax rounds the max subtraction before exponentiation.
        centered = _rounded(centered, dt)
    exps = tl.where(mask, tl.exp(centered), 0.0)
    total = tl.sum(exps, 1, keep_dims=True)
    normalization = tl.where(i[:, None] < WIDTH, total, 1.0)
    softmax = _rounded(tl.div_rn(exps, normalization), dt)
    if SAVE:
        tl.store(SOFTMAX + elem, softmax, mask)
    matrix = _rounded(softmax + EPS, dt)
    matrix = tl.where(mask, matrix, 0.0)
    steps: tl.constexpr = ITERATIONS * 2 - 1
    state_base = (matrix_id * steps * WIDTH * WIDTH) + i[:, None] * WIDTH + i[None, :]
    for step in range(steps):
        if step % 2 == 0:
            denom = _rounded(
                _rounded(
                    tl.broadcast_to(tl.sum(matrix, 0, keep_dims=True), (TILE, TILE)), dt
                )
                + EPS,
                dt,
            )
        else:
            denom = _rounded(
                _rounded(
                    tl.broadcast_to(tl.sum(matrix, 1, keep_dims=True), (TILE, TILE)), dt
                )
                + EPS,
                dt,
            )
        if SAVE:
            offset = state_base + step * WIDTH * WIDTH
            tl.store(STATES + offset, matrix, mask)
            tl.store(DIVISORS + offset, denom, mask)
        matrix = tl.where(mask, _rounded(tl.div_rn(matrix, denom), dt), 0.0)
    if not SAVE:
        tl.store(OUT + elem, matrix, mask)


@triton.jit
def _backward(
    GRAD,
    SOFTMAX,
    STATES,
    DIVISORS,
    GRAD_INPUT,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    matrix_id = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, TILE)
    mask = (i[:, None] < WIDTH) & (i[None, :] < WIDTH)
    elem = matrix_id * WIDTH * WIDTH + i[:, None] * WIDTH + i[None, :]
    dt: tl.constexpr = GRAD.dtype.element_ty
    steps: tl.constexpr = ITERATIONS * 2 - 1
    state_base = (matrix_id * steps * WIDTH * WIDTH) + i[:, None] * WIDTH + i[None, :]
    gradient = tl.load(GRAD + elem, mask, 0).to(tl.float32)
    for index in range(steps):
        step = steps - index - 1
        offset = state_base + step * WIDTH * WIDTH
        matrix = tl.load(STATES + offset, mask, 0).to(tl.float32)
        denom = tl.load(DIVISORS + offset, mask, 1).to(tl.float32)
        direct = _rounded(tl.div_rn(gradient, denom), dt)
        quotient = _rounded(tl.div_rn(matrix, denom), dt)
        quotient = _rounded(tl.div_rn(quotient, denom), dt)
        divisor_gradient = _rounded(-gradient * quotient, dt)
        if step % 2 == 0:
            correction = _rounded(
                tl.broadcast_to(
                    tl.sum(divisor_gradient, 0, keep_dims=True), (TILE, TILE)
                ),
                dt,
            )
        else:
            correction = _rounded(
                tl.broadcast_to(
                    tl.sum(divisor_gradient, 1, keep_dims=True), (TILE, TILE)
                ),
                dt,
            )
        gradient = tl.where(mask, _rounded(direct + correction, dt), 0.0)
    softmax = tl.load(SOFTMAX + elem, mask, 0).to(tl.float32)
    inner = tl.sum(gradient * softmax, 1, keep_dims=True)
    tl.store(GRAD_INPUT + elem, softmax * (gradient - inner), mask)


class _Sinkhorn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, num_iterations, eps):
        width = logits.shape[-1]
        count = logits.numel() // (width * width)
        result = torch.empty_like(logits, memory_format=torch.contiguous_format)
        ctx.save_for_backward(logits)
        ctx.num_iterations = num_iterations
        ctx.eps = eps
        if count:
            with torch_device_fn.device(logits.device):
                _forward[(count,)](
                    logits,
                    result,
                    result,
                    result,
                    result,
                    width,
                    triton.next_power_of_2(width),
                    num_iterations,
                    eps,
                    False,
                    enable_fp_fusion=False,
                )
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        (logits,) = ctx.saved_tensors
        width = logits.shape[-1]
        count = logits.numel() // (width * width)
        result = torch.empty_like(logits, memory_format=torch.contiguous_format)
        if not count:
            return result, None, None

        steps = ctx.num_iterations * 2 - 1
        per_matrix_bytes = (1 + 2 * steps) * width * width * logits.element_size()
        batch = min(count, max(1, _BACKWARD_WORKSPACE_BYTES // per_matrix_bytes))
        matrices = logits.view(count, width, width)
        gradient = grad_output.contiguous().view_as(matrices)
        out = result.view_as(matrices)
        softmax = torch.empty_like(matrices[:batch])
        states = logits.new_empty((batch, steps, width, width))
        divisors = torch.empty_like(states)
        tile = triton.next_power_of_2(width)
        # Recompute stage values into a bounded workspace reused by every chunk.
        with torch_device_fn.device(logits.device):
            for start in range(0, count, batch):
                stop = min(start + batch, count)
                grid = (stop - start,)
                _forward[grid](
                    matrices[start:stop],
                    softmax,
                    softmax,
                    states,
                    divisors,
                    width,
                    tile,
                    ctx.num_iterations,
                    ctx.eps,
                    True,
                    enable_fp_fusion=False,
                )
                _backward[grid](
                    gradient[start:stop],
                    softmax,
                    states,
                    divisors,
                    out[start:stop],
                    width,
                    tile,
                    ctx.num_iterations,
                    enable_fp_fusion=False,
                )
        return result, None, None


def fused_sinkhorn(input_logits, num_iterations, eps=1e-6):
    logits, num_iterations, eps = _prepare_sinkhorn(input_logits, num_iterations, eps)
    if not _fits_triton_tile(logits, num_iterations):
        return _TorchSinkhorn.apply(logits, num_iterations, eps)
    return _Sinkhorn.apply(logits, num_iterations, eps)
