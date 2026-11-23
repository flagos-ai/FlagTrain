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


"""Sinkhorn projection and recomputed first-order backward."""

from operator import index

import torch
import triton
import triton.language as tl

# CUDA Triton ships libdevice under extra.cuda; FlagTree (Hygon/BW1000)
# only provides the backend-neutral extra.libdevice.
try:
    from triton.language.extra.cuda import libdevice
except ImportError:
    from triton.language.extra import libdevice

from flag_train.runtime import torch_device_fn

from ._fused_sinkhorn_stages import as_float32, get_staged_plan


@triton.jit
def _round_to_input(value, DTYPE: tl.constexpr):
    return value.to(DTYPE).to(tl.float32)


@triton.jit
def _softmax(logits):
    exp = libdevice.exp(logits - tl.max(logits, 1, keep_dims=True))
    return exp / tl.sum(exp, 1, keep_dims=True)


@triton.jit
def _forward(X, Y, WIDTH: tl.constexpr, STEPS: tl.constexpr, EPS: tl.constexpr):
    i = tl.arange(0, WIDTH)
    index = (
        tl.program_id(0).to(tl.int64) * WIDTH * WIDTH + i[:, None] * WIDTH + i[None, :]
    )
    dtype: tl.constexpr = X.dtype.element_ty
    softmax = _softmax(tl.load(X + index).to(tl.float32))
    matrix = softmax + EPS
    if STEPS == 1:
        col_sum = _round_to_input(tl.sum(matrix, 0, keep_dims=True), dtype)
        output = matrix / (col_sum + EPS)
    else:
        # Match the intermediate writes in the compiled no-grad graph.
        col0 = _round_to_input(tl.sum(matrix, 0, keep_dims=True), dtype) + EPS
        col_normalized = matrix / col0
        row1 = _round_to_input(tl.sum(col_normalized, 1, keep_dims=True), dtype) + EPS
        col_normalized = _round_to_input(col_normalized, dtype)
        matrix = col_normalized / row1
        col2 = _round_to_input(tl.sum(matrix, 0, keep_dims=True), dtype) + EPS
        matrix = matrix / col2
        row3 = tl.sum(matrix, 1, keep_dims=True) + EPS
        matrix = _round_to_input(matrix / row3, dtype)
        col4 = _round_to_input(tl.sum(matrix, 0, keep_dims=True), dtype) + EPS
        output = matrix / col4
    tl.store(Y + index, output)


@triton.jit
def _forward_three_steps(
    X, Y, WIDTH: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr
):
    """Masked 3-step (num_iterations=2) forward at any width.

    Verified bit-identical to the frozen inference plan: the column sums are
    rounded to the input dtype, the stored row sum is taken from the raw
    (unrounded) matrix, and the second numerator is the rounded matrix.
    """
    i = tl.arange(0, BLOCK)
    row = i[:, None]
    col = i[None, :]
    valid = (row < WIDTH) & (col < WIDTH)
    index = tl.program_id(0).to(tl.int64) * (WIDTH * WIDTH) + row * WIDTH + col
    dtype: tl.constexpr = X.dtype.element_ty
    logits = tl.load(X + index, valid, other=-float("inf")).to(tl.float32)
    exp = tl.where(
        valid, libdevice.exp(logits - tl.max(logits, 1, keep_dims=True)), 0.0
    )
    row_sum = tl.sum(exp, 1, keep_dims=True)
    softmax = exp / tl.where(row_sum > 0, row_sum, 1.0)
    matrix = tl.where(valid, softmax + EPS, 0.0)
    col0 = _round_to_input(tl.sum(matrix, 0, keep_dims=True), dtype) + EPS
    a1 = matrix / col0
    row1 = _round_to_input(tl.sum(a1, 1, keep_dims=True), dtype) + EPS
    e3 = _round_to_input(a1, dtype) / row1
    col2 = _round_to_input(tl.sum(e3, 0, keep_dims=True), dtype) + EPS
    tl.store(Y + index, e3 / col2, valid)


@triton.jit
def _division_vjp(
    grad,
    denominator,
    scaled,
    AXIS: tl.constexpr,
    DTYPE: tl.constexpr,
    ROUND_REDUCTION: tl.constexpr,
):
    correction = tl.sum(-grad * scaled, AXIS, keep_dims=True)
    if ROUND_REDUCTION:
        correction = _round_to_input(correction, DTYPE)
    return grad / denominator + correction


@triton.jit
def _backward(X, G, DX, WIDTH: tl.constexpr, EPS: tl.constexpr):
    i = tl.arange(0, WIDTH)
    index = (
        tl.program_id(0).to(tl.int64) * WIDTH * WIDTH + i[:, None] * WIDTH + i[None, :]
    )
    dtype: tl.constexpr = X.dtype.element_ty
    logits = tl.load(X + index).to(tl.float32)
    grad = tl.load(G + index).to(tl.float32)
    softmax = _softmax(logits)
    if WIDTH == 4:
        # The reduction branch sees the wide sum. The direct branch reloads
        # the sum after it has been stored at the input dtype.
        matrix = softmax + EPS
        col_sum = tl.sum(matrix, 0, keep_dims=True)
        wide_denominator = col_sum + EPS
        correction = tl.sum(
            -grad * ((matrix / wide_denominator) / wide_denominator),
            0,
            keep_dims=True,
        )
        rounded_denominator = _round_to_input(col_sum, dtype) + EPS
        grad = grad / rounded_denominator + _round_to_input(correction, dtype)
    else:
        softmax = _round_to_input(softmax, dtype)
        matrix = softmax + EPS
        col0 = tl.sum(matrix, 0, keep_dims=True) + EPS
        # Recompute the normalization denominators as in the grad-enabled graph.
        col0 = _round_to_input(col0, dtype)
        after_col0 = matrix / col0
        row1 = _round_to_input(tl.sum(after_col0, 1, keep_dims=True) + EPS, dtype)
        after_row1 = after_col0 / row1
        col2 = _round_to_input(tl.sum(after_row1, 0, keep_dims=True) + EPS, dtype)
        after_col2 = after_row1 / col2
        row3 = _round_to_input(tl.sum(after_col2, 1, keep_dims=True) + EPS, dtype)
        after_col2 = _round_to_input(after_col2, dtype)
        after_row3 = after_col2 / row3
        col4 = _round_to_input(tl.sum(after_row3, 0, keep_dims=True) + EPS, dtype)

        grad = _division_vjp(grad, col4, (after_row3 / col4) / col4, 0, dtype, True)
        grad = _division_vjp(grad, row3, after_row3 / row3, 1, dtype, False)
        grad = _round_to_input(grad, dtype)
        grad = _division_vjp(grad, col2, after_col2 / col2, 0, dtype, True)
        grad = _division_vjp(grad, row1, after_row1 / row1, 1, dtype, False)
        grad = _round_to_input(grad, dtype)

        grad = _division_vjp(grad, col0, (matrix / col0) / col0, 0, dtype, True)
    product = grad * softmax
    row_dot = tl.sum(product, 1, keep_dims=True)
    tl.store(DX + index, tl.fma(-softmax, row_dot, product))


@triton.jit
def _project_tile(
    logits, mask, EPS: tl.constexpr, STEPS: tl.constexpr, NEED_GRAD: tl.constexpr
):
    exp = tl.where(mask, libdevice.exp(logits - tl.max(logits, 1, keep_dims=True)), 0.0)
    row_sum = tl.sum(exp, 1, keep_dims=True)
    softmax = exp / tl.where(row_sum > 0, row_sum, 1.0)
    matrix = tl.where(mask, softmax + EPS, 0.0)
    denominators = ()
    numerators = ()
    for step in tl.static_range(STEPS):
        denominator = tl.sum(matrix, step % 2, keep_dims=True) + EPS
        if NEED_GRAD:
            denominators += (denominator,)
            numerators += (matrix,)
        matrix = matrix / denominator
    return matrix, softmax, denominators, numerators


@triton.jit
def _general(
    X,
    G,
    Y,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    STEPS: tl.constexpr,
    EPS: tl.constexpr,
    BACKWARD: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    row, col = i[:, None], i[None, :]
    valid = (row < WIDTH) & (col < WIDTH)
    index = tl.program_id(0).to(tl.int64) * WIDTH * WIDTH + row * WIDTH + col
    logits = tl.load(X + index, valid, other=-float("inf")).to(tl.float32)
    matrix, softmax, denominators, numerators = _project_tile(
        logits, valid, EPS, STEPS, BACKWARD
    )
    if BACKWARD:
        gradient = tl.load(G + index, valid, other=0.0).to(tl.float32)
        for step in tl.static_range(STEPS - 1, -1, -1):
            denominator = denominators[step]
            numerator = numerators[step]
            correction = tl.sum(
                -gradient * ((numerator / denominator) / denominator),
                step % 2,
                keep_dims=True,
            )
            gradient = tl.where(valid, gradient / denominator + correction, 0.0)
        product = gradient * softmax
        matrix = tl.fma(-softmax, tl.sum(product, 1, keep_dims=True), product)
    tl.store(Y + index, matrix, valid)


def _specialized(width, num_iterations):
    return (width, num_iterations) in ((4, 1), (8, 3))


class _Sinkhorn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, num_iterations, eps):
        width = logits.shape[-1]
        ctx.save_for_backward(logits)
        ctx.num_iterations = num_iterations
        ctx.eps = eps
        result = torch.empty_like(logits)
        with torch_device_fn.device(logits.device):
            if result.numel():
                matrices = result.numel() // (width * width)
                if _specialized(width, num_iterations):
                    _forward[(matrices,)](
                        logits,
                        result,
                        width,
                        2 * num_iterations - 1,
                        eps,
                        num_warps=1,
                        enable_fp_fusion=False,
                    )
                else:
                    _general[(matrices,)](
                        logits,
                        logits,
                        result,
                        width,
                        triton.next_power_of_2(width),
                        2 * num_iterations - 1,
                        eps,
                        False,
                        num_warps=1,
                        enable_fp_fusion=False,
                    )
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        (logits,) = ctx.saved_tensors
        width = logits.shape[-1]
        grad_input = torch.empty_like(logits)
        with torch_device_fn.device(logits.device):
            if logits.numel():
                matrices = logits.numel() // (width * width)
                gradient = grad_output.contiguous()
                if _specialized(width, ctx.num_iterations):
                    _backward[(matrices,)](
                        logits,
                        gradient,
                        grad_input,
                        width,
                        ctx.eps,
                        num_warps=1 if width == 4 else 2,
                        enable_fp_fusion=False,
                    )
                else:
                    _general[(matrices,)](
                        logits,
                        gradient,
                        grad_input,
                        width,
                        triton.next_power_of_2(width),
                        2 * ctx.num_iterations - 1,
                        ctx.eps,
                        True,
                        num_warps=1,
                        enable_fp_fusion=False,
                    )
        return grad_input, None, None


class _StagedSinkhorn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, num_iterations, eps):
        ctx.save_for_backward(logits)
        ctx.num_iterations = num_iterations
        ctx.eps = eps
        if logits.numel() == 0:
            return torch.empty_like(logits)
        width = logits.shape[-1]
        with torch_device_fn.device(logits.device):
            if num_iterations == 3:
                # Keep the fused forward while recomputing the staged gradient.
                result = torch.empty_like(logits)
                matrices = result.numel() // (width * width)
                _forward[(matrices,)](
                    logits,
                    result,
                    width,
                    2 * num_iterations - 1,
                    eps,
                    num_warps=1,
                    enable_fp_fusion=False,
                )
                return result
            if num_iterations == 2:
                # Same treatment for the 3-step chain, also verified bit-identical.
                result = torch.empty_like(logits)
                matrices = result.numel() // (width * width)
                _forward_three_steps[(matrices,)](
                    logits,
                    result,
                    width,
                    triton.next_power_of_2(width),
                    eps,
                    num_warps=1,
                    enable_fp_fusion=False,
                )
                return result
            inference, _, _ = get_staged_plan(num_iterations)
            return inference(
                (*logits.shape, logits, as_float32(eps)), logits.device, logits.dtype
            )[0]

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        (logits,) = ctx.saved_tensors
        if logits.numel() == 0:
            return torch.empty_like(logits), None, None
        with torch_device_fn.device(logits.device):
            _, forward, backward = get_staged_plan(ctx.num_iterations)
            shape = tuple(logits.shape)
            intermediate = forward(
                (*shape, logits, ctx.eps), logits.device, logits.dtype
            )
            args = (*shape, *intermediate[1 : -len(shape)], grad_output.contiguous())
            grad = backward(args, logits.device, logits.dtype)[len(shape)]
        return grad, None, None


# These compiled graphs have additional low-precision rounding points.
_STAGED_CONFIGS = {(3, 5, 2), (3, 4, 3), (4, 4, 20), (4, 8, 20)}


def _project_torch(logits, num_iterations, eps):
    matrix = logits.softmax(dim=-1) + eps
    for step in range(2 * num_iterations - 1):
        axis = -2 if step % 2 == 0 else -1
        matrix = matrix / (matrix.sum(dim=axis, keepdim=True) + eps)
    return matrix


class _TorchSinkhorn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, num_iterations, eps):
        ctx.save_for_backward(logits)
        ctx.num_iterations = num_iterations
        ctx.eps = eps
        return _project_torch(logits, num_iterations, eps)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        (logits,) = ctx.saved_tensors
        with torch.enable_grad():
            source = logits.detach().requires_grad_(True)
            output = _project_torch(source, ctx.num_iterations, ctx.eps)
            (grad,) = torch.autograd.grad(output, source, grad_output)
        return grad, None, None


def _prepare_sinkhorn(input_logits, num_iterations, eps):
    if input_logits.ndim < 2 or input_logits.shape[-2] != input_logits.shape[-1]:
        raise ValueError("fused_sinkhorn requires square matrices")
    num_iterations = index(num_iterations)
    width = input_logits.shape[-1]
    if width < 1 or num_iterations < 1:
        raise ValueError(
            "fused_sinkhorn requires nonempty matrices and positive iterations"
        )
    if input_logits.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("fused_sinkhorn supports FP32, FP16 and BF16")
    eps = float(eps)
    if not 0 < eps < float("inf"):
        raise ValueError("fused_sinkhorn requires a finite positive eps")
    return input_logits.contiguous(), num_iterations, eps


def _fits_triton_tile(logits, num_iterations):
    # The fused kernels keep one square tile and its reductions in a program.
    return logits.shape[-1] <= 32 and num_iterations <= 32


def fused_sinkhorn(input_logits, num_iterations, eps=1e-6):
    """Compute the Sinkhorn projection and its recomputed first-order gradient."""
    logits, num_iterations, eps = _prepare_sinkhorn(input_logits, num_iterations, eps)
    if not _fits_triton_tile(logits, num_iterations):
        return _TorchSinkhorn.apply(logits, num_iterations, eps)
    if (
        logits.dtype != torch.float32
        and (logits.ndim, logits.shape[-1], num_iterations) in _STAGED_CONFIGS
    ):
        if logits.numel() >= 2**31:
            return _TorchSinkhorn.apply(logits, num_iterations, eps)
        return _StagedSinkhorn.apply(logits, num_iterations, eps)
    return _Sinkhorn.apply(logits, num_iterations, eps)
