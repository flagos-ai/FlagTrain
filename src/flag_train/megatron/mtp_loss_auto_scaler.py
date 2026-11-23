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


"""Per-graph MTP auxiliary loss scaling."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fill_grad(output, scale, size, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(output + offsets, tl.load(scale), offsets < size)


class _MTPLossAutoScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, mtp_loss, loss_scale):
        ctx.loss_shape = mtp_loss.shape
        ctx.loss_dtype = mtp_loss.dtype
        # Read the live scale in backward so callers can update it in place.
        ctx.loss_scale = loss_scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.needs_input_grad[1]:
            scale = ctx.loss_scale
            grad_loss = torch.empty(
                ctx.loss_shape, dtype=ctx.loss_dtype, device=scale.device
            )
            size = grad_loss.numel()
            if size:
                block = min(triton.next_power_of_2(size), 1024)
                _fill_grad[(triton.cdiv(size, block),)](grad_loss, scale, size, block)
        return grad_output, grad_loss, None


def mtp_loss_auto_scaler(output, mtp_loss, loss_scale):
    """Pass output through and scale the auxiliary loss gradient in backward."""
    if loss_scale.numel() != 1 or loss_scale.device != mtp_loss.device:
        raise ValueError("loss_scale must be a single value on the mtp_loss device")
    return _MTPLossAutoScaler.apply(output, mtp_loss, loss_scale)
