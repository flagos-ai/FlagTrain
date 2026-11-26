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


import torch

from .moe_aux_loss_auto_scaler_backward import moe_aux_loss_auto_scaler_backward


class _MoEAuxLossAutoScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, aux_loss, loss_scale):
        ctx.save_for_backward(aux_loss)
        # Read the live scale when backward runs.
        ctx.loss_scale = loss_scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_aux = None
        if ctx.needs_input_grad[1]:
            (aux_loss,) = ctx.saved_tensors
            grad_output, grad_aux = moe_aux_loss_auto_scaler_backward(
                grad_output, aux_loss, ctx.loss_scale
            )
        return grad_output, grad_aux, None


def moe_aux_loss_auto_scaler(output, aux_loss, loss_scale=None):
    """Pass output through and scale the auxiliary loss gradient in backward.

    The scale defaults to one. A tensor scale must hold a single value on the
    auxiliary loss device and may be updated in place before backward.
    """
    if torch.is_tensor(loss_scale) and (
        loss_scale.numel() != 1 or loss_scale.device != aux_loss.device
    ):
        raise ValueError("loss_scale must be a single value on the aux_loss device")
    return _MoEAuxLossAutoScaler.apply(output, aux_loss, loss_scale)
