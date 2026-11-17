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


"""Explicit per-graph Megatron MTP loss scaling for FlagOS requirement 6471."""

import torch

OP_NAME = "mtp_loss_auto_scaler"
OVERRIDE_KEY = f"megatron.{OP_NAME}"
SHAPES = ((2, 17), (128, 256), (1, 1), (257, 127), (128, 1024), (2, 256, 4096))
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
LOSS_SCALE = 0.375
# Keep output shapes unique across scales for shape-keyed --base-data.
SCALE_VARIANTS = (((3, 19), 0.0), ((129, 257), 2.0))
WORKLOADS = tuple((shape, LOSS_SCALE) for shape in SHAPES) + SCALE_VARIANTS


def make_inputs(shape, dtype, device, scale=LOSS_SCALE):
    return (
        torch.randn(shape, dtype=dtype, device=device),
        torch.tensor(2.0, dtype=torch.float32, device=device),
        torch.tensor(scale, dtype=torch.float32, device=device),
    )


class _MTPLossAutoScalerReference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, mtp_loss, loss_scale):
        ctx.save_for_backward(mtp_loss)
        # A mutable scale must avoid saved-tensor version checking.
        ctx.loss_scale = loss_scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (mtp_loss,) = ctx.saved_tensors
        return grad_output, torch.ones_like(mtp_loss) * ctx.loss_scale, None


def reference(output, mtp_loss, loss_scale):
    return _MTPLossAutoScalerReference.apply(output, mtp_loss, loss_scale)
