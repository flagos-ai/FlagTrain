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


"""Scale the DSA loss gradient using the live scale tensor at backward time."""

import torch

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class _DSAScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, indexer_loss, loss_scale):
        ctx.save_for_backward(indexer_loss)
        ctx.scale = loss_scale.detach() if torch.is_tensor(loss_scale) else loss_scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (loss,) = ctx.saved_tensors
        scale = ctx.scale
        if (
            torch.is_tensor(scale)
            and scale.ndim == 0
            and loss.layout == torch.strided
            and scale.layout == torch.strided
            and scale.device == loss.device
            and loss.dtype in _FLOAT_DTYPES
            and scale.dtype in _FLOAT_DTYPES
            and loss.is_contiguous()
        ):
            # Each backward must produce independent storage even if the
            # caller updates the scale and runs backward on the same graph.
            if loss.ndim == 0 and (
                loss.dtype == scale.dtype or scale.dtype == torch.float32
            ):
                grad_loss = scale.clone()
            else:
                grad_loss = scale.expand_as(loss).to(
                    dtype=torch.result_type(loss, scale),
                    copy=True,
                    memory_format=torch.contiguous_format,
                )
        else:
            if torch.is_tensor(scale):
                scale = scale.to(loss.device)
            grad_loss = torch.ones_like(loss) * (1.0 if scale is None else scale)
        return grad_output, grad_loss, None


def dsa_indexer_loss_auto_scaler(output, indexer_loss, loss_scale=None):
    return _DSAScaler.apply(output, indexer_loss, loss_scale)
