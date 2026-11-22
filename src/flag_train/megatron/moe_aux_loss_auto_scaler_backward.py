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


"""Megatron auxiliary-loss backward with a single copy for tensor scalars."""

import torch

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def moe_aux_loss_auto_scaler_backward(grad_output, aux_loss, loss_scale=None):
    scale = 1.0 if loss_scale is None else loss_scale
    if (
        torch.is_tensor(scale)
        and scale.ndim == 0
        and aux_loss.is_cuda
        and scale.device == aux_loss.device
        and aux_loss.dtype in _FLOAT_DTYPES
        and scale.dtype in _FLOAT_DTYPES
        and (not scale.requires_grad or not torch.is_grad_enabled())
        and aux_loss.is_contiguous()
    ):
        if aux_loss.ndim == 0 and (
            aux_loss.dtype == scale.dtype or scale.dtype == torch.float32
        ):
            grad_aux = scale.clone()
        else:
            # Determine promotion before expanding the scalar. A rank-one
            # tensor and a zero-dimensional tensor have different Torch rules.
            grad_aux = scale.expand_as(aux_loss).to(
                dtype=torch.result_type(aux_loss, scale),
                copy=True,
                memory_format=torch.contiguous_format,
            )
    else:
        grad_aux = torch.ones_like(aux_loss) * scale
    return grad_output, grad_aux
