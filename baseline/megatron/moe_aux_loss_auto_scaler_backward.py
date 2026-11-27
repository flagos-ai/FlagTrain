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

"""Megatron-LM explicit MoEAuxLossAutoScaler backward.

Upstream reference: Megatron-LM c71687a.
"""

from types import SimpleNamespace

import torch

OP_NAME = "moe_aux_loss_auto_scaler_backward"
DTYPES = (torch.float32, torch.bfloat16, torch.float16)
IS_INPLACE = False

_WORKLOADS = (
    (17, 64, "default"),
    (1024, 256, "scale"),
    (1, 32, "default"),
    (32, 128, "scale"),
    (128, 128, "default"),
    (256, 256, "scale"),
    (512, 512, "default"),
    (1024, 1024, "scale"),
)


def grid():
    return [
        {"rows": rows, "width": width, "variant": variant}
        for rows, width, variant in _WORKLOADS
    ]


def _loss_scale(binding):
    return 1.0 if binding["variant"] == "default" else 2.5


def key_shape(binding):
    return [binding["rows"], binding["width"]]


def native():
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    def backward(ctx, grad_output, scale):
        previous_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
        # Bind this call's scale without mutating the previous shared buffer.
        MoEAuxLossAutoScaler.main_loss_backward_scale = scale.detach()
        try:
            return MoEAuxLossAutoScaler.backward(ctx, grad_output)
        finally:
            MoEAuxLossAutoScaler.main_loss_backward_scale = previous_scale

    return backward


def build_inputs(binding, dtype, device):
    n, h = binding["rows"], binding["width"]
    rng = torch.Generator().manual_seed(4234321)

    def rand(*shape, as_type=dtype):
        return torch.randn(shape, generator=rng).to(device, as_type)

    ctx = SimpleNamespace(saved_tensors=(rand(),))
    scale = torch.tensor(_loss_scale(binding), device=device)
    return (ctx, rand(n, h), scale), {}


def _tensor(shape, dtype):
    return {"shape": shape, "dtype": str(dtype)}


def config(binding, dtype):
    n, h = binding["rows"], binding["width"]
    inputs = {
        "grad_output": _tensor([n, h], dtype),
        "aux_loss": _tensor([], dtype),
        "loss_scale": _tensor([], torch.float32),
    }
    return {"inputs": inputs, "loss_scale": _loss_scale(binding), "stage": "backward"}
