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

"""Transformer Engine GroupedLinear evaluation with discrete parameters.

Upstream reference: Transformer Engine 4f5c08e.
"""

import torch

OP_NAME = "grouped_linear"
DTYPES = (torch.float32, torch.bfloat16, torch.float16)
IS_INPLACE = False

_WORKLOADS = (
    (64, 128, "no-bias"),
    (1024, 256, "biased"),
    (64, 64, "biased"),
    (128, 128, "no-bias"),
    (256, 64, "biased"),
    (256, 256, "no-bias"),
    (512, 128, "biased"),
    (512, 256, "no-bias"),
)


def grid():
    return [
        {"rows": rows, "width": width, "variant": variant}
        for rows, width, variant in _WORKLOADS
    ]


def key_shape(binding):
    return [binding["rows"], binding["width"]]


def native():
    _grouped_linear()
    return _forward


def _grouped_linear():
    from transformer_engine.pytorch.module.grouped_linear import GroupedLinear

    return GroupedLinear


@torch.no_grad()
def _forward(module, inp, splits):
    return module(inp, splits)


def build_inputs(binding, dtype, device):
    n, h = binding["rows"], binding["width"]
    mode = binding["variant"]
    rng = torch.Generator().manual_seed(4234321)

    def rand(*shape, as_type=dtype):
        return torch.randn(shape, generator=rng).to(device, as_type)

    inp = rand(n, h)
    weights = rand(3, h // 2, h)
    biases = rand(3, h // 2) if mode == "biased" else None
    module = _grouped_linear()(
        num_gemms=3,
        in_features=h,
        out_features=h // 2,
        bias=biases is not None,
        params_dtype=dtype,
        device=device,
        use_grouped_tensor=False,
    ).eval()
    with torch.no_grad():
        for idx, weight in enumerate(weights):
            getattr(module, f"weight{idx}").copy_(weight)
            if biases is not None:
                getattr(module, f"bias{idx}").copy_(biases[idx])
    return (module, inp, [n // 4, n // 4, n // 2]), {}


def _tensor(shape, dtype):
    return {"shape": shape, "dtype": str(dtype)}


def config(binding, dtype):
    n, h = binding["rows"], binding["width"]
    mode = binding["variant"]
    inputs = {
        "inp": _tensor([n, h], dtype),
        "weights": _tensor([3, h // 2, h], dtype),
    }
    if mode == "biased":
        inputs["biases"] = _tensor([3, h // 2], dtype)
    return {"inputs": inputs, "m_splits": [n // 4, n // 4, n // 2]}
