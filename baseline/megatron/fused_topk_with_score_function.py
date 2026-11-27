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

"""Transformer Engine fused top-k scores and routing-map forward.

Upstream reference: Transformer Engine 4f5c08e.
"""

import torch

OP_NAME = "fused_topk_with_score_function"
DTYPES = (torch.float32, torch.bfloat16, torch.float16)
IS_INPLACE = False

_WORKLOADS = (
    (19, 64, "softmax"),
    (512, 64, "sigmoid-grouped"),
    (8, 16, "softmax"),
    (32, 32, "sigmoid-grouped"),
    (64, 128, "softmax"),
    (128, 32, "softmax"),
    (256, 128, "sigmoid-grouped"),
    (512, 128, "softmax"),
)


def grid():
    return [
        {"rows": rows, "width": width, "variant": variant}
        for rows, width, variant in _WORKLOADS
    ]


def key_shape(binding):
    return [binding["rows"], binding["width"]]


def native():
    from transformer_engine.pytorch.router import fused_topk_with_score_function

    return fused_topk_with_score_function


def build_inputs(binding, dtype, device):
    n, h = binding["rows"], binding["width"]
    mode = binding["variant"]
    rng = torch.Generator().manual_seed(4234321)

    def rand(*shape, as_type=dtype):
        return torch.randn(shape, generator=rng).to(device, as_type)

    grouped = mode == "sigmoid-grouped"
    logits = (
        rand(n, h, as_type=torch.float32) * 0.07 + torch.arange(h, device=device) * 0.13
    ).to(dtype)
    return (
        logits,
        4,
        False,
        4 if grouped else None,
        2 if grouped else None,
        1.75,
        "sigmoid" if grouped else "softmax",
        rand(h, as_type=torch.float32) if grouped else None,
        "bitmap_u8" if grouped else "bytemap",
    ), {}


def _tensor(shape, dtype):
    return {"shape": shape, "dtype": str(dtype)}


def config(binding, dtype):
    n, h = binding["rows"], binding["width"]
    mode = binding["variant"]
    inputs = {"logits": _tensor([n, h], dtype)}
    grouped = mode == "sigmoid-grouped"
    if grouped:
        inputs["expert_bias"] = _tensor([h], torch.float32)
    return {
        "inputs": inputs,
        "topk": 4,
        "scaling_factor": 1.75,
        "score_function": "sigmoid" if grouped else "softmax",
        "num_groups": 4 if grouped else None,
        "group_topk": 2 if grouped else None,
        "routing_map_format": "bitmap_u8" if grouped else "bytemap",
    }
