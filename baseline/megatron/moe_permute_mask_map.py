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

"""Transformer Engine MoE mask-map permute forward.

Upstream reference: Transformer Engine 4f5c08e.
"""

import torch

OP_NAME = "moe_permute_mask_map"
DTYPES = (torch.float32, torch.bfloat16, torch.float16)
IS_INPLACE = False

_WORKLOADS = (
    (19, 33, "weighted"),
    (1024, 128, "padded"),
    (8, 16, "weighted"),
    (33, 64, "padded"),
    (64, 128, "weighted"),
    (129, 96, "padded"),
    (256, 256, "weighted"),
    (512, 256, "padded"),
)


def grid():
    return [
        {"rows": rows, "width": width, "variant": variant}
        for rows, width, variant in _WORKLOADS
    ]


def _output_rows(binding):
    n = binding["rows"]
    return 2 * (n - (n + 6) // 7) + (8 if binding["variant"] == "padded" else 0)


def key_shape(binding):
    return [binding["rows"], binding["width"]]


def native():
    from transformer_engine.pytorch.permutation import moe_permute_mask_map_forward

    return moe_permute_mask_map_forward


def build_inputs(binding, dtype, device):
    n, h = binding["rows"], binding["width"]
    mode = binding["variant"]
    rng = torch.Generator().manual_seed(4234321)

    def rand(*shape, as_type=dtype):
        return torch.randn(shape, generator=rng).to(device, as_type)

    inp = rand(n, h)
    token = torch.arange(n, device=device)[:, None]
    expert = torch.arange(4, device=device)[None]
    routing = (((token + expert) % 4 < 2) & (token % 7 != 0)).int()
    probs = rand(n, 4)
    probs[1] = 0
    pads = (
        torch.tensor([0, 1, 3, 6], dtype=torch.int32, device=device)
        if mode == "padded"
        else None
    )
    return (inp, routing, _output_rows(binding), probs, pads), {}


def _tensor(shape, dtype):
    return {"shape": shape, "dtype": str(dtype)}


def config(binding, dtype):
    n, h = binding["rows"], binding["width"]
    mode = binding["variant"]
    inputs = {
        "inp": _tensor([n, h], dtype),
        "routing_map": _tensor([n, 4], torch.int32),
        "probs": _tensor([n, 4], dtype),
    }
    padded = mode == "padded"
    if padded:
        inputs["pad_offsets"] = _tensor([4], torch.int32)
    return {
        "inputs": inputs,
        "num_out_tokens": _output_rows(binding),
        "pad_offsets": [0, 1, 3, 6] if padded else None,
    }
