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

"""Transformer Engine MoE mask-map unpermute forward.

Upstream reference: Transformer Engine 4f5c08e.
"""

import torch

OP_NAME = "moe_unpermute_mask_map"
DTYPES = (torch.float32, torch.bfloat16, torch.float16)
IS_INPLACE = False

_WORKLOADS = (
    (19, 33, "unweighted"),
    (1024, 128, "padded"),
    (8, 16, "unweighted"),
    (33, 64, "padded"),
    (64, 128, "unweighted"),
    (129, 96, "padded"),
    (256, 256, "unweighted"),
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
    return [_output_rows(binding), binding["width"]]


def native():
    from transformer_engine.pytorch.permutation import moe_unpermute_mask_map_forward

    return moe_unpermute_mask_map_forward


def build_inputs(binding, dtype, device):
    n, h = binding["rows"], binding["width"]
    mode = binding["variant"]
    rng = torch.Generator().manual_seed(4234321)

    def rand(*shape, as_type=dtype):
        return torch.randn(shape, generator=rng).to(device, as_type)

    from transformer_engine.pytorch.permutation import moe_permute_mask_map_forward

    inp = rand(n, h)
    token = torch.arange(n, device=device)[:, None]
    expert = torch.arange(4, device=device)[None]
    routing = (((token + expert) % 4 < 2) & (token % 7 != 0)).int()
    probs = pads = None
    if mode == "padded":
        probs = rand(n, 4)
        probs[1] = 0
        pads = torch.tensor([0, 1, 3, 6], dtype=torch.int32, device=device)
    # Prepare the permuted input and its row map before timing unpermute.
    permuted, row_map, _ = moe_permute_mask_map_forward(
        inp, routing, _output_rows(binding), probs, pads
    )
    return (permuted, row_map, probs, n, 4, h, pads), {}


def _tensor(shape, dtype):
    return {"shape": shape, "dtype": str(dtype)}


def config(binding, dtype):
    n, h = binding["rows"], binding["width"]
    mode = binding["variant"]
    inputs = {
        "inp": _tensor([_output_rows(binding), h], dtype),
        "row_id_map": _tensor([n, 9], torch.int32),
    }
    padded = mode == "padded"
    if padded:
        inputs["merging_probs"] = _tensor([n, 4], dtype)
        inputs["pad_offsets"] = _tensor([4], torch.int32)
    return {
        "inputs": inputs,
        "num_tokens": n,
        "num_experts": 4,
        "hidden_size": h,
        "pad_offsets": [0, 1, 3, 6] if padded else None,
    }
