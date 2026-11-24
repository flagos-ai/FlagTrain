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

"""Transformer Engine MoE chunk-sort forward.

Upstream reference: Transformer Engine 4f5c08e.
"""

import torch

OP_NAME = "moe_chunk_sort"
DTYPES = (torch.float32, torch.bfloat16, torch.float16)
IS_INPLACE = False

_WORKLOADS = (
    (19, 33, "weighted"),
    (1024, 128, "unweighted"),
    (8, 16, "weighted"),
    (33, 64, "unweighted"),
    (64, 128, "weighted"),
    (129, 96, "unweighted"),
    (256, 256, "weighted"),
    (512, 256, "unweighted"),
)


def grid():
    return [
        {"rows": rows, "width": width, "variant": variant}
        for rows, width, variant in _WORKLOADS
    ]


def key_shape(binding):
    return [binding["rows"], binding["width"]]


def native():
    from transformer_engine.pytorch.permutation import moe_chunk_sort_forward

    return moe_chunk_sort_forward


def build_inputs(binding, dtype, device):
    n, h = binding["rows"], binding["width"]
    mode = binding["variant"]
    rng = torch.Generator().manual_seed(4234321)

    def rand(*shape, as_type=dtype):
        return torch.randn(shape, generator=rng).to(device, as_type)

    sizes = [n // 3, 0, n // 4, n - n // 3 - n // 4]
    return (
        rand(n, h),
        torch.tensor(sizes, dtype=torch.int32, device=device),
        torch.tensor([3, 2, 1, 0], dtype=torch.int32, device=device),
        rand(n) if mode == "weighted" else None,
    ), {}


def _tensor(shape, dtype):
    return {"shape": shape, "dtype": str(dtype)}


def config(binding, dtype):
    n, h = binding["rows"], binding["width"]
    mode = binding["variant"]
    inputs = {
        "inp": _tensor([n, h], dtype),
        "split_sizes": _tensor([4], torch.int32),
        "sorted_idxs": _tensor([4], torch.int32),
    }
    if mode == "weighted":
        inputs["probs"] = _tensor([n], dtype)
    return {
        "inputs": inputs,
        "split_sizes": [n // 3, 0, n // 4, n - n // 3 - n // 4],
        "sorted_idxs": [3, 2, 1, 0],
    }
