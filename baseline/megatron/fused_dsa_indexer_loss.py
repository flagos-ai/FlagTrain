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

"""Megatron-LM FusedDSAIndexerLoss forward with tensor parallel size one.

Upstream reference: Megatron-LM c71687a.
"""

from types import SimpleNamespace

import torch

OP_NAME = "fused_dsa_indexer_loss"
DTYPES = (torch.float32, torch.bfloat16)
IS_INPLACE = False

_WORKLOADS = (
    (19, 16, "dense"),
    (128, 32, "sparse"),
    (17, 16, "sparse"),
    (33, 16, "dense"),
    (64, 16, "sparse"),
    (96, 16, "dense"),
    (128, 16, "sparse"),
    (32, 32, "dense"),
    (64, 32, "sparse"),
    (96, 32, "dense"),
    (192, 32, "sparse"),
)


def grid():
    return [
        {"rows": rows, "width": width, "variant": variant}
        for rows, width, variant in _WORKLOADS
    ]


def key_shape(binding):
    return [binding["rows"], 2, 2, binding["width"]]


class _SingleRank:
    def size(self):
        return 1


# The upstream forward checks the TP size before any collective.
_PROCESS_GROUPS = SimpleNamespace(tp=_SingleRank())


def native():
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        FusedDSAIndexerLoss,
    )

    def forward(q, weights, k, query, key, scale, topk, coeff, mask, sparse):
        return FusedDSAIndexerLoss.apply(
            q, weights, k, query, key, scale, topk, coeff, mask, sparse, _PROCESS_GROUPS
        )

    return forward


def build_inputs(binding, dtype, device):
    n, h = binding["rows"], binding["width"]
    mode = binding["variant"]
    rng = torch.Generator().manual_seed(4234321)

    def rand(*shape, as_type=dtype):
        return torch.randn(shape, generator=rng).to(device, as_type)

    q, weights, k = rand(n, 2, 2, h), rand(n, 2, 2), rand(n + 5, 2, h)
    query, key = rand(n, 2, 3, h), rand(n + 5, 2, 3, h)
    ends = torch.arange(n, device=device) + 2
    keys = torch.arange(n + 5, device=device)
    mask = torch.zeros(n, n + 5, device=device).masked_fill(
        keys >= ends[:, None], -torch.inf
    )
    return (q, weights, k, query, key, h ** (-0.5), 4, 0.7, mask, mode == "sparse"), {}


def _tensor(shape, dtype):
    return {"shape": shape, "dtype": str(dtype)}


def config(binding, dtype):
    n, h = binding["rows"], binding["width"]
    mode = binding["variant"]
    return {
        "inputs": {
            "q": _tensor([n, 2, 2, h], dtype),
            "weights": _tensor([n, 2, 2], dtype),
            "k": _tensor([n + 5, 2, h], dtype),
            "query": _tensor([n, 2, 3, h], dtype),
            "key": _tensor([n + 5, 2, 3, h], dtype),
            "mask": _tensor([n, n + 5], torch.float32),
        },
        "sparse_loss": mode == "sparse",
        "topk": 4,
        "softmax_scale": h ** (-0.5),
        "loss_coeff": 0.7,
    }
