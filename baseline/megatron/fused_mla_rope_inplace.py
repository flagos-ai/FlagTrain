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

"""Megatron-LM in-place MLA RoPE with collector-managed input restoration.

Upstream reference: Megatron-LM c71687a.
"""

import torch

OP_NAME = "fused_mla_rope_inplace"
DTYPES = (torch.float32, torch.bfloat16, torch.float16)
IS_INPLACE = True
MUTATED_INPUTS = (0,)

_WORKLOADS = (
    (19, 64, "dense"),
    (128, 64, "packed"),
    (8, 16, "dense"),
    (17, 32, "packed"),
    (33, 64, "dense"),
    (64, 16, "packed"),
    (128, 32, "dense"),
    (256, 64, "packed"),
)


def grid():
    return [
        {"rows": rows, "width": width, "variant": variant}
        for rows, width, variant in _WORKLOADS
    ]


def key_shape(binding):
    n, h = binding["rows"], binding["width"]
    return [n, 2, 3, h + 8] if binding["variant"] == "dense" else [n, 3, h + 8]


def native():
    from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_inplace

    return fused_mla_rope_inplace


def build_inputs(binding, dtype, device):
    n, h = binding["rows"], binding["width"]
    mode = binding["variant"]
    rng = torch.Generator().manual_seed(4234321)

    def rand(*shape, as_type=dtype):
        return torch.randn(shape, generator=rng).to(device, as_type)

    dense = mode == "dense"
    angles = rand(2 * n, 1, 1, h, as_type=torch.float32)
    cu_seqlens = (
        None
        if dense
        else torch.tensor([0, n // 2, n], dtype=torch.int32, device=device)
    )
    return (
        rand(*key_shape(binding)),
        angles.cos(),
        angles.sin(),
        8,
        h,
        cu_seqlens,
    ), {"remove_interleaving": not dense}


def _tensor(shape, dtype):
    return {"shape": shape, "dtype": str(dtype)}


def config(binding, dtype):
    n, h = binding["rows"], binding["width"]
    mode = binding["variant"]
    table_shape = [2 * n, 1, 1, h]
    inputs = {
        "t": _tensor(key_shape(binding), dtype),
        "cos": _tensor(table_shape, torch.float32),
        "sin": _tensor(table_shape, torch.float32),
    }
    packed = mode == "packed"
    if packed:
        inputs["cu_seqlens_q"] = _tensor([3], torch.int32)
    return {
        "inputs": inputs,
        "nope_dim": 8,
        "emb_dim": h,
        "remove_interleaving": packed,
    }
