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

"""NVIDIA baseline for loco_quantized_reduction: DeepSpeed quantized_reduction.

Self-contained module; depends only on torch and deepspeed. It does not import
``flag_train`` or ``benchmark``. The baseline is fixed to DeepSpeed's
``QuantizerBuilder().load().quantized_reduction``; when DeepSpeed is unavailable
it raises instead of collecting a baseline or falling back to the torch
composition in ``benchmark/deepspeed/test_loco_quantized_reduction.py``.

The input construction is copied from that benchmark's ``_make_inputs``, and the
shapes from its ``SINGLE_PASS_SHAPES``/``MULTIPASS_SHAPES``/``NT_SCALING_SHAPES``.
"""

import torch

OP_NAME = "loco_quantized_reduction"
DTYPES = [torch.int8]
IS_INPLACE = False  # returns a fresh output buffer and a fresh scale tensor

# benchmark/deepspeed/test_loco_quantized_reduction.py
_SHAPES = [
    (1024, 4096, 8),
    (2048, 2048, 8),
    (512, 2048, 8),
    (512, 4096, 8),
    (512, 8192, 8),
    (4096, 4096, 8),
    (1024, 8192, 8),
    (256, 1024, 16),
    (512, 16384, 8),
    (256, 49152, 8),
    (1024, 4096, 4),
    (512, 2048, 4),
    (512, 16384, 4),
    (1024, 4096, 16),
    (512, 8192, 16),
    (256, 16384, 16),
]

# (num_bits, symmetric); both are operator arguments, not dtypes.
_VARIANTS = [(4, True), (4, False), (8, True), (8, False)]

_SEED = 0


def _load_deepspeed_reduce():
    """DeepSpeed's quantized_reduction and its QuantizationType enum."""
    try:
        from deepspeed.ops.op_builder import QuantizerBuilder

        mod = QuantizerBuilder().load()
        return mod.quantized_reduction, mod.QuantizationType
    except Exception as exc:
        raise RuntimeError(
            "loco_quantized_reduction baseline requires DeepSpeed's "
            f"quantized_reduction, which could not be loaded: {exc!r}. Install "
            "deepspeed and ensure its CUDA extension builds on this machine."
        ) from exc


def native():
    reduce_fn, qt_enum = _load_deepspeed_reduce()

    def call(
        data,
        scales,
        in_groups,
        out_groups,
        num_bits,
        quant_type,
        devices_per_node,
    ):
        return reduce_fn(
            data,
            scales,
            in_groups,
            out_groups,
            num_bits,
            qt_enum(int(quant_type)),
            devices_per_node,
        )

    return call


def grid():
    return [
        {"shape": shape, "bits": bits, "symmetric": symmetric}
        for shape in _SHAPES
        for bits, symmetric in _VARIANTS
    ]


def build_inputs(binding, dtype, device):
    """Mirror benchmark/deepspeed/test_loco_quantized_reduction.py::_make_inputs."""
    out_groups, bpg, num_tensors = binding["shape"]
    num_bits = binding["bits"]
    symmetric = binding["symmetric"]

    in_groups = out_groups * num_tensors
    epit = out_groups * bpg
    scales_elems = 1 if symmetric else 2

    gen = torch.Generator(device="cpu").manual_seed(_SEED)
    data = torch.randint(
        -128, 128, (num_tensors * epit,), dtype=torch.int8, generator=gen
    ).to(device)

    scales = (
        torch.rand((in_groups * scales_elems,), dtype=torch.float32, generator=gen)
        .mul_(0.05)
        .add_(1e-3)
    )
    if not symmetric:
        scales = scales.view(-1, 2)
        scales = torch.stack([scales[:, 0], scales[:, 1] - 0.026], dim=1).reshape(-1)

    args = (
        data,
        scales.contiguous().to(device),
        in_groups,
        out_groups,
        num_bits,
        0 if symmetric else 1,
        num_tensors,
    )
    return args, {}


def key_shape(binding):
    out_groups, bpg, num_tensors = binding["shape"]
    return [
        out_groups,
        bpg,
        num_tensors,
        binding["bits"],
        int(binding["symmetric"]),
    ]


def config(binding, dtype):
    """Describe the real inputs/outputs for baseline matching."""
    out_groups, bpg, num_tensors = binding["shape"]
    num_bits = binding["bits"]
    symmetric = binding["symmetric"]

    in_groups = out_groups * num_tensors
    scales_elems = 1 if symmetric else 2
    dtype_str = str(dtype)
    scale_dtype = str(torch.float32)

    return {
        "inputs": {
            "input_vals": {
                "shape": [num_tensors * out_groups * bpg],
                "dtype": dtype_str,
            },
            "input_scales": {
                "shape": [in_groups * scales_elems],
                "dtype": scale_dtype,
            },
            "in_groups": {"scalar": in_groups},
            "out_groups": {"scalar": out_groups},
            "num_bits": {"scalar": num_bits},
            "quant_type": {"scalar": 0 if symmetric else 1},
            "devices_per_node": {"scalar": num_tensors},
        },
        "outputs": {
            "out": {"shape": [out_groups * bpg], "dtype": dtype_str},
            "out_st": {
                "shape": [out_groups, scales_elems],
                "dtype": scale_dtype,
            },
        },
        "dims": {
            "out_groups": out_groups,
            "elems_per_out_group": bpg,
            "num_tensors": num_tensors,
            "num_bits": num_bits,
            "symmetric": int(symmetric),
        },
    }
