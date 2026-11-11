# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Collect the baseline from DeepSpeed's native Triton FP8 GEMM."""

import torch

OP_NAME = "fp8_gemm_triton"
DTYPES = [torch.bfloat16]
IS_INPLACE = False
CORE_M = (1, 2, 4, 8, 16, 17, 32, 48, 64, 96, 128, 256, 512, 1024, 2048)
CORE_K = 4096
CORE_N = 8192
CORE_GROUP_SIZE = 128


def make_portable_inputs(m, k, n, dtype, group_size, device):
    """Generate deterministic encoded weights without a quantizer."""
    generator = torch.Generator(device="cpu").manual_seed(20260922)
    x = torch.randn(m, k, generator=generator).to(dtype)
    magnitude = torch.randint(32, 88, (k, n), generator=generator, dtype=torch.int32)
    sign = torch.randint(0, 2, (k, n), generator=generator, dtype=torch.int32)
    weight = (magnitude | (sign << 7)).to(torch.uint8)
    groups = (k * n + group_size - 1) // group_size
    scale = 0.125 + 0.125 * torch.rand(groups, 1, generator=generator)
    return x.to(device), weight.to(device), scale.to(device)


def native():
    """Resolve DeepSpeed's FP8 GEMM; fail if the native collector cannot load it."""
    try:
        from deepspeed.ops.fp_quantizer.fp8_gemm_triton import matmul_fp8_triton
    except Exception as exc:
        raise RuntimeError(f"Cannot load DeepSpeed's Triton FP8 GEMM: {exc!r}") from exc
    return matmul_fp8_triton


def grid():
    return [{"shape": (m, CORE_K, CORE_N)} for m in CORE_M]


def build_inputs(binding, dtype, device):
    tensors = make_portable_inputs(*binding["shape"], dtype, CORE_GROUP_SIZE, device)
    return (*tensors, CORE_GROUP_SIZE), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    m, k, n = binding["shape"]
    return {
        "inputs": {
            "activation": {"shape": [m, k], "dtype": str(dtype)},
            "weight": {"shape": [k, n], "dtype": "torch.uint8"},
            "scale": {"shape": [k * n // CORE_GROUP_SIZE, 1], "dtype": "torch.float32"},
            "group_size": {"scalar": CORE_GROUP_SIZE},
        },
        "outputs": {"out": {"shape": [m, n], "dtype": str(dtype)}},
        "dims": {"M": m, "K": k, "N": n},
    }
