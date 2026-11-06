# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""FP8 GEMM baseline: eager Torch decodes custom E4M3 and multiplies."""

import functools
import math

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


@functools.lru_cache(maxsize=16)
def _custom_e4m3_codebook(device):
    """Immutable format constants only; never cache input weights or indices."""
    return torch.tensor(
        [
            math.ldexp(
                (-1.0 if code & 128 else 1.0) * (8 + (code & 7)),
                ((code >> 3) & 15) - 10,
            )
            for code in range(256)
        ],
        dtype=torch.float32,
        device=device,
    )


def torch_dequantize_custom_e4m3(weight, scale, group_size, dtype):
    """Vectorized Torch eager format lookup, separate from the arithmetic oracle.

    All 256 codes are finite, including exponent zero/15. Cache only the fixed
    1 KiB format table, not decoded/scaled weights, input-derived indices or
    outputs. Each call reads every input byte and performs lookup and scaling.
    Full groups use broadcast scaling, with a separate final partial group.
    No per-call host transfers, scalar reads, compilation or Python row loop.
    """
    indices = weight.reshape(-1).to(torch.int32)
    values = torch.index_select(_custom_e4m3_codebook(weight.device), 0, indices)
    scales = scale.reshape(-1)
    full, tail = divmod(weight.numel(), group_size)
    if not tail:
        scaled = values.reshape(full, group_size) * scales.reshape(-1, 1)
    else:
        prefix = values[: full * group_size].reshape(full, group_size)
        prefix = (prefix * scales[:full].reshape(-1, 1)).reshape(-1)
        suffix = values[full * group_size :] * scales[full]
        scaled = torch.cat((prefix, suffix))
    return scaled.reshape(weight.shape).to(dtype)


def torch_composite_matmul(x, weight, scale, group_size):
    """Timed baseline: decode/scale/round afresh, then native-dtype matmul.

    This deliberately does not call torch_reference: that independent oracle
    uses chunked arithmetic and FP32 matmul (and optionally CPU transfers).
    """
    decoded = torch_dequantize_custom_e4m3(weight, scale, group_size, x.dtype)
    return torch.matmul(x, decoded)


def base_op():
    return torch_composite_matmul


def native():
    return base_op()


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
