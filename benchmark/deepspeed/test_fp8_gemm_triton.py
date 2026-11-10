# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Benchmark FP8 GEMM against DeepSpeed, with an eager Torch fallback."""

import functools
import math
import warnings

import pytest
import torch
from triton.runtime.errors import OutOfResources

import flag_train

from .. import base

OP_NAME = "fp8_gemm_triton"
DTYPES = [torch.bfloat16]
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
    """Decode, scale and round on each call before native-dtype matmul."""
    decoded = torch_dequantize_custom_e4m3(weight, scale, group_size, x.dtype)
    return torch.matmul(x, decoded)


@functools.lru_cache(maxsize=1)
def _load_deepspeed_fp8_gemm():
    try:
        from deepspeed.ops.fp_quantizer.fp8_gemm_triton import matmul_fp8_triton
    except Exception as exc:
        warnings.warn(
            f"DeepSpeed FP8 GEMM could not be loaded: {exc!r}; using Torch.",
            stacklevel=2,
        )
        return None
    return matmul_fp8_triton


class Fp8GemmBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPE_DESC = "M, K, N"

    def set_shapes(self, shape_file=None):
        self.shapes = [(m, CORE_K, CORE_N) for m in CORE_M]

    def set_more_shapes(self):
        return []

    def _measure_input(self, inputs, case_id=None):
        previous_op = self.torch_op
        if "latency_base" in self.to_bench_metrics and not base.Config.skip_native:
            args, kwargs = self.unpack_to_args_kwargs(inputs)
            upstream = _load_deepspeed_fp8_gemm()
            if upstream is not None:
                # Probe this exact shape outside timing; large tiles may not fit.
                try:
                    upstream(*args, **kwargs)
                except (OutOfResources, NotImplementedError) as exc:
                    warnings.warn(
                        f"DeepSpeed FP8 GEMM unavailable for {tuple(args[0].shape)}: "
                        f"{exc!r}; using Torch.",
                        stacklevel=2,
                    )
                else:
                    self.torch_op = upstream
            reference = "DeepSpeed" if self.torch_op is upstream else "Torch"
            print(f"FP8 GEMM baseline {tuple(args[0].shape)}: {reference}")
        try:
            metric = super()._measure_input(inputs, case_id=case_id)
        finally:
            self.torch_op = previous_op
        assert metric.error_msg is None, f"{case_id}: {metric.error_msg}"
        if metric.speedup is not None:
            label = case_id or metric.shape_detail
            assert (
                math.isfinite(metric.speedup) and metric.speedup > 0.8
            ), f"{label}: speedup {metric.speedup}x must be > 0.8x"
        return metric


def fp8_input_fn(shape, dtype, device):
    tensors = make_portable_inputs(*shape, dtype, CORE_GROUP_SIZE, device)
    yield (*tensors, CORE_GROUP_SIZE)


@pytest.mark.fp8_gemm_triton
@pytest.mark.skipif(not flag_train.runtime.device.support_bf16, reason="BF16 required")
def test_fp8_gemm_triton_perf():
    bench = Fp8GemmBenchmark(
        input_fn=fp8_input_fn,
        op_name=OP_NAME,
        torch_op=torch_composite_matmul,
        dtypes=DTYPES,
    )
    bench.set_train(flag_train.deepspeed.fp8_gemm_triton)
    bench.run()
