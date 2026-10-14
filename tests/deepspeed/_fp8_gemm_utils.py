# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared FP8 GEMM test/benchmark oracles; never imported by production code.

The only baseline is the unmodified DeepSpeed FP8 GEMM public API. Known upstream
limitations have a source/version/device-scoped skip policy. No kernel, launch
configuration or input storage is patched. Unexpected failures remain failures.
"""

import contextlib
import functools
import hashlib
import importlib
import subprocess
from pathlib import Path

import torch
import triton

import flag_train

UPSTREAM_M = (1, 2, 4, 8, 32, 64, 128, 256, 512, 1024, 2048)
# Fixed before benchmarking: the M=16/17 tile boundary and partial M=48/96 tiles.
# These supplement, rather than erase, the original unsupported cases.
SUPPLEMENTAL_M = (16, 17, 48, 96)
CORE_M = tuple(sorted(UPSTREAM_M + SUPPLEMENTAL_M))
CORE_K = 4096
CORE_N = 8192
CORE_GROUP_SIZE = 128
ATOL = {torch.bfloat16: 0.025, torch.float16: 0.005}
KNOWN_FP8_SOURCE_SHA256 = (
    "5c376c95cca24995283a13ca2a6ec7773cbad20aaa90e74e0ea7a18fe8826096"
)


@functools.lru_cache(maxsize=1)
def load_deepspeed():
    """Load the required independent implementation, without a skip/fallback."""
    if flag_train.vendor_name != "nvidia":
        raise RuntimeError("The current FP8 GEMM baseline suite requires NVIDIA CUDA")
    try:
        import deepspeed
        from deepspeed.linear import QuantizationConfig
        from deepspeed.ops.fp_quantizer import FP_Quantize, matmul_fp8

        module = importlib.import_module("deepspeed.ops.fp_quantizer.fp8_gemm_triton")
    except Exception as exc:
        raise RuntimeError(
            "DeepSpeed FP8 GEMM is a required baseline on NVIDIA. Install/build "
            "DeepSpeed with DS_BUILD_FP_QUANTIZER=1; no replacement oracle is "
            f"selected automatically. Original error: {exc!r}"
        ) from exc
    return deepspeed, QuantizationConfig, FP_Quantize, matmul_fp8, module


def baseline_metadata():
    deepspeed, _, _, _, module = load_deepspeed()
    source = Path(module.__file__).resolve()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(source.parent), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "mode": "original",
        "deepspeed_version": deepspeed.__version__,
        "deepspeed_commit": commit,
        "deepspeed_source": str(source),
        "deepspeed_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "triton_version": triton.__version__,
        "public_api": "deepspeed.ops.fp_quantizer.matmul_fp8",
        "kernel_modifications": "none",
        "launch_overrides": "none",
        "scale_storage": "unmodified fpq.get_scales() result; no padding",
        "validation_dtype": "torch.bfloat16",
        "upstream_m": UPSTREAM_M,
        "supplemental_m": SUPPLEMENTAL_M,
        "assertion": "BF16 train_assert_close rtol=.016/atol=.025 and upstream relative error <.004",
    }


@contextlib.contextmanager
def strict_matmul_context():
    """Keep reference FP32 accumulation precise and restore process settings."""
    names = (
        "allow_tf32",
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
    )
    backend = torch.backends.cuda.matmul
    previous = {name: getattr(backend, name) for name in names}
    try:
        for name in names:
            setattr(backend, name, False)
        yield
    finally:
        for name, value in previous.items():
            setattr(backend, name, value)


def original_deepspeed_skip_reason(m, k, n, dtype, group_size, device, metadata=None):
    """Explain a known upstream limitation before attempting that GEMM.

    Gate by the actual upstream source fingerprint, including wheel installs
    without Git metadata. An updated source never inherits an old skip. Only
    the four measured BF16 resource failures on H20/Triton 3.6 are excluded;
    numerical failures and candidate failures are never converted to skips.
    """
    metadata = baseline_metadata() if metadata is None else metadata
    if metadata["deepspeed_source_sha256"] != KNOWN_FP8_SOURCE_SHA256:
        return None
    if (k, n, group_size) != (CORE_K, CORE_N, CORE_GROUP_SIZE) or m not in (
        256,
        512,
        1024,
        2048,
    ):
        return None
    version = metadata["triton_version"]
    if dtype == torch.bfloat16 and m >= 256 and version.startswith("3.6."):
        props = torch.cuda.get_device_properties(device)
        limit = props.shared_memory_per_block_optin
        if "H20" in props.name and limit < 263168:
            return (
                "Original DeepSpeed FP8 source 5c376c95 (79caae1), BF16 M>=256 "
                f"on {props.name}/Triton {version}: requires 263168 bytes shared "
                f"memory per block, device limit is {limit}. The candidate is "
                "covered separately by the torch oracle."
            )
    return None


def make_quantized_inputs(m, k, n, dtype, group_size, device):
    """Generate actual DeepSpeed-quantized weights outside any timing region."""
    _, QuantizationConfig, FP_Quantize, _, _ = load_deepspeed()
    generator = torch.Generator(device=device).manual_seed(20260922)
    raw = torch.randn(k, n, device=device, dtype=torch.bfloat16, generator=generator)
    fpq = FP_Quantize(
        quantization_config=QuantizationConfig(
            q_dtype=torch.uint8, group_size=group_size
        )
    )
    weight, _ = fpq.quantize(raw, q_bits=8, return_meta_tensor=True)
    scale = fpq.get_scales()
    x = torch.randn(m, k, device=device, dtype=dtype, generator=generator)
    return x, weight, scale, fpq


def deepspeed_matmul(x, weight, scale, group_size, fpq):
    """Call the original public baseline API without any kernel substitution."""
    return load_deepspeed()[3](x, weight, scale, group_size, fpq)


def decode_custom_e4m3(weight, scale, group_size, dtype):
    """Independent arithmetic decoder, not the Triton/CUDA bitcast algorithm.

    Every byte is finite; code 0 is +2**-7 before scale, and exponent 15 is
    normal. This intentionally does NOT use torch.float8_e4m3fn semantics.
    Decode by rows to bound temporary memory for the 4096x8192 core weights.
    """
    k, n = weight.shape
    decoded = torch.empty_like(weight, dtype=dtype)
    scales = scale.reshape(-1)
    for start in range(0, k, 128):
        codes = weight[start : start + 128].to(torch.int64)
        sign = 1.0 - 2.0 * ((codes >> 7) & 1).to(torch.float64)
        exponent = ((codes >> 3) & 15).to(torch.float64)
        mantissa = (codes & 7).to(torch.float64)
        unscaled = sign * (1.0 + mantissa / 8.0) * torch.exp2(exponent - 7.0)
        groups = (
            torch.arange(
                start * n, start * n + codes.numel(), device=weight.device
            ).reshape_as(codes)
            // group_size
        )
        # The kernel multiplies in FP32 before rounding to the operand dtype.
        # E4M3 values are exactly representable; explicitly reproduce the FP32
        # product rounding, avoiding FP64->FP16 double-rounding differences.
        decoded[start : start + 128] = (unscaled.float() * scales[groups]).to(dtype)
    return decoded


def torch_reference(x, weight, scale, group_size):
    decoded = decode_custom_e4m3(weight, scale, group_size, x.dtype)
    return (x.float() @ decoded.float()).to(x.dtype)


def assert_fp8_close(actual, reference, *, mean_relative=True):
    """Use repository assertions and a predeclared FP8 GEMM-specific atol.

    Repository rtol is unchanged: BF16 .016 and FP16 .001 (stricter than the
    standalone test's .002). The absolute term handles near-zero cancellation;
    it is fixed before testing, not scaled by K or observed output errors.
    """
    from .. import accuracy_utils as utils

    assert bool(
        torch.isfinite(actual).all()
    ), "candidate/baseline contains nonfinite values"
    assert bool(torch.isfinite(reference).all()), "reference contains nonfinite values"
    utils.train_assert_close(
        utils.to_reference(actual),
        utils.to_reference(reference),
        actual.dtype,
        atol=ATOL[actual.dtype],
    )
    if mean_relative and actual.numel():
        upstream_error = (
            ((actual - reference).abs() / (actual.abs() + 1e-5)).sum() / actual.numel()
        ).item()
        assert (
            upstream_error < 0.004
        ), f"upstream mean relative error {upstream_error:.8g} must be <0.004"
        error = (
            ((actual.float() - reference.float()).abs() / (actual.float().abs() + 1e-5))
            .mean()
            .item()
        )
        assert error < 0.004, f"FP32 mean relative error {error:.8g} must be <0.004"
