# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared FP8 GEMM test/benchmark oracles; never imported by production code.

Ascend, MetaX and Hygon use Torch dequantization + matmul as the full baseline;
other vendors retain DeepSpeed's original public API and real quantizer. The
independent arithmetic oracle and these Torch baselines do not require DeepSpeed.
Known upstream limitations have a source/version/device-scoped skip policy.
Unexpected failures remain failures; there is no exception-triggered fallback.
"""

import contextlib
import functools
import hashlib
import importlib
import math
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
LARGE_M = (256, 512, 1024, 2048)
BASE_M = tuple(m for m in CORE_M if m not in LARGE_M)
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
    try:
        import deepspeed
        from deepspeed.linear import QuantizationConfig
        from deepspeed.ops.fp_quantizer import FP_Quantize, matmul_fp8

        module = importlib.import_module("deepspeed.ops.fp_quantizer.fp8_gemm_triton")
    except Exception as exc:
        raise RuntimeError(
            "DeepSpeed FP8 GEMM is required for original-baseline acceptance. Install/build "
            "DeepSpeed (DS_BUILD_FP_QUANTIZER=1 for the CUDA quantizer); no replacement oracle is "
            f"selected automatically. Original error: {exc!r}"
        ) from exc
    return deepspeed, QuantizationConfig, FP_Quantize, matmul_fp8, module


def baseline_metadata():
    common = {
        "vendor": flag_train.vendor_name,
        "device": str(flag_train.device),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "validation_dtype": "torch.bfloat16",
        "upstream_m": UPSTREAM_M,
        "supplemental_m": SUPPLEMENTAL_M,
        "base_m": BASE_M,
        "large_m": LARGE_M,
        "comparison_scope": "base-and-large; all shapes attempted on every vendor",
        "assertion": "BF16 train_assert_close rtol=.016/atol=.025 and upstream relative error <.004",
    }
    if flag_train.vendor_name in ("ascend", "metax", "hygon"):
        source = Path(__file__).resolve()
        return {
            **common,
            "mode": "torch-dequantize-matmul",
            "public_api": "torch_composite_matmul: torch_dequantize_custom_e4m3 + torch.matmul",
            "baseline_source": str(source),
            "baseline_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "deepspeed_version": None,
            "deepspeed_commit": None,
            "deepspeed_source_sha256": None,
            "execution": "torch eager; no torch.compile or cached decoded weights",
            "timed_scope": (
                "uint8-to-int32 indices + format lookup + FP32 scale + BF16 cast "
                "+ BF16 matmul + allocations"
            ),
            "constant_setup": (
                "fixed 256-entry FP32 format table (1 KiB/device), independent "
                "of inputs; initialized before timing"
            ),
            "scale_storage": "pre-generated contiguous float32 [groups, 1]; no padding",
            "input_pipeline": "fixed-seed CPU-generated custom E4M3 uint8 weights; quantizer not evaluated",
        }
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
        **common,
        "mode": "original",
        "deepspeed_version": deepspeed.__version__,
        "deepspeed_commit": commit,
        "deepspeed_source": str(source),
        "deepspeed_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "public_api": "deepspeed.ops.fp_quantizer.matmul_fp8",
        "kernel_modifications": "none",
        "launch_overrides": "none",
        "scale_storage": "unmodified fpq.get_scales() result; no padding",
        "input_pipeline": "original DeepSpeed FP_Quantize.quantize",
        "timed_scope": "original DeepSpeed allocating FP8 GEMM entry",
    }


@contextlib.contextmanager
def strict_matmul_context():
    """Keep reference FP32 accumulation precise and restore process settings."""
    # These precision controls belong to CUDA/ROCm, not every PrivateUse1 backend.
    if torch.device(flag_train.device).type != "cuda":
        yield
        return
    names = (
        "allow_tf32",
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
    )
    backend = torch.backends.cuda.matmul
    previous = {
        name: getattr(backend, name) for name in names if hasattr(backend, name)
    }
    try:
        for name in previous:
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
    # CUDA-compatible APIs on other vendors do not imply the H20 resource limit.
    if flag_train.vendor_name != "nvidia":
        return None
    metadata = baseline_metadata() if metadata is None else metadata
    if metadata["deepspeed_source_sha256"] != KNOWN_FP8_SOURCE_SHA256:
        return None
    if (k, n, group_size) != (CORE_K, CORE_N, CORE_GROUP_SIZE) or m not in LARGE_M:
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


def make_baseline_inputs(m, k, n, dtype, group_size, device):
    """Build identical operands for both implementations outside timing.

    Ascend, MetaX and Hygon use Torch dequantization + matmul, with inputs
    independent of the DeepSpeed quantizer. This is a fixed policy, never an
    exception-triggered fallback. Other vendors keep real DeepSpeed quantization.
    """
    if flag_train.vendor_name in ("ascend", "metax", "hygon"):
        return (*make_portable_inputs(m, k, n, dtype, group_size, device), None)
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


def make_portable_inputs(m, k, n, dtype, group_size, device):
    """Fixed CPU-generated encoded inputs; no vendor RNG or quantizer dependency.

    These are valid custom E4M3 bytes, not a replacement for DeepSpeed's
    quantizer. Explicit Torch-baseline comparisons use these operands; other
    vendors' original-public-API acceptance continues to use the real quantizer.
    """
    generator = torch.Generator(device="cpu").manual_seed(20260922)
    x = torch.randn(m, k, generator=generator).to(dtype)
    magnitude = torch.randint(32, 88, (k, n), generator=generator, dtype=torch.int32)
    sign = torch.randint(0, 2, (k, n), generator=generator, dtype=torch.int32)
    weight = (magnitude | (sign << 7)).to(torch.uint8)
    groups = (k * n + group_size - 1) // group_size
    scale = 0.125 + 0.125 * torch.rand(groups, 1, generator=generator)
    return x.to(device), weight.to(device), scale.to(device)


def baseline_matmul(x, weight, scale, group_size, fpq):
    """Run the explicit vendor baseline; propagate errors without substitution."""
    if flag_train.vendor_name in ("ascend", "metax", "hygon"):
        return torch_composite_matmul(x, weight, scale, group_size)
    return load_deepspeed()[3](x, weight, scale, group_size, fpq)


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
        codes = weight[start : start + 128].to(torch.int32)
        sign = 1.0 - 2.0 * ((codes >> 7) & 1).to(torch.float32)
        exponent = (codes >> 3) & 15
        mantissa = codes & 7
        # (1 + mantissa/8) * 2**(exponent-7), evaluated exactly.
        # Device exp2 can approximate even integer powers of two. The largest
        # shifted integer is 15 << 15, exactly representable in int32/FP32.
        # This remains an arithmetic oracle, independent of kernel bitcasts.
        magnitude = ((8 + mantissa) << exponent).to(torch.float32) * (1.0 / 1024)
        unscaled = sign * magnitude
        groups = (
            torch.arange(
                start * n,
                start * n + codes.numel(),
                device=weight.device,
                dtype=torch.int32,
            ).reshape_as(codes)
            // group_size
        )
        # The kernel multiplies in FP32 before rounding to the operand dtype.
        # E4M3 values are exactly representable; explicitly reproduce the FP32
        # product rounding, avoiding FP64->FP16 double-rounding differences.
        decoded[start : start + 128] = (unscaled.float() * scales[groups]).to(dtype)
    return decoded


def torch_reference(x, weight, scale, group_size):
    from .. import accuracy_utils as utils

    # --ref cpu must move operands before decoding/matmul, not just the result.
    x, weight, scale = (utils.to_reference(t) for t in (x, weight, scale))
    decoded = decode_custom_e4m3(weight, scale, group_size, x.dtype)
    return (x.float() @ decoded.float()).to(x.dtype)


def assert_fp8_close(actual, reference, *, mean_relative=True):
    """Use repository assertions and a predeclared FP8 GEMM-specific atol.

    Repository rtol is unchanged: BF16 .016 and FP16 .001 (stricter than the
    standalone test's .002). The absolute term handles near-zero cancellation;
    it is fixed before testing, not scaled by K or observed output errors.
    """
    from .. import accuracy_utils as utils

    actual, reference = utils.to_reference(actual), utils.to_reference(reference)
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
