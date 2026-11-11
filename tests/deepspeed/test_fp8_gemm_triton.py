# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Check FP8 GEMM through its public interface against independent arithmetic."""

import contextlib
import functools
import math
import warnings

import pytest
import torch
from triton.runtime.errors import OutOfResources

import flag_train

from .. import accuracy_utils as utils

pytestmark = pytest.mark.fp8_gemm_triton
DTYPES = (torch.bfloat16,)
CORE_M = (1, 2, 4, 8, 16, 17, 32, 48, 64, 96, 128, 256, 512, 1024, 2048)
CORE_K = 4096
CORE_N = 8192
CORE_GROUP_SIZE = 128
ATOL = {torch.bfloat16: 0.025}
LARGE_M = (256, 512, 1024, 2048)


def make_test_inputs(m, k, n, dtype, group_size, device):
    """Generate deterministic encoded weights without a quantizer."""
    generator = torch.Generator(device="cpu").manual_seed(20260922)
    x = torch.randn(m, k, generator=generator).to(dtype)
    magnitude = torch.randint(32, 88, (k, n), generator=generator, dtype=torch.int32)
    sign = torch.randint(0, 2, (k, n), generator=generator, dtype=torch.int32)
    weight = (magnitude | (sign << 7)).to(torch.uint8)
    groups = (k * n + group_size - 1) // group_size
    scale = 0.125 + 0.125 * torch.rand(groups, 1, generator=generator)
    return x.to(device), weight.to(device), scale.to(device)


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
    # --ref cpu must move operands before decoding/matmul, not just the result.
    x, weight, scale = (utils.to_reference(t) for t in (x, weight, scale))
    decoded = decode_custom_e4m3(weight, scale, group_size, x.dtype)
    return (x.float() @ decoded.float()).to(x.dtype)


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


def core_reference(x, weight, scale, group_size):
    # Honor --ref=cpu. Edge tests use the arithmetic reference for layouts
    # outside the upstream kernel's aligned BF16 contract.
    upstream = None if utils.TO_CPU else _load_deepspeed_fp8_gemm()
    if upstream is not None:
        try:
            return upstream(x, weight, scale, group_size), "deepspeed"
        except (OutOfResources, NotImplementedError) as exc:
            warnings.warn(
                f"DeepSpeed FP8 GEMM unavailable for {tuple(x.shape)}: "
                f"{exc!r}; using Torch.",
                stacklevel=2,
            )
    return torch_reference(x, weight, scale, group_size), "torch"


def assert_fp8_close(actual, reference, *, mean_relative=True):
    """Use repository assertions and a predeclared FP8 GEMM-specific atol.

    Repository rtol is unchanged: BF16 .016 and FP16 .001 (stricter than the
    standalone test's .002). The absolute term handles near-zero cancellation;
    it is fixed before testing, not scaled by K or observed output errors.
    """
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


@pytest.mark.parametrize("zero_scale", (False, True))
def test_fp8_gemm_triton_large_decoder_codes(zero_scale):
    """Cover all encodings and flat scales in large-M workspace paths."""
    generator = torch.Generator(device="cpu").manual_seed(20260922)
    x = (torch.randn((512, 128), generator=generator) * 0.25).to(torch.bfloat16)
    weight = (torch.arange(128 * 256) % 256).to(torch.uint8).reshape(128, 256)
    scale = 0.123 + 0.037 * (torch.arange(256, dtype=torch.float32) % 7)
    scale[::7] = 0
    if zero_scale:
        scale.zero_()
    x, weight, scale = (t.to(flag_train.device) for t in (x, weight, scale))
    actual = flag_train.deepspeed.fp8_gemm_triton(x, weight, scale, 128)
    reference = torch_reference(x, weight, scale, 128)
    if zero_scale:
        utils.train_assert_equal(
            utils.to_reference(actual), utils.to_reference(reference)
        )
    else:
        assert_fp8_close(actual, reference, mean_relative=False)


@pytest.mark.parametrize("m", (1, 17, 1024))
def test_fp8_gemm_triton_does_not_cache_inputs(m):
    """Fused and workspace paths must observe input mutations on every call."""
    device = flag_train.device
    x = torch.ones((m, 128), dtype=torch.bfloat16, device=device)
    weight = torch.zeros((128, 128), dtype=torch.uint8, device=device)
    scale = torch.full((128, 1), 1 / 128, dtype=torch.float32, device=device)
    for code, scale_value, activation, expected in (
        (0, 1 / 128, 1, 1 / 128),
        (127, 1 / 128, 1, 480),
        (127, 1 / 64, 1, 960),
        (127, 1 / 64, 0, 0),
    ):
        weight.fill_(code)
        scale.fill_(scale_value)
        x.fill_(activation)
        actual = flag_train.deepspeed.fp8_gemm_triton(x, weight, scale, 128)
        utils.train_assert_equal(
            actual.cpu(), torch.full((m, 128), expected, dtype=x.dtype)
        )


@pytest.fixture(scope="module", autouse=True)
def fp8_environment():
    if not utils.bf16_is_supported:
        pytest.skip("The active backend does not support BF16")
    with strict_matmul_context():
        yield


@pytest.fixture(
    scope="module",
    params=[
        pytest.param((m, dtype), id=f"M{m}-{str(dtype).removeprefix('torch.')}")
        for dtype in DTYPES
        for m in CORE_M
    ],
)
def core_case(request):
    m, dtype = request.param
    x, weight, scale = make_test_inputs(
        m, CORE_K, CORE_N, dtype, CORE_GROUP_SIZE, flag_train.device
    )
    reference, reference_name = core_reference(x, weight, scale, CORE_GROUP_SIZE)
    return x, weight, scale, reference, reference_name


def test_fp8_gemm_triton_candidate_vs_reference(core_case, record_property):
    """Prefer DeepSpeed for the aligned BF16 shapes."""
    x, weight, scale, reference, reference_name = core_case
    record_property("reference", reference_name)
    record_property("shape_group", "large-m" if x.shape[0] in LARGE_M else "base")
    actual = flag_train.deepspeed.fp8_gemm_triton(x, weight, scale, CORE_GROUP_SIZE)
    assert_fp8_close(actual, reference)


# All shapes below are tested against the mathematical oracle, not broadened
# into a claim that DeepSpeed's original wrapper supports these extensions.
EDGE_CASES = (
    ("all_256_codes", 1, 1, 256, 128),
    ("tails_mnk_flat_group", 5, 133, 77, 17),
    ("group_crosses_n_tile", 9, 129, 193, 96),
    ("noncontiguous_activation", 7, 65, 71, 23),
    ("negative_activations_and_codes", 4, 31, 67, 19),
    ("zero_scale", 3, 17, 69, 13),
    ("large_decoded_range", 3, 17, 65, 29),
    ("medium_m_before_boundary", 16, 256, 256, 128),
    ("medium_m_at_boundary", 17, 256, 256, 128),
    ("medium_m_tile_end", 32, 256, 256, 128),
    ("medium_m_next_tile", 33, 256, 256, 128),
    ("medium_m_range_end", 64, 256, 256, 128),
    ("medium_m_after_boundary", 65, 256, 256, 128),
    ("noncontiguous_activation_medium_m", 33, 256, 256, 128),
    ("medium_m_unaligned_k", 33, 133, 256, 128),
    ("medium_m_unaligned_n", 33, 256, 193, 128),
    ("medium_m_other_group", 33, 256, 256, 96),
    ("scale_group_spans_n_tiles", 33, 65, 256, 256),
    ("scale_group_larger_than_n_tile", 17, 129, 512, 512),
    ("scale_group_unaligned_tail", 65, 67, 513, 512),
    ("medium_m_last_small_tile", 128, 256, 256, 128),
    ("medium_m_next_large_tile", 129, 256, 256, 128),
    ("large_m_before_boundary", 511, 256, 256, 128),
    ("large_m_at_boundary", 512, 256, 256, 128),
    ("large_m_partial_tiles", 513, 256, 384, 128),
    ("noncontiguous_activation_large_m", 513, 256, 256, 128),
    ("large_m_generic_layout", 512, 133, 77, 17),
    ("large_m_before_workspace", 1023, 256, 256, 128),
    ("large_m_at_workspace", 1024, 256, 256, 128),
    ("large_m_workspace_partial_tile", 1025, 256, 256, 128),
    ("noncontiguous_activation_workspace", 1025, 256, 256, 128),
    ("large_m_workspace_unaligned_k", 1024, 133, 256, 128),
    ("large_m_workspace_other_group", 1024, 256, 256, 96),
    ("empty_m", 0, 17, 19, 7),
    ("empty_n", 3, 17, 0, 7),
    ("empty_k", 3, 0, 19, 7),
)


def test_fp8_gemm_triton_oracle_contract():
    codes = torch.arange(256, device=flag_train.device).to(torch.uint8).reshape(1, 256)
    scale = torch.tensor([1 / 128], dtype=torch.float32, device=flag_train.device)
    values = decode_custom_e4m3(
        utils.to_reference(codes), utils.to_reference(scale), 256, torch.float32
    ).reshape(-1)
    assert values[0].item() == math.ldexp(1.0, -14)  # not zero/subnormal
    assert values[120].item() == 2.0  # exponent 15 is finite
    assert values[127].item() == 3.75
    assert values[255].item() == -3.75
    assert bool(torch.isfinite(values).all())
    utils.train_assert_equal(values[:128], -values[128:])
    # Check every encoding exactly against scalar arithmetic, including on
    # backends whose floating-point exp2 is not exact at integer exponents.
    expected = torch.tensor(
        [
            math.ldexp(
                (-1.0 if code & 128 else 1.0) * (8 + (code & 7)),
                ((code >> 3) & 15) - 17,
            )
            for code in range(256)
        ],
        dtype=torch.float32,
        device=values.device,
    )
    utils.train_assert_equal(values, expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "name,m,k,n,group_size", EDGE_CASES, ids=[c[0] for c in EDGE_CASES]
)
def test_fp8_gemm_triton_edges(name, m, k, n, group_size, dtype):
    # Build the same bytes on every vendor, including those without device RNG.
    device = "cpu"
    generator = torch.Generator(device="cpu").manual_seed(20260922)
    noncontiguous = name.startswith("noncontiguous_activation")
    shape = (k, m) if noncontiguous else (m, k)
    x = (torch.randn(shape, device=device, generator=generator) * 0.5).to(dtype)
    if noncontiguous:
        x = x.t()
        assert not x.is_contiguous()
    index = torch.arange(k * n, device=device)
    codes = (index * 73 + 19) % 256
    groups = triton_cdiv(k * n, group_size)
    scales = (8 + 2 * (torch.arange(groups, device=device) % 4)).float() / 1024
    if name == "all_256_codes":
        x.fill_(1)
        codes = index % 256
        scales.fill_(1 / 128)
    elif name == "negative_activations_and_codes":
        x = -x.abs() - 0.03125
        codes = 128 + ((index * 37 + 11) % 128)
    elif name == "zero_scale":
        scales.zero_()
    elif name == "large_decoded_range":
        x *= 0.25
        choices = torch.tensor([0x7F, 0xFF, 0x78, 0xF8, 0, 0x80], device=device)
        codes = choices[index % choices.numel()]
        scales = torch.tensor([0.25, 0.5, 0.75, 1.0], device=device)[
            torch.arange(groups, device=device) % 4
        ]
    weight = codes.to(torch.uint8).reshape(k, n).contiguous()
    # Cover both the usual [groups, 1] layout and a flat scale tensor.
    if name != "group_crosses_n_tile":
        scales = scales.reshape(groups, 1)
    x, weight, scales = (t.to(flag_train.device) for t in (x, weight, scales))
    reference = torch_reference(x, weight, scales, group_size)
    actual = flag_train.deepspeed.fp8_gemm_triton(x, weight, scales, group_size)
    if name in ("all_256_codes", "zero_scale", "empty_m", "empty_n", "empty_k"):
        utils.train_assert_equal(
            utils.to_reference(actual), utils.to_reference(reference)
        )
    elif name == "large_decoded_range":
        # Retain the predeclared standalone large-range policy, not an observed
        # error-derived tolerance. Atol grows with the reference RMS only here.
        rms = reference.cpu().double().square().mean().sqrt().item()
        factor = 0.004 if dtype == torch.bfloat16 else 0.0005
        assert bool(torch.isfinite(actual).all())
        utils.train_assert_close(
            utils.to_reference(actual),
            utils.to_reference(reference),
            dtype,
            atol=max(ATOL[dtype], factor * rms),
        )
    else:
        assert_fp8_close(actual, reference, mean_relative=False)


def triton_cdiv(a, b):
    return (a + b - 1) // b


INVALID_CASES = (
    ("float32_activation", TypeError),
    ("int8_weight", TypeError),
    ("float16_scale", TypeError),
    ("rank1_activation", ValueError),
    ("rank1_weight", ValueError),
    ("incompatible_k", ValueError),
    ("wrong_scale_count", ValueError),
    ("zero_group", ValueError),
    ("negative_group", ValueError),
    ("float_group", ValueError),
    ("bool_group", ValueError),
    ("cpu_activation", ValueError),
    ("noncontiguous_weight", ValueError),
    ("noncontiguous_scale", ValueError),
)


@pytest.mark.parametrize("case,error", INVALID_CASES, ids=[c[0] for c in INVALID_CASES])
def test_fp8_gemm_triton_invalid_inputs(case, error):
    device = flag_train.device
    x = torch.ones((2, 3), dtype=torch.bfloat16, device=device)
    weight = torch.arange(15, device=device).to(torch.uint8).reshape(3, 5)
    scale = torch.full((4, 1), 1 / 128, dtype=torch.float32, device=device)
    group = 4
    if case == "float32_activation":
        x = x.float()
    elif case == "int8_weight":
        weight = weight.to(torch.int8)
    elif case == "float16_scale":
        scale = scale.half()
    elif case == "rank1_activation":
        x = x.reshape(-1)
    elif case == "rank1_weight":
        weight = weight.reshape(-1)
    elif case == "incompatible_k":
        x = x[:, :2]
    elif case == "wrong_scale_count":
        scale = scale[:-1]
    elif case == "zero_group":
        group = 0
    elif case == "negative_group":
        group = -4
    elif case == "float_group":
        group = 4.0
    elif case == "bool_group":
        group = True
    elif case == "cpu_activation":
        x = x.cpu()
    elif case == "noncontiguous_weight":
        weight = weight.t().contiguous().t()
    elif case == "noncontiguous_scale":
        scale = torch.full((8,), 1 / 128, dtype=torch.float32, device=device)[::2]
    with pytest.raises(error):
        flag_train.deepspeed.fp8_gemm_triton(x, weight, scale, group)
