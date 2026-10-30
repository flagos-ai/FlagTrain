# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Forward-only DeepSpeed custom FP8 GEMM tests, following test_lamb.py.

The mathematical torch oracle is independent of the production bitcast decoder.
Every core candidate case runs against that oracle. The separate baseline is
Torch dequantization + matmul on Ascend/MetaX/Hygon, and the original DeepSpeed API on
other vendors. Only pinned, known-broken NVIDIA baseline cases are skipped;
candidate/oracle checks are never skipped for baseline limitations.

Inputs are quantized uint8 weights (not differentiable torch float8 tensors),
so this operator has a forward-only contract and no parameter-gradient tests.
"""

import json
import math

import pytest
import torch

import flag_train
from flag_train.deepspeed import matmul_fp8_triton

from .. import accuracy_utils as utils
from ._fp8_gemm_utils import (
    ATOL,
    BASE_M,
    CORE_GROUP_SIZE,
    CORE_K,
    CORE_M,
    CORE_N,
    LARGE_M,
    assert_fp8_close,
    baseline_matmul,
    baseline_metadata,
    decode_custom_e4m3,
    make_baseline_inputs,
    make_portable_inputs,
    original_deepspeed_skip_reason,
    strict_matmul_context,
    torch_dequantize_custom_e4m3,
    torch_reference,
)

pytestmark = pytest.mark.fp8_gemm
DTYPES = (torch.bfloat16,)


@pytest.mark.parametrize("vendor", ("ascend", "nvidia", "hygon", "thead", "metax"))
def test_fp8_baseline_entry_policy(monkeypatch, vendor):
    """Explicit Torch vendors and original public API signatures stay intact."""
    from . import _fp8_gemm_utils as helpers

    calls = []
    result = object()

    def public(*args):
        calls.append(("public", args))
        return result

    def composite(*args):
        calls.append(("torch", args))
        return result

    def load():
        assert vendor not in (
            "ascend",
            "metax",
            "hygon",
        ), "Torch paths must not import DeepSpeed"
        return None, None, None, public, None

    monkeypatch.setattr(flag_train, "vendor_name", vendor)
    monkeypatch.setattr(helpers, "load_deepspeed", load)
    monkeypatch.setattr(helpers, "torch_composite_matmul", composite)
    operands = tuple(object() for _ in range(5))
    assert helpers.baseline_matmul(*operands) is result
    expected = (
        ("torch", operands[:4])
        if vendor in ("ascend", "metax", "hygon")
        else ("public", operands)
    )
    assert calls == [expected]


@pytest.mark.parametrize("vendor", ("ascend", "metax", "hygon", "nvidia"))
def test_fp8_baseline_failure_is_not_replaced(monkeypatch, vendor):
    """Both selected baselines propagate errors, without an implicit fallback."""
    from . import _fp8_gemm_utils as helpers

    def failing(*args):
        raise RuntimeError("selected baseline failed")

    monkeypatch.setattr(flag_train, "vendor_name", vendor)
    monkeypatch.setattr(helpers, "torch_composite_matmul", failing)
    monkeypatch.setattr(
        helpers,
        "load_deepspeed",
        lambda: (None, None, None, failing, None),
    )
    with pytest.raises(RuntimeError, match="selected baseline failed"):
        helpers.baseline_matmul(None, None, None, 128, None)


@pytest.mark.parametrize("group_size", (1, 17, 128, 256, 512))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_fp8_torch_baseline_decoder(group_size, dtype):
    """Exact agreement on all codes, zero/non-power scales and partial groups."""
    weight = torch.arange(256).to(torch.uint8).reshape(16, 16).to(flag_train.device)
    groups = triton_cdiv(weight.numel(), group_size)
    scale = (torch.arange(groups, dtype=torch.float32) % 7) * 0.037 + 0.123
    if groups > 1:
        scale[0] = 0
    scale = scale.reshape(-1, 1).to(flag_train.device)
    actual = torch_dequantize_custom_e4m3(weight, scale, group_size, dtype)
    expected = decode_custom_e4m3(weight.cpu(), scale.cpu(), group_size, dtype)
    utils.train_assert_equal(actual.cpu(), expected)


def test_fp8_torch_baseline_times_decode_and_matmul(monkeypatch):
    """Each timed call must decode first; no cached weight or reference call."""
    from . import _fp8_gemm_utils as helpers

    x = torch.empty((1, 1), dtype=torch.bfloat16)
    weight, scale, decoded, output = (object() for _ in range(4))
    calls = []

    def decode(w, s, group, dtype):
        assert (w, s, group, dtype) == (weight, scale, 128, x.dtype)
        calls.append("decode")
        return decoded

    def matmul(a, b):
        assert a is x and b is decoded
        calls.append("matmul")
        return output

    monkeypatch.setattr(helpers, "torch_dequantize_custom_e4m3", decode)
    monkeypatch.setattr(torch, "matmul", matmul)
    for _ in range(2):
        assert helpers.torch_composite_matmul(x, weight, scale, 128) is output
    assert calls == ["decode", "matmul", "decode", "matmul"]


def test_fp8_torch_baseline_does_not_cache_inputs():
    from ._fp8_gemm_utils import torch_composite_matmul

    device = flag_train.device
    x = torch.ones((1, 2), dtype=torch.bfloat16, device=device)
    weight = torch.tensor([[0, 1], [2, 3]], dtype=torch.uint8, device=device)
    scale = torch.full((2, 1), 1 / 128, dtype=torch.float32, device=device)
    first = torch_composite_matmul(x, weight, scale, 2)
    weight.fill_(127)
    second = torch_composite_matmul(x, weight, scale, 2)
    scale.mul_(2)
    third = torch_composite_matmul(x, weight, scale, 2)
    assert not torch.equal(first, second)
    utils.train_assert_equal(second.cpu(), torch.full((1, 2), 7.5, dtype=x.dtype))
    utils.train_assert_equal(third.cpu(), torch.full((1, 2), 15.0, dtype=x.dtype))


@pytest.mark.parametrize("m", (1, 17, 1024))
def test_fp8_gemm_does_not_cache_inputs(m):
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
        actual = matmul_fp8_triton(x, weight, scale, 128)
        utils.train_assert_equal(
            actual.cpu(), torch.full((m, 128), expected, dtype=x.dtype)
        )


@pytest.fixture(scope="module", autouse=True)
def fp8_environment():
    if not utils.bf16_is_supported:
        pytest.skip("The active backend does not support BF16")
    with strict_matmul_context():
        yield


@pytest.fixture(scope="module")
def selected_baseline():
    # Explicit Torch baselines must not import DeepSpeed. Other vendors
    # still require it; missing/build-broken baselines never imply a pass.
    metadata = baseline_metadata()
    print("FP8_GEMM_BASELINE " + json.dumps(metadata, sort_keys=True))
    return metadata


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
    x, weight, scale = make_portable_inputs(
        m, CORE_K, CORE_N, dtype, CORE_GROUP_SIZE, flag_train.device
    )
    reference = torch_reference(x, weight, scale, CORE_GROUP_SIZE)
    return x, weight, scale, reference


def test_fp8_gemm_candidate_vs_torch(core_case, record_property):
    """All 15 BF16 candidate shapes run, including the four baseline skips."""
    x, weight, scale, reference = core_case
    record_property("shape_group", "large-m" if x.shape[0] in LARGE_M else "base")
    actual = matmul_fp8_triton(x, weight, scale, CORE_GROUP_SIZE)
    assert_fp8_close(actual, reference)


@pytest.mark.parametrize("m", CORE_M)
def test_fp8_gemm_against_baseline(m, selected_baseline, record_property):
    """Compare against the explicit platform baseline and an independent oracle.

    ``original_deepspeed_skip_reason`` is deliberately consulted before the
    DeepSpeed call.  It covers only the pinned upstream resource limit;
    the candidate has already been checked against the independent oracle above.
    """
    dtype = torch.bfloat16
    record_property("shape_group", "large-m" if m in LARGE_M else "base")
    reason = original_deepspeed_skip_reason(
        m,
        CORE_K,
        CORE_N,
        dtype,
        CORE_GROUP_SIZE,
        flag_train.device,
        metadata=selected_baseline,
    )
    if reason:
        pytest.skip(reason)
    record_property("baseline_mode", selected_baseline["mode"])
    record_property("baseline_metadata", json.dumps(selected_baseline, sort_keys=True))
    x, weight, scale, fpq = make_baseline_inputs(
        m, CORE_K, CORE_N, dtype, CORE_GROUP_SIZE, flag_train.device
    )
    if fpq is None:
        reference = torch_reference(x, weight, scale, CORE_GROUP_SIZE)
    else:
        decoded = fpq.dequantize(weight, scale=fpq.scale)
        mathematical = decode_custom_e4m3(
            utils.to_reference(weight),
            utils.to_reference(scale),
            CORE_GROUP_SIZE,
            dtype,
        )
        utils.train_assert_equal(mathematical, utils.to_reference(decoded))
        reference = torch.matmul(utils.to_reference(x), utils.to_reference(decoded))
    actual = matmul_fp8_triton(x, weight, scale, CORE_GROUP_SIZE)
    baseline = baseline_matmul(x, weight, scale, CORE_GROUP_SIZE, fpq)
    assert_fp8_close(actual, reference)
    assert_fp8_close(baseline, reference)
    assert_fp8_close(actual, baseline, mean_relative=False)


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


def test_fp8_gemm_oracle_contract():
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
def test_fp8_gemm_edges(name, m, k, n, group_size, dtype):
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
    actual = matmul_fp8_triton(x, weight, scales, group_size)
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
def test_fp8_gemm_invalid_inputs(case, error):
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
        matmul_fp8_triton(x, weight, scale, group)


@pytest.mark.parametrize("vendor", ("ascend", "hygon", "thead", "metax"))
def test_fp8_gemm_h20_skip_is_nvidia_only(monkeypatch, vendor):
    """Policy test, not evidence of execution on the named hardware."""
    monkeypatch.setattr(flag_train, "vendor_name", vendor)
    assert len(BASE_M) == 11 and len(LARGE_M) == 4
    assert set(CORE_M) == set(BASE_M) | set(LARGE_M)
    assert not set(BASE_M) & set(LARGE_M)
    assert (
        original_deepspeed_skip_reason(
            256,
            CORE_K,
            CORE_N,
            torch.bfloat16,
            CORE_GROUP_SIZE,
            flag_train.device,
            metadata={},
        )
        is None
    )


def test_fp8_gemm_portable_path_without_deepspeed(monkeypatch):
    from . import _fp8_gemm_utils as helpers

    def forbidden():
        raise AssertionError("Portable correctness must not load DeepSpeed")

    monkeypatch.setattr(helpers, "load_deepspeed", forbidden)
    x, weight, scale = make_portable_inputs(
        17, 133, 77, torch.bfloat16, 17, flag_train.device
    )
    actual = matmul_fp8_triton(x, weight, scale, 17)
    assert_fp8_close(actual, torch_reference(x, weight, scale, 17))


@pytest.mark.parametrize("vendor", ("ascend", "metax", "hygon"))
def test_fp8_torch_baseline_without_deepspeed(monkeypatch, vendor):
    """Exercise metadata, inputs and the full Torch path without the DS loader."""
    from . import _fp8_gemm_utils as helpers

    def forbidden():
        raise AssertionError("The Torch baseline must not load DeepSpeed")

    monkeypatch.setattr(flag_train, "vendor_name", vendor)
    monkeypatch.setattr(helpers, "load_deepspeed", forbidden)
    metadata = baseline_metadata()
    assert metadata["mode"] == "torch-dequantize-matmul"
    assert metadata["deepspeed_source_sha256"] is None
    x, weight, scale, fpq = make_baseline_inputs(
        17, 133, 77, torch.bfloat16, 17, flag_train.device
    )
    assert fpq is None
    baseline = baseline_matmul(x, weight, scale, 17, fpq)
    assert_fp8_close(baseline, torch_reference(x, weight, scale, 17))
