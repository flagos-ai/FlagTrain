# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Forward-only DeepSpeed custom FP8 GEMM tests, following test_lamb.py.

The mathematical torch oracle is independent of the production bitcast decoder.
Every core candidate case runs against that oracle.  The separate original
DeepSpeed comparison uses the unmodified public API and skips only the pinned,
known-broken upstream cases before that API is called; it never skips a
candidate/oracle check.

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
    CORE_GROUP_SIZE,
    CORE_K,
    CORE_M,
    CORE_N,
    assert_fp8_close,
    baseline_metadata,
    decode_custom_e4m3,
    deepspeed_matmul,
    make_quantized_inputs,
    original_deepspeed_skip_reason,
    strict_matmul_context,
    torch_reference,
)

pytestmark = pytest.mark.fp8_gemm
DTYPES = (torch.bfloat16,)


@pytest.fixture(scope="module", autouse=True)
def fp8_environment():
    if flag_train.vendor_name != "nvidia":
        pytest.skip("FP8 GEMM integration currently targets NVIDIA only")
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA FP8 GEMM tests require an available CUDA device")
    metadata = baseline_metadata()  # required dependency: failures are not skipped
    print("FP8_GEMM_BASELINE " + json.dumps(metadata, sort_keys=True))
    with strict_matmul_context():
        yield metadata


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
    x, weight, scale, fpq = make_quantized_inputs(
        m, CORE_K, CORE_N, dtype, CORE_GROUP_SIZE, flag_train.device
    )
    # Also pin the arithmetic oracle to DeepSpeed's independent CUDA decoder.
    # BF16 uses the exact dequantize + torch.matmul reference from upstream.
    decoded = fpq.dequantize(weight, scale=fpq.scale)
    reference = torch.matmul(x, decoded)
    mathematical = decode_custom_e4m3(weight, scale, CORE_GROUP_SIZE, dtype)
    utils.train_assert_equal(
        utils.to_reference(mathematical), utils.to_reference(decoded)
    )
    del mathematical, decoded
    return m, dtype, x, weight, scale, fpq, reference


def test_fp8_gemm_candidate_vs_torch(core_case):
    """All 15 BF16 candidate shapes run, including the four baseline skips."""
    _, _, x, weight, scale, _, reference = core_case
    actual = matmul_fp8_triton(x, weight, scale, CORE_GROUP_SIZE)
    assert_fp8_close(actual, reference)


def test_fp8_gemm_against_original_deepspeed(
    core_case, fp8_environment, record_property
):
    """Compare to unmodified DeepSpeed where its pinned implementation runs.

    ``original_deepspeed_skip_reason`` is deliberately consulted before the
    DeepSpeed call.  It covers only the pinned upstream resource limit;
    the candidate has already been checked against the independent oracle above.
    """
    m, dtype, x, weight, scale, fpq, reference = core_case
    reason = original_deepspeed_skip_reason(
        m,
        CORE_K,
        CORE_N,
        dtype,
        CORE_GROUP_SIZE,
        x.device,
        metadata=fp8_environment,
    )
    if reason:
        pytest.skip(reason)
    record_property("baseline_mode", "original")
    record_property("baseline_metadata", json.dumps(fp8_environment, sort_keys=True))
    actual = matmul_fp8_triton(x, weight, scale, CORE_GROUP_SIZE)
    original = deepspeed_matmul(x, weight, scale, CORE_GROUP_SIZE, fpq)
    assert_fp8_close(actual, reference)
    assert_fp8_close(original, reference)
    assert_fp8_close(actual, original, mean_relative=False)


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
    ("empty_m", 0, 17, 19, 7),
    ("empty_n", 3, 17, 0, 7),
    ("empty_k", 3, 0, 19, 7),
)


def test_fp8_gemm_oracle_contract():
    codes = torch.arange(256, device=flag_train.device).to(torch.uint8).reshape(1, 256)
    scale = torch.tensor([1 / 128], dtype=torch.float32, device=flag_train.device)
    values = decode_custom_e4m3(codes, scale, 256, torch.float64).reshape(-1)
    assert values[0].item() == math.ldexp(1.0, -14)  # not zero/subnormal
    assert values[120].item() == 2.0  # exponent 15 is finite
    assert values[127].item() == 3.75
    assert values[255].item() == -3.75
    assert bool(torch.isfinite(values).all())
    utils.train_assert_equal(values[:128], -values[128:])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "name,m,k,n,group_size", EDGE_CASES, ids=[c[0] for c in EDGE_CASES]
)
def test_fp8_gemm_edges(name, m, k, n, group_size, dtype):
    device = flag_train.device
    generator = torch.Generator(device=device).manual_seed(20260922)
    shape = (k, m) if name == "noncontiguous_activation" else (m, k)
    x = (torch.randn(shape, device=device, generator=generator) * 0.5).to(dtype)
    if name == "noncontiguous_activation":
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
    reference = torch_reference(x, weight, scales, group_size)
    actual = matmul_fp8_triton(x, weight, scales, group_size)
    if name in ("all_256_codes", "zero_scale", "empty_m", "empty_n", "empty_k"):
        utils.train_assert_equal(
            utils.to_reference(actual), utils.to_reference(reference)
        )
    elif name == "large_decoded_range":
        # Retain the predeclared standalone large-range policy, not an observed
        # error-derived tolerance. Atol grows with the reference RMS only here.
        rms = reference.double().square().mean().sqrt().item()
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
