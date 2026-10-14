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

import importlib

import pytest
import torch

import flag_train
from flag_train.deepspeed import loco_quantized_reduction
from flag_train.deepspeed.loco_quantized_reduction import QuantizationType

from .. import accuracy_utils as utils

SEED = 0

CASES = (
    [(1024, 4096, 8)]
    if utils.QUICK_MODE
    else [
        (1024, 4096, 1),
        (1024, 4096, 2),
        (1024, 4096, 8),
        (2048, 2048, 8),
        (512, 2048, 8),
        (256, 1024, 8),
        (256, 1024, 16),
    ]
)

NOALIGN_CASES = (
    [(1024, 3000, 8)] if utils.QUICK_MODE else [(1024, 3000, 8), (1024, 5000, 2)]
)

LARGE_CASES = [] if utils.QUICK_MODE else [(1024, 8192, 8), (512, 8192, 2)]

MULTIPASS_CASES = (
    [(512, 16384, 8)]
    if utils.QUICK_MODE
    else [
        (512, 16384, 2),
        (512, 16384, 8),
        (256, 10000, 2),
        (256, 49152, 8),
    ]
)

ND_CASES = [((512, 1024), 16, 8), ((64, 96), 4, 4)]

BITS = [4, 8]

HAND_OUT_GROUPS = 64
HAND_EPOG = 256
HAND_STEP = 0.25
HAND_OFF = 0.1
HAND_NT = [8] if utils.QUICK_MODE else [1, 8]

HAND_ASYM_MT_NT = 8
HAND_ASYM_MT_FIRST = 0x78
HAND_ASYM_MT_REST = 0x88

_DEEPSPEED_BASELINE_VENDORS = {"nvidia", "hygon"}


def _load_deepspeed_reduce():
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None, None, None

    try:
        import deepspeed
        from deepspeed.ops.op_builder import QuantizerBuilder

        mod = QuantizerBuilder().load()
        return mod.quantized_reduction, mod.QuantizationType, deepspeed.__version__
    except Exception as exc:
        raise RuntimeError(
            f"{flag_train.vendor_name!r} must use DeepSpeed's quantized_reduction "
            f"as its reference, but it could not be loaded: {exc!r}. Build "
            f"deepspeed, or drop the backend from _DEEPSPEED_BASELINE_VENDORS."
        ) from exc


_deepspeed_reduce, _deepspeed_quant_type, _DEEPSPEED_VERSION = _load_deepspeed_reduce()

requires_deepspeed_reference = pytest.mark.skipif(
    _deepspeed_reduce is None,
    reason=(
        "DeepSpeed's quantized_reduction reference is unavailable; a CUDA "
        "op_builder is required to host it."
    ),
)


def _ds_call(data, scales, in_groups, out_groups, num_bits, symmetric, num_tensors):
    return _deepspeed_reduce(
        data,
        scales,
        in_groups,
        out_groups,
        num_bits,
        (
            _deepspeed_quant_type.Symmetric
            if symmetric
            else _deepspeed_quant_type.Asymmetric
        ),
        num_tensors,
    )


def _ours_call(data, scales, in_groups, out_groups, num_bits, symmetric, num_tensors):
    return loco_quantized_reduction(
        data,
        scales,
        in_groups,
        out_groups,
        num_bits,
        QuantizationType.Symmetric if symmetric else QuantizationType.Asymmetric,
        num_tensors,
    )


def _unpack(b, num_bits):
    x = b.to(torch.int32) & 0xFF
    if num_bits == 8:
        return torch.where(x >= 128, x - 256, x)

    hi = (x >> 4) & 0xF
    lo = x & 0xF
    hi = torch.where(hi >= 8, hi - 16, hi)
    lo = torch.where(lo >= 8, lo - 16, lo)
    return torch.stack([hi, lo], dim=-1).reshape(*b.shape[:-1], b.shape[-1] * 2)


def _pack(e, num_bits):
    if num_bits == 8:
        return (e.to(torch.int32) & 0xFF).to(torch.int8)

    e = e.to(torch.int32) & 0xF
    return ((e[..., 0::2] << 4) | e[..., 1::2]).to(torch.int8)


def _reference(data, scales, in_groups, out_groups, num_bits, symmetric, num_tensors):
    dev = data.device
    dref = utils.to_reference(data)
    sref = utils.to_reference(scales)

    epit = dref.numel() // num_tensors
    epog = epit // out_groups
    gpit = in_groups // num_tensors

    q = _unpack(dref.reshape(num_tensors, out_groups, epog), num_bits).float()

    if symmetric:
        s = sref.reshape(num_tensors, gpit)[:, :, None]
        acc = (q * s).sum(0)
        amax = acc.abs().amax(-1)
        qscale = torch.where(
            amax == 0, torch.ones_like(amax), (2**num_bits) / (2.0 * amax)
        )
        off = torch.zeros_like(qscale)
    else:
        sc = sref.reshape(num_tensors, gpit, 2)
        s, o = sc[:, :, 0][:, :, None], sc[:, :, 1][:, :, None]
        acc = (q * s + o).sum(0)
        mx, mn = acc.amax(-1), acc.amin(-1)
        rng = mx - mn
        qscale = torch.where(rng == 0, torch.ones_like(rng), (2**num_bits) / rng)
        off = (mx + mn) / 2

    step = 1.0 / qscale
    lo_q, hi_q = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1
    qout = torch.round((acc - off[..., None]) * qscale[..., None]).clamp(lo_q, hi_q)

    packed = _pack(qout.to(torch.int32), num_bits).reshape(-1).to(dev)
    return packed, step.to(dev), off.to(dev)


def _diff(got, ref, num_bits):
    d = (_unpack(got.reshape(-1), num_bits) - _unpack(ref.reshape(-1), num_bits)).abs()
    return int((d > 0).sum()), int(d.max()), int((d > 1).sum())


def _step_tolerance(step):
    return 1e-3 * float(step.abs().median().clamp_min(1e-30))


def _inputs(out_groups, bpg, num_tensors, num_bits, symmetric, seed=SEED):
    in_groups = out_groups * num_tensors
    epit = out_groups * bpg
    scales_elems = 1 if symmetric else 2

    utils.init_seed(seed)

    data = torch.randint(-128, 128, (num_tensors * epit,), dtype=torch.int8).to(
        flag_train.device
    )
    raw = (
        torch.rand((in_groups * scales_elems,), dtype=torch.float32)
        .mul_(0.05)
        .add_(1e-3)
    )
    if not symmetric:
        raw = raw.view(-1, 2)
        raw = torch.stack([raw[:, 0], raw[:, 1] - 0.026], dim=1).reshape(-1)

    return data, raw.contiguous().to(flag_train.device), in_groups


def _const_bytes(byte, num_tensors, out_groups, epog):
    signed = byte - 256 if byte >= 128 else byte
    return torch.full(
        (num_tensors * out_groups * epog,),
        signed,
        dtype=torch.int8,
        device=flag_train.device,
    )


def _sym_scales(value, num_tensors, out_groups):
    return torch.full(
        (num_tensors * out_groups,),
        value,
        dtype=torch.float32,
        device=flag_train.device,
    )


def _asym_scales(step, off, num_tensors, out_groups):
    s = torch.empty(
        (num_tensors * out_groups * 2,),
        dtype=torch.float32,
        device=flag_train.device,
    )
    s[0::2] = step
    s[1::2] = off
    return s


@pytest.mark.loco_quantized_reduction
@pytest.mark.parametrize("num_bits", BITS)
@pytest.mark.parametrize("case", CASES)
def test_loco_quantized_reduction(case, num_bits):
    out_groups, bpg, num_tensors = case
    data, scales, in_groups = _inputs(out_groups, bpg, num_tensors, num_bits, True)
    data_ref = utils.to_reference(data).clone()
    scales_ref = utils.to_reference(scales).clone()

    res = _ours_call(data, scales, in_groups, out_groups, num_bits, True, num_tensors)

    assert isinstance(res, list) and len(res) == 2
    out, out_st = res

    assert out.shape[:-1] == data.shape[:-1]
    assert out.shape[-1] == data.shape[-1] // num_tensors
    assert out.dtype == torch.int8
    assert out_st.shape == (out_groups, 1)
    assert out_st.dtype == torch.float32

    assert out.data_ptr() != data.data_ptr()
    utils.train_assert_equal(data, data_ref)
    utils.train_assert_equal(scales, scales_ref)

    q = _unpack(out, num_bits)
    assert int(q.min()) >= -(2 ** (num_bits - 1))
    assert int(q.max()) <= 2 ** (num_bits - 1) - 1
    assert torch.isfinite(out_st).all() and (out_st > 0).all()


@pytest.mark.loco_quantized_reduction
@pytest.mark.parametrize("num_tensors", HAND_NT)
@pytest.mark.parametrize("num_bits", BITS)
def test_loco_quantized_reduction_hand_amax_zero(num_bits, num_tensors):
    data = _const_bytes(0x00, num_tensors, HAND_OUT_GROUPS, HAND_EPOG)
    scales = _sym_scales(HAND_STEP, num_tensors, HAND_OUT_GROUPS)

    out, out_st = _ours_call(
        data,
        scales,
        num_tensors * HAND_OUT_GROUPS,
        HAND_OUT_GROUPS,
        num_bits,
        True,
        num_tensors,
    )

    got = (out.reshape(-1).to(torch.int32) & 0xFF).to(torch.int32)
    utils.train_assert_equal(got, torch.zeros_like(got))
    utils.train_assert_equal(out_st, torch.ones_like(out_st))


@pytest.mark.loco_quantized_reduction
@pytest.mark.parametrize("num_bits,byte", [(4, 0x11), (8, 0x01)])
def test_loco_quantized_reduction_hand_asym_max_eq_min(num_bits, byte):
    nt = 4
    data = _const_bytes(byte, nt, HAND_OUT_GROUPS, HAND_EPOG)
    scales = _asym_scales(HAND_STEP, HAND_OFF, nt, HAND_OUT_GROUPS)

    out, out_st = _ours_call(
        data, scales, nt * HAND_OUT_GROUPS, HAND_OUT_GROUPS, num_bits, False, nt
    )

    got = (out.reshape(-1).to(torch.int32) & 0xFF).to(torch.int32)
    utils.train_assert_equal(got, torch.zeros_like(got))
    utils.train_assert_equal(out_st[:, 0], torch.ones_like(out_st[:, 0]))

    expect_off = nt * (1 * HAND_STEP + HAND_OFF)
    utils.train_assert_close(
        out_st[:, 1],
        torch.full_like(out_st[:, 1], expect_off),
        torch.float32,
        atol=_step_tolerance(out_st[:, 0]),
    )


@pytest.mark.loco_quantized_reduction
def test_loco_quantized_reduction_hand_asym_multi_tensor():
    nt = HAND_ASYM_MT_NT
    og, epog = HAND_OUT_GROUPS, HAND_EPOG
    epit = og * epog

    data = torch.zeros((nt * epit,), dtype=torch.int8, device=flag_train.device)
    for j in range(nt):
        for g in range(og):
            data[j * epit + g * epog] = 0x10

    scales = _asym_scales(HAND_STEP, HAND_OFF, nt, og)
    out, out_st = _ours_call(data, scales, nt * og, og, 4, False, nt)

    got = (out.reshape(-1).to(torch.int32) & 0xFF).reshape(og, epog)
    expected = torch.full(
        (og, epog), HAND_ASYM_MT_REST, dtype=torch.int32, device=got.device
    )
    expected[:, 0] = HAND_ASYM_MT_FIRST
    utils.train_assert_equal(got, expected)

    atol = _step_tolerance(out_st[:, 0])
    utils.train_assert_close(
        out_st[:, 0],
        torch.full_like(out_st[:, 0], nt * HAND_STEP / 16),
        torch.float32,
        atol=atol,
    )
    utils.train_assert_close(
        out_st[:, 1],
        torch.full_like(out_st[:, 1], nt * (HAND_STEP + 2 * HAND_OFF) / 2),
        torch.float32,
        atol=atol,
    )


@pytest.mark.loco_quantized_reduction
def test_loco_quantized_reduction_multipass_equals_singlepass():
    lqr = importlib.import_module("flag_train.deepspeed.loco_quantized_reduction")

    out_groups, bpg, num_tensors = 512, 8192, 8
    data, scales, in_groups = _inputs(out_groups, bpg, num_tensors, 4, True)

    single, s1 = _ours_call(data, scales, in_groups, out_groups, 4, True, num_tensors)

    old = lqr._BLOCK_MAX
    try:
        lqr._BLOCK_MAX = 4096
        multi, s2 = _ours_call(
            data, scales, in_groups, out_groups, 4, True, num_tensors
        )
    finally:
        lqr._BLOCK_MAX = old

    assert multi.shape == single.shape
    utils.train_assert_equal(multi, single)
    utils.train_assert_equal(s2, s1)


@pytest.mark.loco_quantized_reduction
@pytest.mark.parametrize("num_bits", BITS)
@pytest.mark.parametrize("case", MULTIPASS_CASES)
def test_loco_quantized_reduction_multipass_matches_reference(case, num_bits):
    out_groups, bpg, num_tensors = case
    data, scales, in_groups = _inputs(out_groups, bpg, num_tensors, num_bits, True)

    ref, ref_step, _ref_off = _reference(
        data, scales, in_groups, out_groups, num_bits, True, num_tensors
    )
    out, out_st = _ours_call(
        data, scales, in_groups, out_groups, num_bits, True, num_tensors
    )

    n_diff, dmax, n_gt1 = _diff(out, ref, num_bits)
    n_elem = out.numel() * (8 // num_bits)

    assert n_gt1 == 0, f"bpg={bpg}: {n_gt1} elements differ by more than 1 level"
    assert n_diff / n_elem < 1e-4, f"bpg={bpg}: diff rate {n_diff / n_elem:.2e}"

    utils.train_assert_close(
        out_st[:, 0], ref_step, torch.float32, atol=_step_tolerance(ref_step)
    )


@requires_deepspeed_reference
@pytest.mark.loco_quantized_reduction
@pytest.mark.parametrize("num_bits", BITS)
@pytest.mark.parametrize("case", MULTIPASS_CASES)
def test_loco_quantized_reduction_multipass_matches_deepspeed(case, num_bits):
    out_groups, bpg, num_tensors = case
    data, scales, in_groups = _inputs(out_groups, bpg, num_tensors, num_bits, True)

    ds_out, _ = _ds_call(
        data, scales, in_groups, out_groups, num_bits, True, num_tensors
    )
    out, _ = _ours_call(
        data, scales, in_groups, out_groups, num_bits, True, num_tensors
    )

    assert out.shape == ds_out.shape
    n_diff, dmax, n_gt1 = _diff(out, ds_out, num_bits)
    n_elem = out.numel() * (8 // num_bits)

    assert n_gt1 == 0, f"bpg={bpg}: {n_gt1} elements differ by more than 1 level"
    assert n_diff / n_elem < (5e-2 if num_bits == 4 else 0.3)


@pytest.mark.loco_quantized_reduction
def test_loco_quantized_reduction_multipass_amax_zero():
    out_groups, bpg, num_tensors = 256, 16384, 4
    data = _const_bytes(0x00, num_tensors, out_groups, bpg)
    scales = _sym_scales(HAND_STEP, num_tensors, out_groups)

    out, out_st = _ours_call(
        data, scales, num_tensors * out_groups, out_groups, 4, True, num_tensors
    )

    got = (out.reshape(-1).to(torch.int32) & 0xFF).to(torch.int32)
    utils.train_assert_equal(got, torch.zeros_like(got))
    utils.train_assert_equal(out_st, torch.ones_like(out_st))


@pytest.mark.loco_quantized_reduction
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("num_bits", BITS)
@pytest.mark.parametrize("case", CASES)
def test_loco_quantized_reduction_matches_reference(case, num_bits, symmetric):
    out_groups, bpg, num_tensors = case
    data, scales, in_groups = _inputs(out_groups, bpg, num_tensors, num_bits, symmetric)

    ref, ref_step, ref_off = _reference(
        data, scales, in_groups, out_groups, num_bits, symmetric, num_tensors
    )
    out, out_st = _ours_call(
        data, scales, in_groups, out_groups, num_bits, symmetric, num_tensors
    )

    n_diff, dmax, n_gt1 = _diff(out, ref, num_bits)
    n_elem = out.numel() * (8 // num_bits)

    assert n_gt1 == 0, f"{n_gt1} elements differ by more than 1 level, max {dmax}"
    assert n_diff / n_elem < 1e-4, f"diff rate {n_diff / n_elem:.2e}"

    atol = _step_tolerance(ref_step)
    utils.train_assert_close(out_st[:, 0], ref_step, torch.float32, atol=atol)
    if not symmetric:
        utils.train_assert_close(out_st[:, 1], ref_off, torch.float32, atol=atol)


@requires_deepspeed_reference
@pytest.mark.loco_quantized_reduction
@pytest.mark.parametrize("num_bits", BITS)
@pytest.mark.parametrize("case", CASES)
def test_loco_quantized_reduction_matches_deepspeed(case, num_bits):
    out_groups, bpg, num_tensors = case
    data, scales, in_groups = _inputs(out_groups, bpg, num_tensors, num_bits, True)

    ds_out, ds_st = _ds_call(
        data, scales, in_groups, out_groups, num_bits, True, num_tensors
    )
    out, out_st = _ours_call(
        data, scales, in_groups, out_groups, num_bits, True, num_tensors
    )

    assert out.shape == ds_out.shape

    n_diff, dmax, n_gt1 = _diff(out, ds_out, num_bits)
    n_elem = out.numel() * (8 // num_bits)

    assert n_gt1 == 0, f"{n_gt1} elements differ by more than 1 level, max {dmax}"
    assert n_diff / n_elem < (
        5e-2 if num_bits == 4 else 0.3
    ), f"diff rate {n_diff / n_elem:.2e}"

    ds_st = ds_st.reshape(out_groups, -1)
    utils.train_assert_close(
        out_st[:, 0],
        ds_st[:, 0],
        torch.float32,
        atol=1e-2 * float(ds_st[:, 0].abs().median()),
    )


@requires_deepspeed_reference
@pytest.mark.loco_quantized_reduction
@pytest.mark.parametrize("num_bits", BITS)
def test_loco_quantized_reduction_asymmetric_single_tensor_matches_deepspeed(num_bits):
    out_groups, bpg, num_tensors = 1024, 4096, 1
    data, scales, in_groups = _inputs(out_groups, bpg, num_tensors, num_bits, False)

    ds_out, _ = _ds_call(
        data, scales, in_groups, out_groups, num_bits, False, num_tensors
    )
    out, _ = _ours_call(
        data, scales, in_groups, out_groups, num_bits, False, num_tensors
    )

    _, dmax, n_gt1 = _diff(out, ds_out, num_bits)
    assert n_gt1 == 0 and dmax <= 1, f"max diff {dmax}"


@requires_deepspeed_reference
@pytest.mark.loco_quantized_reduction
@pytest.mark.parametrize("num_bits", BITS)
def test_loco_quantized_reduction_asymmetric_multi_tensor_differs_by_design(num_bits):
    out_groups, bpg, num_tensors = 1024, 4096, 8
    data, scales, in_groups = _inputs(out_groups, bpg, num_tensors, num_bits, False)

    ds_out, _ = _ds_call(
        data, scales, in_groups, out_groups, num_bits, False, num_tensors
    )
    out, _ = _ours_call(
        data, scales, in_groups, out_groups, num_bits, False, num_tensors
    )

    _, dmax, _ = _diff(out, ds_out, num_bits)
    assert dmax > 1, "outputs match upstream, check whether the off-by-2 was copied"


@pytest.mark.loco_quantized_reduction
@pytest.mark.parametrize("num_bits", BITS)
def test_loco_quantized_reduction_noalign(num_bits):
    for case in NOALIGN_CASES:
        out_groups, bpg, num_tensors = case
        data, scales, in_groups = _inputs(out_groups, bpg, num_tensors, num_bits, True)
        ref, _, _ = _reference(
            data, scales, in_groups, out_groups, num_bits, True, num_tensors
        )
        out, _ = _ours_call(
            data, scales, in_groups, out_groups, num_bits, True, num_tensors
        )
        _, dmax, n_gt1 = _diff(out, ref, num_bits)
        assert n_gt1 == 0, f"bpg={bpg}: max diff {dmax}"


@requires_deepspeed_reference
@pytest.mark.loco_quantized_reduction
@pytest.mark.parametrize("num_bits", BITS)
@pytest.mark.parametrize("case", LARGE_CASES)
def test_loco_quantized_reduction_large_groups(case, num_bits):
    out_groups, bpg, num_tensors = case
    data, scales, in_groups = _inputs(out_groups, bpg, num_tensors, num_bits, True)

    ds_out, _ = _ds_call(
        data, scales, in_groups, out_groups, num_bits, True, num_tensors
    )
    out, _ = _ours_call(
        data, scales, in_groups, out_groups, num_bits, True, num_tensors
    )

    _, dmax, n_gt1 = _diff(out, ds_out, num_bits)
    assert n_gt1 == 0, f"bpg={bpg}: max diff {dmax}"


@pytest.mark.loco_quantized_reduction
def test_loco_quantized_reduction_oversize_group_now_works():
    out_groups, bpg, num_tensors = 64, 16384, 8
    data, scales, in_groups = _inputs(out_groups, bpg, num_tensors, 4, True)

    ref, _, _ = _reference(data, scales, in_groups, out_groups, 4, True, num_tensors)
    out, _ = _ours_call(data, scales, in_groups, out_groups, 4, True, num_tensors)

    _, dmax, n_gt1 = _diff(out, ref, 4)
    assert n_gt1 == 0, f"max diff {dmax}"


@pytest.mark.loco_quantized_reduction
@pytest.mark.parametrize("shape,out_groups,num_tensors", ND_CASES)
def test_loco_quantized_reduction_nd_input(shape, out_groups, num_tensors):
    in_groups = out_groups * num_tensors

    data = torch.randint(-128, 128, shape, dtype=torch.int8, device=flag_train.device)
    scales = (
        torch.rand((in_groups,), dtype=torch.float32, device=flag_train.device) + 1e-3
    )

    out, out_st = loco_quantized_reduction(
        data,
        scales,
        in_groups,
        out_groups,
        4,
        QuantizationType.Symmetric,
        num_tensors,
    )

    assert out.shape == (shape[0], shape[-1] // num_tensors)
    assert out_st.shape == (out_groups, 1)

    ref, _, _ = _reference(data, scales, in_groups, out_groups, 4, True, num_tensors)
    _, dmax, n_gt1 = _diff(out, ref, 4)
    assert n_gt1 == 0, f"shape={shape}: max diff {dmax}"


@pytest.mark.loco_quantized_reduction
@pytest.mark.parametrize(
    "kind,exc,match",
    [
        ("int32_data", TypeError, "must be int8"),
        ("fp16_scales", TypeError, "must be float32"),
        ("noncontig_data", ValueError, "input_vals must be contiguous"),
        ("noncontig_scales", ValueError, "input_scales must be contiguous"),
        ("short_scales", ValueError, r"input_scales\.numel"),
        ("zero_out_groups", ValueError, "out_groups must be"),
        ("bad_num_bits", ValueError, "num_bits must be"),
        ("zero_devices", ValueError, "devices_per_node must be"),
    ],
)
def test_loco_quantized_reduction_rejects_bad_args(kind, exc, match):
    out_groups, bpg, num_tensors = 128, 1024, 4
    in_groups = out_groups * num_tensors
    epit = out_groups * bpg

    data = torch.randint(
        -128, 128, (num_tensors * epit,), dtype=torch.int8, device=flag_train.device
    )
    scales = (
        torch.rand((in_groups,), dtype=torch.float32, device=flag_train.device) + 1e-3
    )
    kwargs = dict(
        in_groups=in_groups,
        out_groups=out_groups,
        num_bits=4,
        quant_type=QuantizationType.Symmetric,
        devices_per_node=num_tensors,
    )

    if kind == "int32_data":
        data = data.to(torch.int32)
    elif kind == "fp16_scales":
        scales = scales.half()
    elif kind == "noncontig_data":
        data = torch.randint(
            -128,
            128,
            (2 * num_tensors * epit,),
            dtype=torch.int8,
            device=flag_train.device,
        )[::2]
    elif kind == "noncontig_scales":
        scales = (
            torch.rand((2 * in_groups,), dtype=torch.float32, device=flag_train.device)
            + 1e-3
        )[::2]
    elif kind == "short_scales":
        scales = scales[: in_groups // 2].contiguous()
    elif kind == "zero_out_groups":
        kwargs["out_groups"] = 0
        kwargs["in_groups"] = 0
    elif kind == "bad_num_bits":
        kwargs["num_bits"] = 3
    elif kind == "zero_devices":
        kwargs["devices_per_node"] = 0

    with pytest.raises(exc, match=match):
        loco_quantized_reduction(data, scales, **kwargs)


@pytest.mark.loco_quantized_reduction
def test_loco_quantized_reduction_deterministic():
    out_groups, bpg, num_tensors = 512, 4096, 8
    data, scales, in_groups = _inputs(out_groups, bpg, num_tensors, 4, True)

    a, sa = _ours_call(data, scales, in_groups, out_groups, 4, True, num_tensors)
    b, sb = _ours_call(data, scales, in_groups, out_groups, 4, True, num_tensors)

    utils.train_assert_equal(a, b)
    utils.train_assert_equal(sa, sb)


@pytest.mark.loco_quantized_reduction
def test_loco_quantized_reduction_rejects_invalid_group_count():
    data, scales, _ = _inputs(1024, 4096, 8, 4, True)
    with pytest.raises(ValueError):
        _ours_call(data, scales, 1024 * 8 + 1, 1024, 4, True, 8)


@pytest.mark.loco_quantized_reduction
def test_loco_quantized_reduction_rejects_unaligned_last_dim():
    shape = (3, 4)
    num_tensors, out_groups = 3, 2
    in_groups = out_groups * num_tensors
    total = shape[0] * shape[-1]

    assert total % num_tensors == 0
    assert shape[-1] % num_tensors != 0

    data = torch.randint(-128, 128, shape, dtype=torch.int8, device=flag_train.device)
    scales = (
        torch.rand((in_groups,), dtype=torch.float32, device=flag_train.device) + 1e-3
    )

    with pytest.raises(ValueError, match="last dim"):
        loco_quantized_reduction(
            data,
            scales,
            in_groups,
            out_groups,
            4,
            QuantizationType.Symmetric,
            num_tensors,
        )


@requires_deepspeed_reference
@pytest.mark.loco_quantized_reduction
def test_loco_quantized_reduction_accepts_upstream_enum():
    out_groups, bpg, num_tensors = 1024, 4096, 8
    data, scales, in_groups = _inputs(out_groups, bpg, num_tensors, 4, True)

    a, _ = loco_quantized_reduction(
        data, scales, in_groups, out_groups, 4, QuantizationType.Symmetric, num_tensors
    )
    b, _ = loco_quantized_reduction(
        data, scales, in_groups, out_groups, 4, "symmetric", num_tensors
    )
    c, _ = loco_quantized_reduction(
        data,
        scales,
        in_groups,
        out_groups,
        4,
        _deepspeed_quant_type.Symmetric,
        num_tensors,
    )

    utils.train_assert_equal(a, b)
    utils.train_assert_equal(a, c)
