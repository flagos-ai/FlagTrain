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
"""Correctness tests for the fused output RMSNorm/SiLU gating operator.

The operator is forward-only; gradients are out of scope.

The oracle follows ``_MCORE_ORACLE_VENDORS`` below: on NVIDIA, when the installed
``megatron-core`` provides ``megatron.core.fusions.fused_gated_norm``, that
upstream kernel is the oracle -- it is the implementation this port is a port
*of*, so agreement with it is the contract.  Elsewhere, whenever the module is
absent, and under ``--ref cpu`` (the upstream kernel cannot run on a CPU), the
oracle is ``fused_gated_norm_ref``, a plain-torch composition of the same
contract.

``test_mcore_fused_gated_norm_forward`` holds the upstream oracle itself to that
torch composition, so the oracle is checked rather than merely trusted; it skips
on platforms where megatron-core's operator is unavailable.

The reference mirrors the kernel's *intermediate rounding*: the normalised
activation is cast to ``x.dtype`` before the FP32 gating multiply.  That is the
behaviour being pinned -- the unfused ``RMSNorm -> x * silu(gate)`` pipeline
materialises ``RMSNorm`` in the activation dtype, and the fusion must not change
the numerics.
"""

import warnings

import pytest
import torch

import flag_train
import flag_train.megatron

from .. import accuracy_utils as utils

_EPS = 1e-5

# dim must be a power of two (the kernel addresses it with ``tl.arange(0, D)``).
# Besides ordinary GDN shapes, this covers one row (only the tail lane is valid),
# an odd row count (21 rows leaves a masked tail in the last program), and the
# widest head dimension the kernel supports.
_SHAPES = [
    (1, 1, 1, 64),
    (2, 8, 4, 64),
    (1, 7, 3, 128),
    (1, 32, 8, 128),
    (2, 16, 16, 256),
    (2, 4, 8, 512),
]

# The shape most of the focused tests below use.
_SHAPE = (2, 8, 4, 64)

# The sweep the performance suite reports latency for (see
# ``benchmark/megatron/test_fused_gated_norm.py``). The functional and
# performance suites own their shape lists separately; this one exists so every
# benchmarked shape also has a correctness check. Update it when the benchmark
# sweep changes.
_BENCHMARK_SHAPES = [
    (1, 1, 16, 128),
    (2, 1024, 16, 128),
    (2, 2048, 16, 128),
    (2, 4096, 16, 128),
    (2, 8192, 16, 128),
    (8, 1024, 16, 128),
    (16, 512, 16, 128),
    (2, 1024, 8, 64),
    (2, 1024, 32, 128),
    (2, 1024, 16, 256),
    (2, 512, 16, 512),
]

# The reference (and the upstream operator) target FP16/BF16 activations; FP32 is
# always available and exercises the exact-arithmetic path.
_DTYPES = [torch.float16, torch.float32]
if utils.bf16_is_supported:
    _DTYPES.append(torch.bfloat16)


def fused_gated_norm_ref(x, gate, weight, eps, zero_centered=False):
    """Reference for the operator, composed from plain torch ops.

    Mirrors the kernel's order of operations, including casting the normalised
    activation back to ``x.dtype`` before the FP32 SiLU gate multiply.
    """
    w = weight.to(torch.float32) + (1.0 if zero_centered else 0.0)
    xf = x.to(torch.float32)
    inv = torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) / x.shape[-1] + eps)
    norm = (xf * inv * w).to(x.dtype).to(torch.float32)
    gate_f = gate.to(torch.float32)
    return (norm * (gate_f * torch.sigmoid(gate_f))).to(x.dtype)


# Backends whose oracle is Megatron-LM's fused_gated_norm when the installed
# megatron-core provides it. The upstream kernel is CUDA Triton, so only NVIDIA
# can execute it; elsewhere the torch composition above is the oracle.
_MCORE_ORACLE_VENDORS = {"nvidia"}


def _load_mcore_fused_gated_norm():
    """Megatron-LM's fused_gated_norm, or ``None`` when it is not the oracle.

    Imported from the installed ``megatron-core`` package, so the tests compare
    against whatever upstream revision the platform ships. A missing module is
    not an error: the torch composition is a complete oracle, and the benchmark
    records which baseline it measured.
    """
    if flag_train.vendor_name not in _MCORE_ORACLE_VENDORS:
        return None

    try:
        from megatron.core.fusions.fused_gated_norm import fused_gated_norm

        return fused_gated_norm
    except Exception as exc:
        warnings.warn(
            f"{flag_train.vendor_name!r} can use Megatron-LM's fused_gated_norm "
            f"as its oracle when megatron-core provides it, but the import "
            f"failed: {exc!r}. Falling back to fused_gated_norm_ref.",
            stacklevel=2,
        )
        return None


_mcore_fused_gated_norm = _load_mcore_fused_gated_norm()


def _oracle(x, gate, weight, eps, zero_centered=False):
    """The implementation the port is checked against.

    The upstream kernel cannot run on a CPU, so ``--ref cpu`` keeps the torch
    composition as the oracle even where megatron-core is available; that is the
    independent-oracle mode the flag exists for.
    """
    if _mcore_fused_gated_norm is not None and not utils.TO_CPU:
        return _mcore_fused_gated_norm(x, gate, weight, eps, zero_centered)
    return fused_gated_norm_ref(x, gate, weight, eps, zero_centered)


def _assert_close_to_oracle(res, ref, dtype):
    """Compare an operator result against an independent implementation.

    Both sides round the normalised activation to ``x.dtype`` *before* the FP32
    gating multiply, and their FP32 terms differ in the last bit: sums over the
    same values run in a different order (CPU vs Triton; the upstream kernel
    also ships ``enable_fp_fusion=True``), and sigmoid is not one library's
    implementation. An element sitting on an FP16 rounding boundary can
    therefore land one ULP apart, and the final rounding to the output dtype can
    add another. At ``|out| ~ 9`` two ULP are 1.6e-2, or ~2e-3 relative -- past
    the house ``1e-4 + 1e-3 * |ref|`` band. FP16 is therefore compared at
    ``atol=2e-3, rtol=2.5e-3``, the activation's own granularity; the absolute
    term covers near-zero outputs. BF16's 0.016 relative tolerance already
    covers its much coarser ULP, and FP32 has no intermediate cast at all, so
    both keep the house defaults.
    """
    res = utils.to_reference(res)
    ref = utils.to_reference(ref).to(dtype)
    assert res.dtype == dtype
    if dtype is torch.float16:
        torch.testing.assert_close(res, ref, atol=2e-3, rtol=2.5e-3)
    else:
        utils.train_assert_close(res, ref, dtype)


@pytest.mark.fused_gated_norm
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("zero_centered", [False, True], ids=["norm", "zero_centered"])
def test_fused_gated_norm_forward(shape, dtype, zero_centered):
    """Forward pass must match the oracle, including the rounding."""
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    gate = torch.randn(shape, dtype=dtype, device=flag_train.device)
    weight = torch.randn((shape[-1],), dtype=dtype, device=flag_train.device)

    out = flag_train.megatron.fused_gated_norm(x, gate, weight, _EPS, zero_centered)
    ref_out = _oracle(
        utils.to_reference(x),
        utils.to_reference(gate),
        utils.to_reference(weight),
        _EPS,
        zero_centered,
    )
    _assert_close_to_oracle(out, ref_out, dtype)


# ---------------------------------------------------------------------------
# The shapes the performance suite measures
#
# The tests above stay small so they run quickly and read clearly, while the
# benchmark sweeps long sequences, where the kernel's bandwidth behaviour shows.
# The benchmark's shapes are still correctness-checked here, at the benchmark's
# own dtypes. The benchmark's zero_centered, FP32-gate and strided variants only
# add modes -- covered above -- and their shapes are a subset of this list.
# ---------------------------------------------------------------------------

_BENCHMARK_DTYPES = [torch.float16] + (
    [torch.bfloat16] if utils.bf16_is_supported else []
)


@pytest.mark.fused_gated_norm
@pytest.mark.parametrize("shape", _BENCHMARK_SHAPES)
@pytest.mark.parametrize("dtype", _BENCHMARK_DTYPES)
def test_fused_gated_norm_benchmark_shape(shape, dtype):
    """Every shape whose latency the benchmark reports must be correct."""
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    gate = torch.randn(shape, dtype=dtype, device=flag_train.device)
    weight = torch.randn((shape[-1],), dtype=dtype, device=flag_train.device)

    out = flag_train.megatron.fused_gated_norm(x, gate, weight, _EPS)
    ref_out = _oracle(
        utils.to_reference(x),
        utils.to_reference(gate),
        utils.to_reference(weight),
        _EPS,
    )
    _assert_close_to_oracle(out, ref_out, dtype)


# ---------------------------------------------------------------------------
# The upstream oracle is itself checked against the torch composition
#
# On NVIDIA the oracle is Megatron-LM's kernel, and the port is measured against
# it. This test holds the oracle to the same plain-torch contract, so a baseline
# regression cannot silently become a port regression. It skips wherever the
# upstream operator is unavailable.
# ---------------------------------------------------------------------------

needs_mcore_oracle = pytest.mark.skipif(
    _mcore_fused_gated_norm is None,
    reason="megatron-core's fused_gated_norm is not available (NVIDIA-only oracle)",
)


@pytest.mark.fused_gated_norm
@needs_mcore_oracle
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("zero_centered", [False, True], ids=["norm", "zero_centered"])
def test_mcore_fused_gated_norm_forward(shape, dtype, zero_centered):
    """The upstream oracle must agree with the torch composition."""
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    gate = torch.randn(shape, dtype=dtype, device=flag_train.device)
    weight = torch.randn((shape[-1],), dtype=dtype, device=flag_train.device)

    out = _mcore_fused_gated_norm(x, gate, weight, _EPS, zero_centered)
    ref_out = fused_gated_norm_ref(
        utils.to_reference(x),
        utils.to_reference(gate),
        utils.to_reference(weight),
        _EPS,
        zero_centered,
    )
    _assert_close_to_oracle(out, ref_out, dtype)


@pytest.mark.fused_gated_norm
@pytest.mark.parametrize(
    "dtype",
    [torch.float16] + ([torch.bfloat16] if utils.bf16_is_supported else []),
    ids=lambda dt: str(dt).replace("torch.", ""),
)
@pytest.mark.parametrize("shape", [_SHAPE, (1, 7, 3, 128)])
@pytest.mark.parametrize("zero_centered", [False, True], ids=["norm", "zero_centered"])
def test_fused_gated_norm_fp32_gate(shape, dtype, zero_centered):
    """The gate may be FP32 while the activation is a lower precision."""
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    gate = torch.randn(shape, dtype=torch.float32, device=flag_train.device)
    weight = torch.randn((shape[-1],), dtype=torch.float32, device=flag_train.device)

    out = flag_train.megatron.fused_gated_norm(x, gate, weight, _EPS, zero_centered)
    ref_out = _oracle(
        utils.to_reference(x),
        utils.to_reference(gate),
        utils.to_reference(weight),
        _EPS,
        zero_centered,
    )
    _assert_close_to_oracle(out, ref_out, dtype)


# ---------------------------------------------------------------------------
# Strided inputs
#
# The kernel reads logical ``[batch, sequence, head, dim]`` rows through their
# strides rather than copying views, so every way a caller can present that shape
# needs to stay correct: a transposed output, a view with a storage offset, and a
# non-unit last-dimension stride.
# ---------------------------------------------------------------------------

_LAYOUT_CONTIGUOUS = {
    "transposed-both": (False, False),
    "transposed-x": (False, True),
    "transposed-gate": (True, False),
    "strided-rows": (False, False),
    "strided-last": (False, False),
}


def _strided_operands(layout, shape, dtype):
    """Build ``(x, gate, weight)`` with the requested layout."""
    batch, sequence, heads, dim = shape
    device = flag_train.device
    weight = torch.randn((dim,), dtype=dtype, device=device)

    if layout == "strided-last":
        # ``dim`` is the inner axis of the base allocation, so the last logical
        # dimension ends up with stride ``heads`` instead of 1.
        base_shape = (batch, sequence, dim, heads)
        x = torch.randn(base_shape, dtype=dtype, device=device)
        gate = torch.randn(base_shape, dtype=dtype, device=device)
        return x.permute(0, 1, 3, 2), gate.permute(0, 1, 3, 2), weight

    if layout == "strided-rows":
        # A doubly-strided, offset slice: neither the row start nor the row-to-row
        # distance is the contiguous one.
        base_shape = (2 * batch, 2 * sequence, heads, dim)
        x = torch.randn(base_shape, dtype=dtype, device=device)
        gate = torch.randn(base_shape, dtype=dtype, device=device)
        return x[1::2, 1::2], gate[1::2, 1::2], weight

    base_shape = (batch, heads, sequence, dim)
    x = torch.randn(base_shape, dtype=dtype, device=device).permute(0, 2, 1, 3)
    gate = torch.randn(base_shape, dtype=dtype, device=device).permute(0, 2, 1, 3)
    if layout == "transposed-x":
        gate = gate.contiguous()
    elif layout == "transposed-gate":
        x = x.contiguous()
    return x, gate, weight


@pytest.mark.fused_gated_norm
@pytest.mark.parametrize("layout", list(_LAYOUT_CONTIGUOUS))
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_gated_norm_strided(layout, dtype):
    """Strided views must be read in place and produce the contiguous result."""
    x, gate, weight = _strided_operands(layout, _SHAPE, dtype)
    x_contiguous, gate_contiguous = _LAYOUT_CONTIGUOUS[layout]
    assert x.is_contiguous() is x_contiguous
    assert gate.is_contiguous() is gate_contiguous

    out = flag_train.megatron.fused_gated_norm(x, gate, weight, _EPS)
    ref_out = _oracle(
        utils.to_reference(x),
        utils.to_reference(gate),
        utils.to_reference(weight),
        _EPS,
    )
    assert out.is_contiguous()
    _assert_close_to_oracle(out, ref_out, dtype)


@pytest.mark.fused_gated_norm
@needs_mcore_oracle
@pytest.mark.parametrize("layout", list(_LAYOUT_CONTIGUOUS))
def test_mcore_fused_gated_norm_strided(layout):
    """The upstream oracle must also agree with the composition on strided views."""
    x, gate, weight = _strided_operands(layout, _SHAPE, torch.float32)
    out = _mcore_fused_gated_norm(x, gate, weight, _EPS)
    ref_out = fused_gated_norm_ref(
        utils.to_reference(x),
        utils.to_reference(gate),
        utils.to_reference(weight),
        _EPS,
    )
    _assert_close_to_oracle(out, ref_out, torch.float32)


# ---------------------------------------------------------------------------
# Edge values and API contract
# ---------------------------------------------------------------------------


@pytest.mark.fused_gated_norm
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_gated_norm_zero_activation(dtype):
    """An all-zero activation must produce zeros, not ``0 * rsqrt(0)`` NaNs."""
    x = torch.zeros(_SHAPE, dtype=dtype, device=flag_train.device)
    gate = torch.randn(_SHAPE, dtype=dtype, device=flag_train.device)
    weight = torch.randn((_SHAPE[-1],), dtype=dtype, device=flag_train.device)

    out = flag_train.megatron.fused_gated_norm(x, gate, weight, _EPS)
    ref_out = _oracle(
        utils.to_reference(x),
        utils.to_reference(gate),
        utils.to_reference(weight),
        _EPS,
    )
    _assert_close_to_oracle(out, ref_out, dtype)
    assert out.count_nonzero().item() == 0


@pytest.mark.fused_gated_norm
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_gated_norm_saturated_gate(dtype):
    """A gate far from zero drives SiLU into saturation; it must stay finite."""
    x = torch.randn(_SHAPE, dtype=dtype, device=flag_train.device) * 10
    gate = torch.randn(_SHAPE, dtype=dtype, device=flag_train.device) * 50
    weight = torch.randn((_SHAPE[-1],), dtype=dtype, device=flag_train.device)

    out = flag_train.megatron.fused_gated_norm(x, gate, weight, _EPS)
    ref_out = _oracle(
        utils.to_reference(x),
        utils.to_reference(gate),
        utils.to_reference(weight),
        _EPS,
    )
    assert torch.isfinite(out).all().item()
    _assert_close_to_oracle(out, ref_out, dtype)


@pytest.mark.fused_gated_norm
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_gated_norm_output_contract(dtype):
    """The result keeps ``x``'s shape/dtype, is contiguous, and inputs are intact."""
    x = torch.randn(_SHAPE, dtype=dtype, device=flag_train.device)
    gate = torch.randn(_SHAPE, dtype=dtype, device=flag_train.device)
    weight = torch.randn((_SHAPE[-1],), dtype=dtype, device=flag_train.device)
    x_before, gate_before, weight_before = x.clone(), gate.clone(), weight.clone()

    out = flag_train.megatron.fused_gated_norm(x, gate, weight, _EPS)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.is_contiguous()
    assert torch.equal(x, x_before)
    assert torch.equal(gate, gate_before)
    assert torch.equal(weight, weight_before)


@pytest.mark.fused_gated_norm
@pytest.mark.parametrize(
    "bad_input",
    ["x-ndim", "gate-shape", "head-dim", "weight-shape", "weight-stride"],
)
def test_fused_gated_norm_rejects_invalid_input(bad_input):
    """Unsupported layouts fail with a diagnostic instead of reading out of bounds."""
    x = torch.randn(_SHAPE, dtype=torch.float32, device=flag_train.device)
    gate = torch.randn(_SHAPE, dtype=torch.float32, device=flag_train.device)
    weight = torch.randn((_SHAPE[-1],), dtype=torch.float32, device=flag_train.device)

    if bad_input == "x-ndim":
        x = x.reshape(16, 4, 64)
        gate = gate.reshape(16, 4, 64)
        match = "4-D shape"
    elif bad_input == "gate-shape":
        gate = gate[..., :32]
        match = "4-D shape"
    elif bad_input == "head-dim":
        x = torch.randn((2, 8, 4, 96), dtype=torch.float32, device=flag_train.device)
        gate = torch.randn((2, 8, 4, 96), dtype=torch.float32, device=flag_train.device)
        weight = torch.randn((96,), dtype=torch.float32, device=flag_train.device)
        match = "power-of-two"
    elif bad_input == "weight-shape":
        weight = torch.randn((32,), dtype=torch.float32, device=flag_train.device)
        match = r"contiguous \(dim,\)"
    else:
        weight = torch.randn(
            (2 * _SHAPE[-1],), dtype=torch.float32, device=flag_train.device
        )[::2]
        match = r"contiguous \(dim,\)"

    with pytest.raises(ValueError, match=match):
        flag_train.megatron.fused_gated_norm(x, gate, weight, _EPS)
