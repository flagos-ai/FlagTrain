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
"""Correctness tests for the MoE GEMM operator.

One oracle: ``_reference`` -- a plain-torch, CPU FP64 composition of the same
contract. It runs on the CPU in double precision, so the tiled FP32 reduction
and the activation approximations of the kernel cannot agree with it by sharing
the same device arithmetic; the comparison helpers move the operator's own
output across afterwards.

The oracle is deliberately not a second kernel reading: expert boundaries, the
tail of a non-contiguous output, bias handling and the activation are all
re-derived here from ``counts`` alone, so a shared misreading of the contract
would have to be made twice, independently, to survive.
"""

import pytest
import torch
import torch.nn.functional as F

import flag_train
from flag_train.runtime import torch_device_fn

from .. import accuracy_utils as utils

# The module-level marks apply to every test below: the operator is only
# meaningful on backends that ship the fused MoE GEMM, and each test is tagged
# so it can be selected with ``-m moe_gemm``.
pytestmark = [
    pytest.mark.moe_gemm,
    pytest.mark.skipif(
        flag_train.vendor_name
        not in ("nvidia", "hygon", "metax", "thead", "iluvatar", "ascend"),
        reason="Unsupported accelerator",
    ),
]

# Activation codes as the operator's ABI numbers them, mirroring the dispatch in
# the kernel. Code 6 is the identity, used where a test wants the matmul alone.
_ACTIVATIONS = {0: F.gelu, 1: F.relu, 2: F.silu, 6: lambda x: x}

# Shapes chosen to exercise different boundary conditions rather than to be
# representative: a single expert, leading/trailing empty experts, a ragged
# 64-way split, and K/N that are not multiples of the usual tile sizes.
_CASES = [
    ([37], 128, 64),
    ([0, 3, 0, 19, 1, 0, 12, 0], 129, 65),
    ([i % 5 for i in range(64)], 33, 31),
    ([7, 35, 129], 127, 129),
]

# ---------------------------------------------------------------------------
# Reference implementation and shared fixtures
#
# Exists so the operator can be checked against a plain-torch, CPU FP64
# composition of the same contract on any supported backend. The reference runs
# in double precision on the CPU, so it is an independent oracle rather than a
# second reading of the same device arithmetic.
# ---------------------------------------------------------------------------


def _inputs(counts, k, n, dtype, strided=False):
    """Build one operator invocation's operands.

    ``counts[e]`` is the number of rows routed to expert ``e``; ``rows`` is their
    cumulative sum, i.e. the expert boundaries the kernel reads. Operands are
    created on ``flag_train.device`` so the operator runs on the backend under
    test.

    With ``strided=True`` the input and output are non-contiguous views into
    larger storages: a stride test must observe the kernel's handling of the
    gaps, so the storage is filled with NaN and the untouched padding is checked
    afterwards by the caller.
    """
    torch.manual_seed(0)
    m, experts = sum(counts), len(counts)
    weights = torch.randn(
        (experts, k, n),
        dtype=dtype,
        device=flag_train.device,
    )
    rows = torch.tensor(counts, dtype=torch.int64, device=flag_train.device).cumsum(0)
    biases = torch.randn((experts, n), dtype=dtype, device=flag_train.device)
    if strided:
        x = torch.randn((2 * m, 2 * k), dtype=dtype, device=flag_train.device)[::2, ::2]
        output = torch.full(
            (2 * m, 2 * n), float("nan"), dtype=dtype, device=flag_train.device
        )[::2, ::2]
    else:
        x = torch.randn((m, k), dtype=dtype, device=flag_train.device)
        output = torch.full((m, n), float("nan"), dtype=dtype, device=flag_train.device)
    return output, x, weights, biases, rows


def _reference(x, weights, biases, counts, activation):
    """Reference for the MoE GEMM operator, composed from plain torch ops.

    Slices ``x`` by the ``counts``-derived expert boundaries and applies each
    expert's weight, optional bias and activation, all in CPU FP64. The double
    precision is what makes the oracle independent of the kernel's FP32 tiled
    reduction: rounding differences must stay within the shared tolerance rather
    than cancel by construction.

    The result is moved back to ``x``'s dtype and device before
    ``utils.to_reference``, so the comparison happens on whichever device the
    test framework asked the reference to run on.
    """
    x_cpu = x.cpu().double()
    weights_cpu = weights.cpu().double()
    result = torch.empty((x.shape[0], weights.shape[2]), dtype=torch.float64)
    start = 0
    for expert, count in enumerate(counts):
        end = start + count
        values = torch.matmul(x_cpu[start:end], weights_cpu[expert])
        if biases is not None:
            values = values + biases[expert].cpu().double()
        result[start:end] = _ACTIVATIONS[activation](values)
        start = end
    return utils.to_reference(result.to(x.dtype).to(x.device))


def _assert_close(output, expected):
    """Compare the operator's output to the CPU FP64 oracle.

    The CPU FP64 oracle and the tiled FP32 kernel need not round exactly alike,
    so the shared floating-point defaults are kept rather than a tightened
    tolerance; the tolerances themselves are dtype-derived from ``output``.
    """
    # Tiled FP32 reduction and activation approximations need not round exactly
    # like the CPU FP64 oracle. Keep the shared floating-point defaults.
    utils.train_assert_close(output, expected, output.dtype)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("counts,k,n", _CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation", [0, 1, 2, 6])
@pytest.mark.parametrize("has_bias", [False, True])
def test_moe_gemm(counts, k, n, dtype, activation, has_bias):
    """Catch wrong expert boundaries, skipped tails, bias, or activation.

    The full cross product of shapes, dtypes, activations and bias presence is
    what separates the failure modes: an off-by-one boundary is invisible on a
    single expert, and a dropped bias looks like a tolerance wobble on one
    shape only. The operator returns its output tensor, which is asserted by
    identity so a copy cannot hide the contract.
    """
    output, x, weights, biases, rows = _inputs(counts, k, n, dtype)
    if not has_bias:
        biases = None
    expected = _reference(x, weights, biases, counts, activation)
    result = flag_train.deepspeed.moe_gemm(output, x, weights, biases, rows, activation)
    assert result is output
    _assert_close(output, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moe_gemm_strides(dtype):
    """Catch assuming contiguous input/output or overwriting view padding.

    Both operands are non-contiguous views: the kernel must index by strides
    rather than assume a packed layout, and must not write into the padding
    between the viewed elements, which is checked as NaN afterwards.
    """
    counts = [0, 1, 17, 0, 5]
    output, x, weights, biases, rows = _inputs(counts, 71, 35, dtype, strided=True)
    expected = _reference(x, weights, biases, counts, 6)
    storage = torch.full(
        (2 * sum(counts), 70), float("nan"), dtype=dtype, device=x.device
    )
    output = storage[::2, ::2]
    flag_train.deepspeed.moe_gemm(output, x, weights, biases, rows)
    _assert_close(output, expected)
    # The padding must be untouched: a kernel that writes a full contiguous tile
    # would fill these with values instead of leaving the NaN sentinel.
    assert torch.isnan(storage[1::2]).all()
    assert torch.isnan(storage[::2, 1::2]).all()


@pytest.mark.parametrize(
    "counts,k,n", [([0, 0], 17, 7), ([3, 0, 2], 0, 7), ([3, 0, 2], 17, 0)]
)
def test_moe_gemm_empty_dimensions(counts, k, n):
    """Catch invalid zero-size launches or losing the bias when K is zero.

    Empty routing, zero K and zero N reach different degenerate paths in the
    kernel; a launch with a zero grid dimension can fail outright, and with
    K == 0 the matmul contributes nothing, so the bias must still be produced.
    """
    output, x, weights, biases, rows = _inputs(counts, k, n, torch.float16)
    expected = _reference(x, weights, biases, counts, 2)
    result = flag_train.deepspeed.moe_gemm(output, x, weights, biases, rows, 2)
    assert result is output
    _assert_close(output, expected)


def test_moe_gemm_graph_capture():
    """Catch hidden host synchronization; changed routing must work on replay.

    A kernel that synchronizes, allocates, or reads ``rows`` on the host cannot
    be captured; one that caches the routing at capture time would replay the
    old boundaries. ``rows`` is rewritten between capture and replay precisely
    so the second failure mode shows up as a mismatch rather than as silently
    correct output for the wrong routing.
    """
    counts = [1, 3, 0, 2]
    output, x, weights, biases, rows = _inputs(counts, 33, 17, torch.float16)
    # The capture API has different names per backend, so probe for it rather
    # than assuming one; a backend without either skips instead of failing.
    graph_type = getattr(torch_device_fn, "CUDAGraph", None)
    if graph_type is None:
        graph_type = getattr(torch_device_fn, "NPUGraph", None)
    if graph_type is None:
        pytest.skip("The backend has no graph capture API")
    # Warm the build up on a side stream first -- a JIT-compiling or autotuning
    # op cannot be captured on its first invocation.
    stream = torch_device_fn.Stream()
    stream.wait_stream(torch_device_fn.current_stream())
    with torch_device_fn.stream(stream):
        for _ in range(3):
            flag_train.deepspeed.moe_gemm(output, x, weights, biases, rows)
    torch_device_fn.current_stream().wait_stream(stream)
    graph = graph_type()
    with torch_device_fn.graph(graph):
        flag_train.deepspeed.moe_gemm(output, x, weights, biases, rows)
    # New routing, same replay: the result must follow ``rows`` as it is at
    # replay time, not as it was at capture time.
    new_counts = [0, 0, 5, 1]
    rows.copy_(torch.tensor(new_counts, dtype=torch.int64, device=x.device).cumsum(0))
    graph.replay()
    expected = _reference(x, weights, biases, new_counts, 6)
    _assert_close(output, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mixed_moe_gemm_large_k(dtype):
    """Catch a kernel that compiles for small K but hangs or miscomputes at K=2048.

    K=2048 forces the reduction loop past the point where the small-shape tests
    stop; a mis-tiled or unguarded loop can still pass those and fail here.
    """
    counts = [128]
    output, x, weights, biases, rows = _inputs(counts, 2048, 2048, dtype)
    expected = _reference(x, weights, biases, counts, 2)
    flag_train.deepspeed.moe_gemm(output, x, weights, biases, rows, 2)
    _assert_close(output, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moe_gemm_two_layers(dtype):
    """Catch per-layer state (workspace, routing) leaking between invocations.

    The second invocation's input is the first's output, so a kernel that
    reuses an internal buffer incorrectly corrupts the chain rather than a
    single step.
    """
    counts = [0, 2, 17, 1]
    hidden, x, weights, biases, rows = _inputs(counts, 33, 47, dtype)
    output, _, weights2, biases2, _ = _inputs(counts, 47, 19, dtype)
    flag_train.deepspeed.moe_gemm(hidden, x, weights, biases, rows, 2)
    flag_train.deepspeed.moe_gemm(output, hidden, weights2, biases2, rows)
    hidden_ref = _reference(x, weights, biases, counts, 2)
    expected = _reference(hidden_ref, weights2, biases2, counts, 6)
    _assert_close(output, expected)


@pytest.mark.parametrize(
    "invalid",
    [
        "activation",
        "weight_dtype",
        "rows",
        "bias",
        "weight_shape",
        "device",
    ],
)
def test_moe_gemm_rejects_invalid_inputs(invalid):
    """Catch a missing precondition check on each individually invalid operand.

    One operand is corrupted at a time so the failure names the offending
    argument; the contract is that a bad operand raises ``ValueError`` or
    ``TypeError`` rather than producing a plausible-looking result.
    """
    output, x, weights, biases, rows = _inputs([3, 1], 17, 9, torch.float16)
    activation = 6
    if invalid == "activation":
        activation = 3
    elif invalid == "weight_dtype":
        weights = weights.float()
    elif invalid == "rows":
        rows = rows.int()
    elif invalid == "bias":
        biases = biases.float()
    elif invalid == "weight_shape":
        weights = weights[:, :, :1]
    elif invalid == "device":
        weights = weights.cpu()
    with pytest.raises((ValueError, TypeError)):
        flag_train.deepspeed.moe_gemm(output, x, weights, biases, rows, activation)
