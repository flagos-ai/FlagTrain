import math

import pytest
import torch
import torch.nn.functional as F

import flag_train
from flag_train.runtime import torch_device_fn

from .. import accuracy_utils as utils

pytestmark = [
    pytest.mark.moe_gemm,
    pytest.mark.skipif(
        flag_train.vendor_name
        not in ("nvidia", "hygon", "metax", "thead", "iluvatar", "ascend"),
        reason="Unsupported accelerator",
    ),
]

_ACTIVATIONS = {0: F.gelu, 1: F.relu, 2: F.silu, 6: lambda x: x}
_CASES = [
    ([37], 128, 64),
    ([0, 3, 0, 19, 1, 0, 12, 0], 129, 65),
    ([i % 5 for i in range(64)], 33, 31),
    ([7, 35, 129], 127, 129),
]

def _inputs(counts, k, n, dtype, strided=False):
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
    # Tiled FP32 reduction and activation approximations need not round exactly
    # like the CPU FP64 oracle. Keep the shared floating-point defaults.
    utils.train_assert_close(output, expected, output.dtype)


@pytest.mark.parametrize("counts,k,n", _CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation", [0, 1, 2, 6])
@pytest.mark.parametrize("has_bias", [False, True])
def test_moe_gemm(counts, k, n, dtype, activation, has_bias):
    """Catch wrong expert boundaries, skipped tails, bias, or activation."""
    output, x, weights, biases, rows = _inputs(
        counts, k, n, dtype
    )
    if not has_bias:
        biases = None
    expected = _reference(x, weights, biases, counts, activation)
    result = flag_train.deepspeed.moe_gemm(
        output, x, weights, biases, rows, activation
    )
    assert result is output
    _assert_close(output, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moe_gemm_strides(dtype):
    """Catch assuming contiguous input/output or overwriting view padding."""
    counts = [0, 1, 17, 0, 5]
    output, x, weights, biases, rows = _inputs(
        counts, 71, 35, dtype, strided=True
    )
    expected = _reference(x, weights, biases, counts, 6)
    storage = torch.full(
        (2 * sum(counts), 70), float("nan"), dtype=dtype, device=x.device
    )
    output = storage[::2, ::2]
    flag_train.deepspeed.moe_gemm(
        output, x, weights, biases, rows
    )
    _assert_close(output, expected)
    assert torch.isnan(storage[1::2]).all()
    assert torch.isnan(storage[::2, 1::2]).all()


@pytest.mark.parametrize(
    "counts,k,n", [([0, 0], 17, 7), ([3, 0, 2], 0, 7), ([3, 0, 2], 17, 0)]
)
def test_moe_gemm_empty_dimensions(counts, k, n):
    """Catch invalid zero-size launches or losing the bias when K is zero."""
    output, x, weights, biases, rows = _inputs(
        counts, k, n, torch.float16
    )
    expected = _reference(x, weights, biases, counts, 2)
    result = flag_train.deepspeed.moe_gemm(
        output, x, weights, biases, rows, 2
    )
    assert result is output
    _assert_close(output, expected)


def test_moe_gemm_graph_capture():
    """Catch hidden host synchronization; changed routing must work on replay."""
    counts = [1, 3, 0, 2]
    output, x, weights, biases, rows = _inputs(
        counts, 33, 17, torch.float16
    )
    graph_type = getattr(torch_device_fn, "CUDAGraph", None)
    if graph_type is None:
        graph_type = getattr(torch_device_fn, "NPUGraph", None)
    if graph_type is None:
        pytest.skip("The backend has no graph capture API")
    stream = torch_device_fn.Stream()
    stream.wait_stream(torch_device_fn.current_stream())
    with torch_device_fn.stream(stream):
        for _ in range(3):
            flag_train.deepspeed.moe_gemm(
                output, x, weights, biases, rows
            )
    torch_device_fn.current_stream().wait_stream(stream)
    graph = graph_type()
    with torch_device_fn.graph(graph):
        flag_train.deepspeed.moe_gemm(
            output, x, weights, biases, rows
        )
    new_counts = [0, 0, 5, 1]
    rows.copy_(torch.tensor(new_counts, dtype=torch.int64, device=x.device).cumsum(0))
    graph.replay()
    expected = _reference(x, weights, biases, new_counts, 6)
    _assert_close(output, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mixed_moe_gemm_large_k(dtype):
    """Catch a kernel that compiles for small K but hangs or miscomputes at K=2048."""
    counts = [128]
    output, x, weights, biases, rows = _inputs(
        counts, 2048, 2048, dtype
    )
    expected = _reference(x, weights, biases, counts, 2)
    flag_train.deepspeed.moe_gemm(
        output, x, weights, biases, rows, 2
    )
    _assert_close(output, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moe_gemm_two_layers(dtype):
    counts = [0, 2, 17, 1]
    hidden, x, weights, biases, rows = _inputs(
        counts, 33, 47, dtype
    )
    output, _, weights2, biases2, _ = _inputs(
        counts, 47, 19, dtype
    )
    flag_train.deepspeed.moe_gemm(
        hidden, x, weights, biases, rows, 2
    )
    flag_train.deepspeed.moe_gemm(
        output, hidden, weights2, biases2, rows
    )
    hidden_ref = _reference(x, weights, biases, counts, 2)
    expected = _reference(
        hidden_ref, weights2, biases2, counts, 6
    )
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
    output, x, weights, biases, rows = _inputs(
        [3, 1], 17, 9, torch.float16
    )
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
        flag_train.deepspeed.moe_gemm(
            output, x, weights, biases, rows, activation
        )
