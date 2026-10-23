import pytest
import torch
import torch.nn.functional as F

import flag_train

from .. import base


_SHAPES = [
    (1, 128, 2048, 2048),
    (8, 16, 1024, 2048),
    (64, 1, 1024, 1024),
    (8, 128, 1024, 2048),
]


class MoEGEMMBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _SHAPES
    DEFAULT_SHAPE_DESC = "experts, tokens_per_expert, K, N"

    def set_shapes(self, shape_file=None):
        self.shapes = list(_SHAPES)

    def set_more_shapes(self):
        return []


def torch_op(output, x, weights, biases, rows, activation):
    """PyTorch baseline for standard FP16/BF16 MoE GEMM."""
    experts = weights.shape[0]
    tokens_per_expert = x.shape[0] // experts
    for expert in range(experts):
        start = expert * tokens_per_expert
        end = start + tokens_per_expert
        result = torch.matmul(
            x[start:end].float(),
            weights[expert].float(),
        )
        if biases is not None:
            result = result + biases[expert].float()
        if activation == 0:
            result = F.gelu(result)
        elif activation == 1:
            result = F.relu(result)
        elif activation == 2:
            result = F.silu(result)
        output[start:end].copy_(result)
    return output


def input_fn(shape, dtype, device, activation):
    torch.manual_seed(0)
    experts, tokens, k, n = shape
    x = torch.randn(
        (experts * tokens, k),
        dtype=dtype,
        device=device,
    )
    weights = torch.randn(
        (experts, k, n),
        dtype=dtype,
        device=device,
    )
    biases = torch.randn(
        (experts, n),
        dtype=dtype,
        device=device,
    )
    rows = (
        torch.arange(1, experts + 1, dtype=torch.int64, device=device)
        * tokens
    )
    output = torch.empty(
        (experts * tokens, n),
        dtype=dtype,
        device=device,
    )
    args = (output, x, weights, biases, rows, activation)
    expected = torch_op(*args).clone()
    flag_train.deepspeed.moe_gemm(*args)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(
        output,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )
    yield args


@pytest.mark.moe_gemm
@pytest.mark.skipif(
    flag_train.vendor_name
    not in ("nvidia", "hygon", "metax", "thead", "iluvatar", "ascend"),
    reason="Unsupported accelerator",
)
@pytest.mark.parametrize("activation", [6, 2], ids=["identity", "silu"])
def test_moe_gemm_perf(activation):
    def make_inputs(shape, dtype, device):
        yield from input_fn(shape, dtype, device, activation)
    bench = MoEGEMMBenchmark(
        input_fn=make_inputs,
        op_name="moe_gemm",
        torch_op=torch_op,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_train(flag_train.deepspeed.moe_gemm)
    bench.run()
