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
"""Performance benchmark for the mixed_moe_gemm operator.

There is no vendor-native baseline: DeepSpeed has no ``mixed_moe_gemm``, so the
only local implementation of the same contract is a dequantization plus
per-expert PyTorch GEMM loop. A torch composition is not a competitor and is
not timed: ``torch_op`` is ``None``, the benchmark reports Train latency only,
and ``--base-data`` is the only comparison. The same composition is still used
as an untimed correctness check in the input builder.
"""

import math

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


class MixedMoEGEMMBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _SHAPES
    DEFAULT_SHAPE_DESC = "experts, tokens_per_expert, K, N"

    def set_shapes(self, shape_file=None):
        self.shapes = list(_SHAPES)

    def set_more_shapes(self):
        return []


_BASELINE = "none (no on-machine baseline; --base-data is the only comparison)"


def _reference(output, x, weights, scales, biases, rows, num_bits, activation):
    """Untimed oracle: unpacking, dequantization, bias, and activation.

    Benchmark workloads allocate equally many tokens to every expert, so row
    boundaries are known from shapes; no device-to-host copies are timed.
    """
    experts, n = scales.shape
    if num_bits == 4:
        unsigned = weights.to(torch.int16)
        low = unsigned & 15
        high = (unsigned >> 4) & 15
        values = torch.stack((low, high), dim=-1).flatten(-2)[..., :n]
        signed = torch.where(values >= 8, values - 16, values)
    else:
        signed = weights
    dequantized = (signed.float() * scales.float()[:, None, :]).to(x.dtype)
    tokens_per_expert = x.shape[0] // experts
    for expert in range(experts):
        start = expert * tokens_per_expert
        end = start + tokens_per_expert
        result = torch.matmul(x[start:end], dequantized[expert]).float()
        result = result + biases[expert].float()
        if activation == 0:
            result = F.gelu(result)
        elif activation == 1:
            result = F.relu(result)
        elif activation == 2:
            result = F.silu(result)
        output[start:end].copy_(result)
    return output


def input_fn(shape, dtype, device, num_bits, activation):
    torch.manual_seed(0)
    experts, tokens, k, n = shape
    limit = 2 ** (num_bits - 1)
    signed = torch.randint(
        -limit, limit, (experts, k, n), dtype=torch.int8, device=device
    )
    if num_bits == 4:
        values = signed.to(torch.int16)
        weights = ((values[..., 0::2] & 15) | ((values[..., 1::2] & 15) << 4)).to(
            torch.uint8
        )
    else:
        weights = signed
    scales = (torch.rand((experts, n), device=device) / (limit * math.sqrt(k))).to(
        dtype
    )
    biases = torch.randn((experts, n), dtype=dtype, device=device)
    rows = torch.arange(1, experts + 1, dtype=torch.int64, device=device) * tokens
    x = torch.randn((experts * tokens, k), dtype=dtype, device=device)
    output = torch.empty((experts * tokens, n), dtype=dtype, device=device)
    args = (output, x, weights, scales, biases, rows, num_bits, activation)
    expected = _reference(*args).clone()
    flag_train.deepspeed.mixed_moe_gemm(*args)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(output, expected, rtol=tolerance, atol=tolerance)
    yield args


@pytest.mark.mixed_moe_gemm
@pytest.mark.skipif(
    flag_train.vendor_name
    not in ("nvidia", "hygon", "metax", "thead", "iluvatar", "ascend", "mthreads"),
    reason="Unsupported accelerator",
)
@pytest.mark.parametrize("num_bits", [4, 8])
@pytest.mark.parametrize("activation", [6, 2], ids=["identity", "silu"])
def test_mixed_moe_gemm_perf(num_bits, activation):
    def make_inputs(shape, dtype, device):
        yield from input_fn(shape, dtype, device, num_bits, activation)

    print(f"\nBaseline: {_BASELINE}")
    bench = MixedMoEGEMMBenchmark(
        input_fn=make_inputs,
        op_name="mixed_moe_gemm",
        # None: no vendor-native implementation exists, and a torch composition
        # is not timed as a baseline, so --base-data supplies the comparison.
        torch_op=None,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_train(flag_train.deepspeed.mixed_moe_gemm)
    bench.run()
