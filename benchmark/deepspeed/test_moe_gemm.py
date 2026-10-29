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
"""Performance benchmark for the MoE GEMM operator.

The baseline is ``torch_op``, a plain-torch loop over experts evaluated in FP32.
It is not a competitor and is not meant to be: the reported speedup says how far
the fused kernel is from *a* correct reference rather than from the best
implementation. The upcasting to FP32 in the baseline is deliberate -- it keeps
the precision of the reference independent of the half-precision operands, so
the timing measures routing/tiling overhead rather than the cost of a different
arithmetic path.

Every input is validated against the baseline before it is handed to the
benchmark; see ``input_fn``. A perf test that silently benchmarks a wrong kernel
is worse than one that fails.
"""

import pytest
import torch
import torch.nn.functional as F

import flag_train

from .. import base

# Shape is ``experts, tokens_per_expert, K, N``. The values are chosen to span the
# regimes the kernel dispatches on: one expert against many, tokens-per-expert
# small against large, and square against rectangular K/N.
_SHAPES = [
    (1, 128, 2048, 2048),
    (8, 16, 1024, 2048),
    (64, 1, 1024, 1024),
    (8, 128, 1024, 2048),
]

# ---------------------------------------------------------------------------
# Benchmark plumbing
#
# The benchmark class only carries the shape list; the operands are built in
# ``input_fn`` so the check can happen on the same values the benchmark times.
# ---------------------------------------------------------------------------


class MoEGEMMBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _SHAPES
    DEFAULT_SHAPE_DESC = "experts, tokens_per_expert, K, N"

    def set_shapes(self, shape_file=None):
        # The shapes are fixed rather than swept: each one corresponds to a
        # dispatch regime listed above, so a different set would time different
        # code paths rather than the same paths at another size.
        self.shapes = list(_SHAPES)

    def set_more_shapes(self):
        return []


def torch_op(output, x, weights, biases, rows, activation):
    """PyTorch baseline for standard FP16/BF16 MoE GEMM.

    Mirrors the operator's contract: equal-sized expert slices derived from
    ``rows``, optional bias, then the activation. ``output`` is written in place
    and returned, so the baseline can be called with the same argument tuple as
    the operator.
    """
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
    """Build one benchmark case and check it against the baseline first.

    The check runs here rather than in a separate correctness test so the timing
    measures values known to be correct: an operator that is faster because it
    computes the wrong thing would otherwise be reported as a speedup. The
    tolerances are set per dtype because BF16 loses more mantissa than FP16.

    ``expected`` is cloned before the operator writes ``output`` in place, so
    the two buffers do not alias. The same ``args`` tuple is yielded to the
    benchmark afterwards.
    """
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
    rows = torch.arange(1, experts + 1, dtype=torch.int64, device=device) * tokens
    output = torch.empty(
        (experts * tokens, n),
        dtype=dtype,
        device=device,
    )
    args = (output, x, weights, biases, rows, activation)
    # The baseline writes ``output`` in place; clone so the operator's call
    # below cannot overwrite the values being compared against.
    expected = torch_op(*args).clone()
    flag_train.deepspeed.moe_gemm(*args)
    # BF16 has fewer mantissa bits than FP16, so its tolerance is one order of
    # magnitude looser; both are chosen to admit the kernel's tiled FP32
    # reduction and fused activation without admitting a wrong result.
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(
        output,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )
    yield args


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.moe_gemm
@pytest.mark.skipif(
    flag_train.vendor_name
    not in ("nvidia", "hygon", "metax", "thead", "iluvatar", "ascend"),
    reason="Unsupported accelerator",
)
@pytest.mark.parametrize("activation", [6, 2], ids=["identity", "silu"])
def test_moe_gemm_perf(activation):
    """Catch a kernel that is fast on the identity path but not on SiLU.

    Identity (code 6) and SiLU (code 2) exercise different epilogues; timing
    only the identity would leave a slow fused activation unmeasured. Each case
    is validated by ``input_fn`` before it is timed.
    """

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
