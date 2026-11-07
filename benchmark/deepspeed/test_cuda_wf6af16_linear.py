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
"""Performance benchmark for the WF6-AF16 linear operator.

The baseline follows ``_DEEPSPEED_BASELINE_VENDORS`` below:

* on a listed backend with an Ampere GPU, DeepSpeed's ``cuda_wf6af16_linear``
  **is** the baseline. The CUDA kernel asserts ``__CUDA_ARCH__ == 800``, so on
  any other architecture it cannot run;
* everywhere else there is no on-machine baseline. An unfused torch composition
  (unpack, dequantize to an fp16 weight matrix, cuBLAS GEMM) is not a
  competitor and is not timed: a ratio against it says how far the kernel is
  from *a* correct implementation rather than from the best one. ``torch_op`` is
  ``None``, so the framework reports Train latency only and ``--base-data`` is
  the only comparison.

The weight packings differ between the two implementations (DeepSpeed
interleaves bits across threads for its MMA pipeline, FlagTrain packs densely
row-major), so the inputs carry both packings; packing itself is setup, not
part of the timed region.
"""

import pytest
import torch

import flag_train

from .. import base

# Inference-style shapes: (tokens, in_channels, out_channels). The decode
# ladder (1..64) is dense because the split-K heuristic switches there; the
# prefill shapes reach past the map's 768-token ceiling. The channel pairs
# mirror the sizes DeepSpeed's split-K map is profiled for.
_CUDA_WF6AF16_SHAPES = [
    (1, 4096, 4096),
    (4, 4096, 4096),
    (8, 4096, 4096),
    (16, 4096, 4096),
    (16, 14336, 4096),
    (32, 4096, 4096),
    (64, 4096, 4096),
    (128, 4096, 14336),
    (256, 8192, 8192),
    (512, 4096, 4096),
    (1024, 4096, 4096),
]

# ---------------------------------------------------------------------------
# Input preparation helpers (torch is fine here: this is benchmark setup code,
# not the library's host path, and packing happens outside the timed region).
# ---------------------------------------------------------------------------


def _fp6_e3m2_table(device=None):
    """All 64 values representable in FP6-E3M2, indexed by the 6-bit code.

    Built on the CPU: it is only 64 exact values, and some device stacks
    (Iluvatar's ixrtc) fail to JIT-compile torch's fp32 ``exp2`` kernel.
    """
    codes = torch.arange(64, dtype=torch.int32)
    sign = torch.where((codes >> 5) & 1 == 1, -1.0, 1.0)
    e = ((codes >> 2) & 7).to(torch.float32)
    m = (codes & 3).to(torch.float32)
    mag = torch.where(
        (codes >> 2) & 7 == 0,
        m * 0.0625,
        torch.exp2(e - 3.0) * (1.0 + m * 0.25),
    )
    return (sign * mag).to(device)


def _pack_weights_fp6(codes):
    """Pack FP6 codes into the 2-bit and 4-bit slices the kernel consumes."""
    out_channels, in_channels = codes.shape

    top2 = codes >> 4
    low4 = codes & 0xF

    t = top2.reshape(out_channels, in_channels // 4, 4).to(torch.uint8)
    weights_2bit = (t[..., 0] << 6) | (t[..., 1] << 4) | (t[..., 2] << 2) | t[..., 3]

    low = low4.reshape(out_channels, in_channels // 2, 2)
    weights_4bit = (low[..., 0] << 4) | low[..., 1]

    return weights_2bit.contiguous(), weights_4bit.contiguous()


# Backends whose baseline is DeepSpeed when the extension and GPU allow it.
# cuda_wf6af16_linear ships in the InferenceCore CUDA extension, so only a
# backend that can compile and execute one can host it. Hygon is deliberately
# absent: the extension builds through HIP there, but the kernel body is
# guarded by __CUDA_ARCH__ in [800, 900) (kernel_matmul.cuh), so the DCU build
# compiles to a stub that returns garbage or fails the launch outright; a DCU
# therefore has no on-machine baseline.
_DEEPSPEED_BASELINE_VENDORS = {"nvidia"}


def _load_deepspeed_cuda_wf6af16():
    """Harness-ready cuda_wf6af16_linear, or ``None`` when it cannot run here.

    Besides the extension building at all, the kernel hard-codes
    ``__CUDA_ARCH__ == 800``, so on non-Ampere NVIDIA GPUs it is treated as
    unavailable and no on-machine baseline is timed.
    """
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None
    if flag_train.vendor_name == "nvidia" and torch.cuda.get_device_capability() != (
        8,
        0,
    ):
        return None

    try:
        from deepspeed.ops.op_builder import InferenceCoreBuilder

        inf_module = InferenceCoreBuilder().load()
        inf_module.create_handle()
    except Exception:
        # The baseline is advisory, not contractual: a missing extension just
        # means no on-machine baseline is timed and --base-data is the only
        # comparison.
        return None

    def deepspeed_cuda_wf6af16(hidden_states, weights_2bit, weights_4bit, scales):
        out_channels = scales.shape[0]
        tokens, in_channels = hidden_states.shape
        # weights_2bit/weights_4bit here are DeepSpeed's own packing, produced
        # in the input setup; only the kernel call is timed.
        output = torch.empty(
            (tokens, out_channels), dtype=torch.float16, device=hidden_states.device
        )
        workspace = torch.empty(
            (1, out_channels, tokens), dtype=torch.float32, device=hidden_states.device
        )
        inf_module.cuda_wf6af16_linear(
            output,
            hidden_states,
            weights_2bit,
            weights_4bit,
            scales,
            workspace,
            out_channels,
            tokens,
            in_channels,
            1,
        )
        return output

    def torch_op(hidden, weights_2bit, weights_4bit, scales, ds_w2, ds_w4):
        return deepspeed_cuda_wf6af16(hidden, ds_w2, ds_w4, scales)

    # The DeepSpeed packing needs the host-side preprocess_weight; expose it on
    # the harness-ready callable so the input setup can pre-pack.
    torch_op.preprocess_weight = inf_module.preprocess_weight
    return torch_op


# Resolved once, so the first-use JIT compile is not counted in the measurement;
# ``None`` is the benchmark harness's "no on-machine baseline" sentinel.
torch_op = _load_deepspeed_cuda_wf6af16()

_BASELINE = (
    "deepspeed cuda_wf6af16_linear"
    if torch_op is not None
    else "none (no on-machine baseline; --base-data is the only comparison)"
)


class CudaWf6Af16LinearBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _CUDA_WF6AF16_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_CUDA_WF6AF16_SHAPES)

    def set_more_shapes(self):
        return []


def cuda_wf6af16_input_fn(shape, dtype, device):
    tokens, in_channels, out_channels = shape
    gen = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 64, (out_channels, in_channels), generator=gen).to(
        torch.uint8
    )
    scales = (torch.rand(out_channels, generator=gen) + 0.5).to(torch.float16)
    hidden = torch.randn(tokens, in_channels, generator=gen).to(dtype)

    weights_2bit, weights_4bit = _pack_weights_fp6(codes.to(device))

    ds_w2 = ds_w4 = None
    if torch_op is not None:
        # DeepSpeed's packer wants the fake-quantized fp16 weights on CPU; the
        # table values are exact in fp16, so the repack is lossless.
        table = _fp6_e3m2_table()
        fake_fp16 = table[codes.long()].to(torch.float16)
        ds_w2, ds_w4 = torch_op.preprocess_weight(fake_fp16)
        ds_w2 = ds_w2.to(device)
        ds_w4 = ds_w4.to(device)

    yield (
        hidden.to(device),
        weights_2bit,
        weights_4bit,
        scales.to(device),
        ds_w2,
        ds_w4,
    )


def train_op(hidden, weights_2bit, weights_4bit, scales, ds_w2, ds_w4):
    """The operator under test.

    ``ds_w2``/``ds_w4`` are the baseline's packing and are accepted only so both
    ops are handed the same tuple.
    """
    out_channels = scales.shape[0]
    tokens, in_channels = hidden.shape
    output = torch.empty(
        (tokens, out_channels), dtype=torch.float16, device=hidden.device
    )
    return flag_train.deepspeed.cuda_wf6af16_linear(
        output,
        hidden,
        weights_2bit,
        weights_4bit,
        scales,
        out_channels,
        tokens,
        in_channels,
    )


@pytest.mark.cuda_wf6af16_linear
def test_cuda_wf6af16_linear_perf():
    print(f"\nBaseline: {_BASELINE}")

    bench = CudaWf6Af16LinearBenchmark(
        input_fn=cuda_wf6af16_input_fn,
        op_name="cuda_wf6af16_linear",
        # None when DeepSpeed's kernel cannot run here: a torch composition is
        # not timed as a baseline, so --base-data supplies the comparison.
        torch_op=torch_op,
        dtypes=[torch.float16],
    )
    bench.set_train(train_op)
    bench.run()
