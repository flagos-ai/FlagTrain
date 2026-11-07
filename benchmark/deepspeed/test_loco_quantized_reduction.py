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

import warnings
from enum import IntEnum

import pytest
import torch

import flag_train

from .. import base, consts


class QuantizationType(IntEnum):
    Symmetric = 0
    Asymmetric = 1


SINGLE_PASS_SHAPES = [
    (1024, 4096, 8),
    (2048, 2048, 8),
    (512, 2048, 8),
    (512, 4096, 8),
    (512, 8192, 8),
    (4096, 4096, 8),
    (1024, 8192, 8),
]
MULTIPASS_SHAPES = [
    (512, 16384, 8),
    (256, 49152, 8),
]
NT_SCALING_SHAPES = [
    (1024, 4096, 4),
    (512, 16384, 4),
    (512, 2048, 4),
    (1024, 4096, 16),
    (512, 8192, 16),
    (256, 16384, 16),
    (256, 1024, 16),
]
SHAPES = SINGLE_PASS_SHAPES + MULTIPASS_SHAPES + NT_SCALING_SHAPES

BITS = [4, 8]
SYMMETRIC = [True, False]

SEED = 0

_DEEPSPEED_BASELINE_VENDORS = {"nvidia"}

# The baseline is DeepSpeed's quantized_reduction kernel on the vendors that
# ship it; everywhere else, or when it cannot be built, there is no on-machine
# baseline and ``torch_op`` is None, so --base-data is the only comparison.


def _load_deepspeed_reduce():
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None, None

    try:
        from deepspeed.ops.op_builder import QuantizerBuilder

        mod = QuantizerBuilder().load()
        return mod.quantized_reduction, mod.QuantizationType
    except Exception as exc:
        warnings.warn(
            f"{flag_train.vendor_name!r} is expected to use DeepSpeed's "
            f"quantized_reduction as its baseline, but it could not be loaded: "
            f"{exc!r}. This run has no on-machine baseline; build deepspeed "
            f"to enable the quantized_reduction baseline.",
            stacklevel=2,
        )
        return None, None


_deepspeed_reduce, _deepspeed_quant_type = _load_deepspeed_reduce()

_BASELINE = (
    "deepspeed quantized_reduction"
    if _deepspeed_reduce is not None
    else "none (no on-machine baseline; --base-data is the only comparison)"
)


def train_reduction(
    data, scales, in_groups, out_groups, num_bits, quant_type, devices_per_node
):
    return flag_train.deepspeed.loco_quantized_reduction(
        data, scales, in_groups, out_groups, num_bits, quant_type, devices_per_node
    )


def _baseline(
    data, scales, in_groups, out_groups, num_bits, quant_type, devices_per_node
):
    """DeepSpeed's quantized_reduction kernel."""
    return _deepspeed_reduce(
        data,
        scales,
        in_groups,
        out_groups,
        num_bits,
        (
            _deepspeed_quant_type.Symmetric
            if int(quant_type) == int(QuantizationType.Symmetric)
            else _deepspeed_quant_type.Asymmetric
        ),
        devices_per_node,
    )


# ``None`` is the benchmark harness's "no on-machine baseline" sentinel.
torch_op = _baseline if _deepspeed_reduce is not None else None


def _make_inputs(out_groups, bpg, num_tensors, num_bits, symmetric, device):
    in_groups = out_groups * num_tensors
    epit = out_groups * bpg
    scales_elems = 1 if symmetric else 2

    gen = torch.Generator(device="cpu").manual_seed(SEED)
    data = torch.randint(
        -128, 128, (num_tensors * epit,), dtype=torch.int8, generator=gen
    ).to(device)

    scales = (
        torch.rand((in_groups * scales_elems,), dtype=torch.float32, generator=gen)
        .mul_(0.05)
        .add_(1e-3)
    )
    if not symmetric:
        scales = scales.view(-1, 2)
        scales = torch.stack([scales[:, 0], scales[:, 1] - 0.026], dim=1).reshape(-1)

    return (
        data,
        scales.contiguous().to(device),
        in_groups,
        out_groups,
        num_bits,
        QuantizationType.Symmetric if symmetric else QuantizationType.Asymmetric,
        num_tensors,
    )


class LocoQrBenchmark(base.GenericBenchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["gbps"]

    def set_shapes(self, shape_file=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "out_groups, elems_per_out_group, num_tensors"

    def set_more_shapes(self):
        return []

    def record_shapes(self, *args, **kwargs):
        data, scales, in_groups, out_groups, num_bits, qt, num_tensors = args
        bpg = data.numel() // num_tensors // out_groups
        path = "1pass" if bpg <= 8192 else "2pass"
        sym = "sym" if int(qt) == int(QuantizationType.Symmetric) else "asym"
        return (
            f"out_groups={out_groups} bpg={bpg} nt={num_tensors} "
            f"bits={num_bits} {sym} {path}"
        )

    def get_gbps(self, args, latency):
        data, scales, in_groups, out_groups, _num_bits, _qt, num_tensors = args
        scales_elems = scales.numel() // in_groups
        io_bytes = (
            data.numel()
            + scales.numel() * 4
            + data.numel() // num_tensors
            + out_groups * scales_elems * 4
        )
        return io_bytes * 1e-9 / (latency * 1e-3)


@pytest.mark.loco_quantized_reduction
@pytest.mark.parametrize("symmetric", SYMMETRIC)
@pytest.mark.parametrize("num_bits", BITS)
def test_loco_quantized_reduction_perf(num_bits, symmetric):
    print(f"\nBaseline: {_BASELINE}")

    def input_fn(shape, dtype, device):
        out_groups, bpg, num_tensors = shape
        yield _make_inputs(out_groups, bpg, num_tensors, num_bits, symmetric, device)

    bench = LocoQrBenchmark(
        op_name="loco_quantized_reduction",
        torch_op=torch_op,
        input_fn=input_fn,
        dtypes=[torch.int8],
    )
    bench.set_train(train_reduction)
    bench.run()
