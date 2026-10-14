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

import pytest
import torch

import flag_train
from flag_train.deepspeed import loco_quantized_reduction
from flag_train.deepspeed.loco_quantized_reduction import QuantizationType

from .. import base, consts

SINGLE_PASS_SHAPES = [
    (1024, 4096, 8),
    (1024, 4096, 2),
    (1024, 4096, 1),
    (2048, 2048, 8),
    (512, 2048, 8),
    (1024, 8192, 8),
    (256, 1024, 16),
]

MULTIPASS_SHAPES = [
    (512, 16384, 8),
    (256, 49152, 8),
]

SHAPES = SINGLE_PASS_SHAPES + MULTIPASS_SHAPES

BITS = [4, 8]
SYMMETRIC = [True, False]

SEED = 0

_DEEPSPEED_BASELINE_VENDORS = {"nvidia", "hygon"}


def _load_deepspeed_reduce():
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None, None

    try:
        from deepspeed.ops.op_builder import QuantizerBuilder

        mod = QuantizerBuilder().load()
        return mod.quantized_reduction, mod.QuantizationType
    except Exception as exc:
        raise RuntimeError(
            f"{flag_train.vendor_name!r} must use DeepSpeed's quantized_reduction "
            f"as its baseline, but it could not be loaded: {exc!r}. Build "
            f"deepspeed, or drop the backend from _DEEPSPEED_BASELINE_VENDORS."
        ) from exc


_deepspeed_reduce, _deepspeed_quant_type = _load_deepspeed_reduce()

_BASELINE = (
    "deepspeed quantized_reduction"
    if _deepspeed_reduce is not None
    else "torch composition"
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


def torch_reduction(
    data, scales, in_groups, out_groups, num_bits, quant_type, devices_per_node
):
    symmetric = int(quant_type) == int(QuantizationType.Symmetric)
    scales_elems = 1 if symmetric else 2

    epit = data.numel() // devices_per_node
    epog = epit // out_groups
    gpit = in_groups // devices_per_node

    q = _unpack(data.reshape(devices_per_node, out_groups, epog), num_bits).float()

    if symmetric:
        s = scales.reshape(devices_per_node, gpit)[:, :, None]
        acc = (q * s).sum(0)
        amax = acc.abs().amax(-1)
        qscale = torch.where(
            amax == 0, torch.ones_like(amax), (2**num_bits) / (2.0 * amax)
        )
        off = torch.zeros_like(qscale)
    else:
        sc = scales.reshape(devices_per_node, gpit, 2)
        s, o = sc[:, :, 0][:, :, None], sc[:, :, 1][:, :, None]
        acc = (q * s + o).sum(0)
        mx, mn = acc.amax(-1), acc.amin(-1)
        rng = mx - mn
        qscale = torch.where(rng == 0, torch.ones_like(rng), (2**num_bits) / rng)
        off = (mx + mn) / 2

    step = 1.0 / qscale
    lo_q, hi_q = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1
    qout = torch.round((acc - off[..., None]) * qscale[..., None]).clamp(lo_q, hi_q)

    packed = _pack(qout.to(torch.int32), num_bits).reshape(-1)
    return [packed, step.reshape(out_groups, scales_elems)]


def train_reduction(
    data, scales, in_groups, out_groups, num_bits, quant_type, devices_per_node
):
    return loco_quantized_reduction(
        data, scales, in_groups, out_groups, num_bits, quant_type, devices_per_node
    )


def torch_op(
    data, scales, in_groups, out_groups, num_bits, quant_type, devices_per_node
):
    if _deepspeed_reduce is not None:
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
    return torch_reduction(
        data, scales, in_groups, out_groups, num_bits, quant_type, devices_per_node
    )


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
        data, scales, in_groups, out_groups, num_bits, _qt, num_tensors = args
        bpg = data.numel() // num_tensors // out_groups
        path = "1pass" if bpg <= 8192 else "2pass"
        return (
            f"out_groups={out_groups} bpg={bpg} nt={num_tensors} "
            f"bits={num_bits} {path}"
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
