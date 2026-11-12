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
"""Performance benchmark for the LAMB operator.

The baseline follows ``_DEEPSPEED_BASELINE_VENDORS`` below and mirrors the
torch_op / ``--base-data`` matrix of FlagGems-vllm:

* on a listed backend, DeepSpeed's ``fused_lamb`` **is** the baseline when it can be
  built. It is a real on-machine reference, so the local speedup is reported, and
  ``--base-data`` additionally produces ``compared_speedup`` when it has a record;
* everywhere else -- including a listed backend where ``fused_lamb`` cannot be
  loaded -- ``torch_op`` is ``None``: the framework then measures Train latency
  only (``latency_base`` and ``speedup`` are reported as N/A) and ``--base-data``
  becomes the only available comparison. Without ``--base-data`` the benchmark
  still passes; with it, a matching record produces ``compared_speedup`` while a
  missing one fails the benchmark -- there is no baseline at all in that case.
  A plain-torch composition of the contract is not timed either: a ratio against
  it would say how far the kernel is from *a* correct implementation rather than
  from the best one.

The baseline takes the same argument tuple as the operator, so unlike the
blocked-flash benchmark there is nothing to pre-compute outside the timed region.
"""

import pytest
import torch

import flag_train
from flag_train.deepspeed import lamb
from flag_train.runtime import device as runtime_device

from .. import base

# One-dimensional parameter tensors of realistic optimizer sizes. LAMB (like the
# other fused optimizers) operates on a flattened parameter array, so the element
# count is the only shape dimension that matters.
_LAMB_SHAPES = [
    (1024,),
    (4096,),
    (16384,),
    (65536,),
    (262144,),
    (1048576,),
    (11048576,),
]

# Fixed hyper-parameters shared by both implementations so the comparison is
# apples-to-apples. These mirror DeepSpeed's own fused_lamb defaults.
_LR = 1e-3
_BETA1 = 0.9
_BETA2 = 0.999
_MAX_COEFF = 10.0
_MIN_COEFF = 0.01
_EPS = 1e-8
_GRAD_SCALE = 1.0
_STEP = 1
_MODE = 1  # eps outside the sqrt (DeepSpeed default)
_BIAS_CORRECTION = 1
_DECAY = 0.01

# Backends whose baseline is DeepSpeed. fused_lamb ships as a CUDA op builder, so
# only a backend that can compile and execute one can host it.
_DEEPSPEED_BASELINE_VENDORS = {"nvidia", "hygon"}


def _load_deepspeed_lamb():
    """Harness-ready fused_lamb and its load failure, if it failed.

    ``FusedLambBuilder`` JIT-compiles the CUDA source shipped inside the
    ``deepspeed`` package, then reuses the build cached under ``torch_extensions``.
    A failure is reported, not raised: on a backend in
    ``_DEEPSPEED_BASELINE_VENDORS`` it makes the vendor-native baseline missing,
    which the module resolves below.
    """
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None, None

    try:
        from deepspeed.ops.op_builder import FusedLambBuilder

        fused_lamb = FusedLambBuilder().load().lamb
    except Exception as exc:  # noqa: BLE001
        return None, f"could not be loaded: {exc!r}"

    def torch_op(p, p_copy, m, v, g):
        return _call(fused_lamb, p, p_copy, m, v, g)

    return torch_op, None


# Resolved once, so the first-use JIT compile is not counted in the measurement;
# ``None`` is the benchmark harness's "no on-machine baseline" sentinel.
torch_op, _DEEPSPEED_REASON = _load_deepspeed_lamb()

# Whether this platform owes a DeepSpeed baseline that is not there. On such a
# backend ``--base-data`` is the only comparison available.
_DEEPSPEED_BASELINE_MISSING = (
    flag_train.vendor_name in _DEEPSPEED_BASELINE_VENDORS and torch_op is None
)

if torch_op is not None:
    _BASELINE = "deepspeed fused_lamb"
elif _DEEPSPEED_BASELINE_MISSING:
    # ``torch_op=None``: the framework measures Train latency only and reports
    # latency_base/speedup as N/A. ``--base-data`` then supplies the comparison,
    # and a baseline without a matching record fails -- there is nothing else.
    _BASELINE = "none (vendor-native missing; --base-data is the only baseline)"
    _MISSING_NOTE = (
        f"{flag_train.vendor_name!r} is expected to use DeepSpeed's fused_lamb as "
        f"its baseline, but it {_DEEPSPEED_REASON}.\n"
        "This run has no on-machine baseline: latency_base and speedup are\n"
        "reported as N/A, and the comparison comes from --base-data (a shape\n"
        "without a baseline record fails the benchmark). No torch composition\n"
        "is used as a substitute."
    )
    print("\n" + "!" * 78)
    for _line in _MISSING_NOTE.splitlines():
        print(f"!! {_line}".rstrip())
    print("!" * 78)
else:
    # Any other backend has no vendor-native op; the torch composition is not a
    # baseline either, so --base-data is the only comparison.
    _BASELINE = "none (no vendor-native baseline; --base-data is the only baseline)"


class LambBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _LAMB_SHAPES

    def set_shapes(self, shape_file=None):
        # lamb needs 5 tensors per case (param, p_copy, m, v, g); keep the shapes
        # moderate and explicit to avoid OOM on CI GPUs.
        self.shapes = list(_LAMB_SHAPES)

    def set_more_shapes(self):
        return []


def lamb_input_fn(shape, dtype, device):
    p = torch.randn(shape, dtype=dtype, device=device)
    p_copy = torch.empty((0,), dtype=dtype, device=device)  # skip the copy
    m = torch.zeros(shape, dtype=dtype, device=device)
    v = torch.zeros(shape, dtype=dtype, device=device)
    g = torch.randn(shape, dtype=dtype, device=device)
    yield p, p_copy, m, v, g


def _call(op, p, p_copy, m, v, g):
    return op(
        p,
        p_copy,
        m,
        v,
        g,
        _LR,
        _BETA1,
        _BETA2,
        _MAX_COEFF,
        _MIN_COEFF,
        _EPS,
        _GRAD_SCALE,
        _STEP,
        _MODE,
        _BIAS_CORRECTION,
        _DECAY,
    )


def train_op(p, p_copy, m, v, g):
    """The operator under test."""
    return _call(lamb, p, p_copy, m, v, g)


@pytest.mark.lamb
def test_lamb_perf():
    print(f"\nBaseline: {_BASELINE}")

    bench = LambBenchmark(
        input_fn=lamb_input_fn,
        op_name="lamb",
        # None when this backend has no vendor-native baseline: the framework
        # then skips the baseline timing (N/A) and leaves the comparison to
        # --base-data.
        torch_op=torch_op,
        # fp32 is the common baseline; fp64 is measured where the device
        # declares fp64 support (NVIDIA/Hygon).  The fp16-gradient + fp32-master
        # combination is correctness-only, so it is not benchmarked here.
        dtypes=[torch.float32]
        + ([torch.float64] if runtime_device.support_fp64 else []),
    )
    bench.set_train(train_op)
    bench.run()
