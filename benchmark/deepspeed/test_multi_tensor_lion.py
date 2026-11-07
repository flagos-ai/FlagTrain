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
"""Performance benchmark for the multi_tensor_lion operator.

The baseline follows ``_DEEPSPEED_BASELINE_VENDORS`` below:

* on a listed backend, DeepSpeed's ``multi_tensor_lion`` **is** the baseline.
  If it cannot be built there the loader warns and the run continues with no
  on-machine baseline, so a missing kernel is reported rather than hidden
  behind another measurement;
* on any other backend there is no on-machine baseline. A plain-torch
  composition of the same contract is not timed: a ratio against it says how
  far the kernel is from *a* correct implementation rather than from the best
  one. ``torch_op`` is ``None``, so the framework reports Train latency only
  and ``--base-data`` is the only comparison.

The benchmark times repeated steps on a fixed tensor set, which is the
training-loop pattern: the operator's device-side metadata is built on the
first call and served from the cache on every later call, so the timed region
is the single fused kernel launch. DeepSpeed's op rebuilds its host-side
pointer arrays on every call; that asymmetry is the optimization being
measured, not a benchmark artifact.
"""

import warnings

import pytest
import torch

import flag_train

from .. import base

# (n_tensors, numel) cases: small tensor counts with tiny/large tensors up to
# a realistic training scale (256 tensors x 1M elements).
_MULTI_TENSOR_LION_SHAPES = [
    (8, 1024),
    (64, 1024),
    (256, 1024),
    (1024, 1024),
    (8, 1048576),
    (64, 1048576),
    (256, 1048576),
]

# Fixed hyper-parameters shared by both implementations so the comparison is
# apples-to-apples. These mirror DeepSpeed's own FusedLion defaults.
_LR = 1e-4
_BETA1 = 0.9
_BETA2 = 0.99
_WEIGHT_DECAY = 0.01
_CHUNK_SIZE = 2048
_STEP = 1

# Backends whose baseline is DeepSpeed. multi_tensor_lion ships as a CUDA op
# builder, so only a backend that can compile and execute one can host it. A
# load failure is reported, not raised: with no runnable kernel there is no
# on-machine baseline, and a torch composition is not timed.
_DEEPSPEED_BASELINE_VENDORS = {"nvidia", "hygon"}


def _load_deepspeed_multi_tensor_lion():
    """Harness-ready multi_tensor_lion, or ``None`` when it cannot be loaded.

    ``FusedLionBuilder`` JIT-compiles the CUDA source shipped inside the
    ``deepspeed`` package, then reuses the build cached under
    ``torch_extensions``.
    """
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None

    try:
        from deepspeed.ops.op_builder import FusedLionBuilder

        multi_tensor_lion = FusedLionBuilder().load().multi_tensor_lion
    except Exception as exc:
        warnings.warn(
            f"{flag_train.vendor_name!r} is expected to use DeepSpeed's "
            f"multi_tensor_lion as its baseline, but it could not be loaded: "
            f"{exc!r}. This run has no on-machine baseline; a torch "
            f"composition is not timed, so --base-data is the only comparison.",
            stacklevel=2,
        )
        return None

    def torch_op(gs, ps, ms):
        return _call(multi_tensor_lion, gs, ps, ms)

    return torch_op


# Resolved once, so the first-use JIT compile is not counted in the measurement;
# ``None`` is the benchmark harness's "no on-machine baseline" sentinel.
torch_op = _load_deepspeed_multi_tensor_lion()

_BASELINE = (
    "deepspeed multi_tensor_lion"
    if torch_op is not None
    else "none (no on-machine baseline; --base-data is the only comparison)"
)


class MultiTensorLionBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _MULTI_TENSOR_LION_SHAPES

    def set_shapes(self, shape_file=None):
        # multi_tensor_lion needs 3 lists of tensors per case (grads, params, exp_avgs);
        # keep the (n_tensors, numel) cases explicit to avoid OOM on CI GPUs.
        self.shapes = list(_MULTI_TENSOR_LION_SHAPES)

    def set_more_shapes(self):
        return []

    def record_shapes(self, *args, **kwargs):
        # The inputs are (grads, params, exp_avgs) lists of n_tensors tensors;
        # the default deep_parse would print every tensor's shape, which is
        # unreadable at 1024 tensors. Compress to the case description.
        gs = args[0]
        return f"{len(gs)} tensors x {list(gs[0].shape)}"


def multi_tensor_lion_input_fn(shape, dtype, device):
    n_tensors, numel = shape
    gs = [torch.randn((numel,), dtype=dtype, device=device) for _ in range(n_tensors)]
    ps = [torch.randn((numel,), dtype=dtype, device=device) for _ in range(n_tensors)]
    ms = [torch.zeros((numel,), dtype=dtype, device=device) for _ in range(n_tensors)]
    yield gs, ps, ms


_NOOP_FLAG = None


def _noop_flag():
    global _NOOP_FLAG
    if _NOOP_FLAG is None:
        _NOOP_FLAG = torch.zeros(1, dtype=torch.int32)
    return _NOOP_FLAG


def _call(op, gs, ps, ms):
    return op(
        _CHUNK_SIZE,
        _noop_flag(),
        [gs, ps, ms],
        _LR,
        _BETA1,
        _BETA2,
        _STEP,
        _WEIGHT_DECAY,
    )


def train_op(gs, ps, ms):
    """The operator under test."""
    return _call(flag_train.deepspeed.multi_tensor_lion, gs, ps, ms)


@pytest.mark.multi_tensor_lion
def test_multi_tensor_lion_perf():
    print(f"\nBaseline: {_BASELINE}")

    bench = MultiTensorLionBenchmark(
        input_fn=multi_tensor_lion_input_fn,
        op_name="multi_tensor_lion",
        # None when DeepSpeed is unavailable: a torch composition is not timed
        # as a baseline, so --base-data supplies the comparison.
        torch_op=torch_op,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )
    bench.set_train(train_op)
    bench.run()
