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
"""Performance benchmark for the evoformer attention operator.

Upstream has no performance test for this operator -- only
``tests/unit/ops/deepspeed4science/test_DS4Sci_EvoformerAttention.py``, which is
functional. So this file is new work, and the baseline is chosen the same way the
other DeepSpeed benchmarks here choose theirs:

* on a backend in ``_DEEPSPEED_BASELINE_VENDORS``, DeepSpeed's
  ``DS4Sci_EvoformerAttention`` **is** the baseline when it can be loaded. If it
  cannot be loaded there the module warns and runs without an on-machine
  baseline, leaving the comparison to ``--base-data``;
* on any other backend there is no on-machine baseline either. A plain-torch
  composition is not a valid baseline -- a ratio against it would say how far
  the kernel is from *a* correct implementation rather than from the best one --
  so ``torch_op`` is ``None`` and ``--base-data`` is the only comparison.

Only the forward is measured, because only the forward is ported: upstream's
``attention_bwd`` and its trainable entry are absent from this package.

FLOPS are not reported. The quantity is ``4 * B * N * H * L^2 * D`` (two matmuls,
counting a multiply-add as two), but the operator is
memory-bound at these sizes -- ``L`` is a few hundred and ``D`` at most 64 -- so
the number would flatter a kernel that is waiting on the bias matrices rather
than computing.
"""

import warnings

import pytest
import torch

import flag_train
import flag_train.deepspeed

from .. import base

# ``(batch, pairs, seq_len, heads, head_size)``.
#
# Upstream's two cases are in here. The rest move the axes this operator is made
# of, rather than sweeping sizes, because what the kernel does with a shape
# depends on which axis it is:
#
#   * ``N`` (pairs) is the evoformer axis and the one that costs without
#     computing: every pair reads the *same* ``bias2`` (``[B, 1, H, L, L]``, and
#     L x L per head regardless of N) and every pair tile attends over its own
#     L x L scores. So N is the "how much work per byte of bias" axis: the two
#     N = 1024 rows below are the pair-heavy end, and everything else is the
#     small-N end.
#   * ``L`` sets both the score matrix (L^2) and the bias2 read (L^2 per head), so
#     it is the axis the operator scales on -- and the one where it can least
#     afford to re-read. L = 1024 is there for that.
#   * ``L`` also has a floor: below the 64-row tile there is a single query tile,
#     so the grid collapses to one block per (head, pair) and any fixed per-CTA
#     cost shows up -- hence the two L = 64 rows.
#   * ``D`` caps at 64, and the wide end is where the qk^T dot is worth tensor
#     cores while the narrow end is mostly bias arithmetic.
#   * ``B`` only tells the two biases apart (``bias1`` is per pair, ``bias2`` per
#     batch), so one multi-batch row is enough.
_EVOFORMER_SHAPES = [
    (1, 32, 128, 4, 32),
    (1, 128, 256, 4, 32),
    (1, 256, 256, 4, 32),
    (1, 512, 256, 8, 8),
    (2, 64, 256, 8, 32),
    (1, 64, 384, 8, 64),
    # Pair-heavy: bias2 is read once per pair, so this is where sharing it pays.
    (1, 1024, 256, 4, 32),
    (1, 512, 64, 4, 32),
    # Long sequence, few pairs: the score matrix dominates instead.
    (1, 8, 1024, 4, 32),
    # A single query tile with the widest head.
    (1, 64, 64, 8, 64),
    # The large end of each axis, sized from what the operator is actually run
    # at rather than from a power of two: an MSA stack is hundreds of rows deep
    # and a protein is hundreds of residues long, so these are (MSA depth x
    # residues) at the top of each range.
    (1, 2048, 256, 4, 32),
    (1, 512, 384, 8, 32),
    (1, 4, 2048, 4, 32),
]

# The head size has to be a multiple of 8 and at most 64 (upstream's kMaxK), so
# 32 and 64 are the interesting widths and 8 is the narrow end where the dot
# product is mostly bias arithmetic.
_DTYPES = [torch.float16, torch.bfloat16]


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

# Backends whose baseline is DeepSpeed. Its operator is a CUDA extension builder,
# so only a backend that can compile and run one can host it. On these the
# baseline is expected, not mandatory: if it fails to load we warn and run with
# no on-machine baseline, leaving the comparison to --base-data.
_DEEPSPEED_BASELINE_VENDORS = {"nvidia"}


def _load_deepspeed_evoformer():
    """Harness-ready DS4Sci_EvoformerAttention, or ``None`` when unavailable.

    On a backend in ``_DEEPSPEED_BASELINE_VENDORS`` a load failure is warned about
    rather than raised: the benchmark still runs, with ``torch_op=None`` and
    ``--base-data`` as the only comparison.
    """
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None

    try:
        from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention
        from deepspeed.ops.op_builder import EvoformerAttnBuilder

        EvoformerAttnBuilder().load()
    except Exception as exc:
        warnings.warn(
            f"{flag_train.vendor_name!r} is expected to use DeepSpeed's "
            f"DS4Sci_EvoformerAttention as its baseline, but it could not be "
            f"loaded: {exc!r}. This run has no on-machine baseline; "
            f"--base-data is the only comparison. Build deepspeed with "
            f"CUTLASS >= 3.1 (`pip install nvidia-cutlass`) to enable the "
            f"DeepSpeed baseline.",
            stacklevel=2,
        )
        return None

    def torch_op(q, k, v, bias1, bias2):
        # A fresh list per call: upstream's entry appends to the one it is handed.
        return DS4Sci_EvoformerAttention(q, k, v, [bias1, bias2])

    return torch_op


# Resolved once, so the first-use JIT compile is not counted in the measurement;
# ``None`` is the benchmark harness's "no on-machine baseline" sentinel.
torch_op = _load_deepspeed_evoformer()

_BASELINE = (
    "deepspeed DS4Sci_EvoformerAttention"
    if torch_op is not None
    else "none (no on-machine baseline; --base-data is the only comparison)"
)


class EvoformerAttnBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _EVOFORMER_SHAPES
    DEFAULT_SHAPE_DESC = "batch, pairs, seq_len, heads, head_size"

    def set_shapes(self, shape_file=None):
        self.shapes = list(_EVOFORMER_SHAPES)

    def set_more_shapes(self):
        return []


def evoformer_attn_input_fn(shape, dtype, device):
    """One case's operands, in the two-bias arrangement upstream's test uses.

    ``bias1`` is the key-padding mask (large negative values on masked keys) and
    ``bias2`` the pair bias, drawn from a normal -- the same pair of roles the
    functional tests use, and the same two broadcasts the kernel has branches for.
    """
    batch, pairs, seq_len, heads, head_size = shape

    # A generator rather than a `manual_seed` call: the shapes are timed in
    # sequence, and reseeding the global RNG would make each case depend on how
    # many draws the previous one happened to make.
    generator = torch.Generator(device="cpu").manual_seed(0)

    def randn(*s):
        return torch.randn(*s, generator=generator, dtype=torch.float32).to(
            dtype=dtype, device=device
        )

    q = randn(batch, pairs, seq_len, heads, head_size)
    k = randn(batch, pairs, seq_len, heads, head_size)
    v = randn(batch, pairs, seq_len, heads, head_size)

    mask = torch.randint(0, 2, (batch, pairs, 1, 1, seq_len), generator=generator).to(
        device=device
    )
    bias1 = (1e9 * (mask - 1)).to(dtype)
    bias2 = randn(batch, 1, heads, seq_len, seq_len)

    yield q, k, v, bias1, bias2


def train_op(q, k, v, bias1, bias2):
    """The operator under test; returns the same output tensor as the baseline."""
    # A fresh list per call: upstream's public entry (and this port) pads the
    # list to two entries in place.
    return flag_train.deepspeed.evoformer_attn(q, k, v, [bias1, bias2])


@pytest.mark.evoformer_attn
def test_evoformer_attn_perf():
    print(f"\nBaseline: {_BASELINE}")

    bench = EvoformerAttnBenchmark(
        input_fn=evoformer_attn_input_fn,
        op_name="evoformer_attn",
        # None when DeepSpeed is unavailable: a torch composition is not timed
        # as a baseline, so --base-data supplies the comparison.
        torch_op=torch_op,
        dtypes=_DTYPES,
    )
    bench.set_train(train_op)
    bench.run()
