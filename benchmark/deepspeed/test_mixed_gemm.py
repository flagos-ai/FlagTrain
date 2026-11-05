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
"""Performance benchmark for the mixed_gemm operator.

Run ``tests/deepspeed/test_mixed_gemm.py`` green first, then::

    pytest benchmark/deepspeed/test_mixed_gemm.py -v -s

The baseline is DeepSpeed's own ``mixed_gemm`` on the backends in
``_DEEPSPEED_BASELINE_VENDORS`` -- the kernel this port is a port *of*. When it cannot be
loaded, or on any other backend, there is no on-machine baseline: a torch composition
(dequantize on device, then cuBLAS) is not timed, ``torch_op`` is ``None``, and
``--base-data`` is the only comparison. The report line says which case ran.

``base.py`` hands the same args/kwargs to ``torch_op`` and to ``train_op``, so the two must
have identical signatures; ``_call`` is what guarantees that here.
"""

import warnings

import pytest
import torch

import flag_train
from flag_train.deepspeed import mixed_gemm
from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn

from .. import base

# ``(m, k, n, num_bits)``. The bit width travels with the shape because the
# framework fixes its dtype list before the shapes are walked. Thin and past-L2
# are the two axes: thin is where quantization should win, past L2 is where the
# saved bandwidth can first turn into saved time.
_SHAPES = [
    # Thin: few tokens, wide weight -- the decoding shape, where the weight dominates traffic.
    (1, 4096, 4096, 4),
    (64, 4096, 4096, 4),
    (1, 4096, 4096, 8),
    (64, 4096, 4096, 8),
    # Square, walking past L2: fp16 weights are 2 MiB at 1024 cubed, 8 MiB at 2048 and
    # 32 MiB at 4096, so a part with a 50 MB L2 stops fitting near 3072.
    (1024, 1024, 1024, 4),
    (2048, 2048, 2048, 4),
    (4096, 4096, 4096, 4),
    (1024, 1024, 1024, 8),
    (2048, 2048, 2048, 8),
    (4096, 4096, 4096, 8),
    # Wide and deep: many tokens, a long reduction, 16384 output columns.
    (8192, 4096, 4096, 4),
    (2048, 4096, 16384, 4),
    (8192, 4096, 4096, 8),
    (2048, 4096, 16384, 8),
]

_ACT = 6  # IDENTITY: keep activation time out of the GEMM measurement

# Both: ``mixed_gemm.cu``'s ``ACT_DTYPE_SWITCH`` instantiates exactly these two.
_DTYPES = [torch.float16, torch.bfloat16]
_DTYPE = _DTYPES[0]  # the dtype the probe call and the byte table use


# ---------------------------------------------------------------------------
# Quantization
#
# ``weight`` reaches the operator already quantized *and* already run through
# ``preprocess_weights_for_mixed_gemm``; only the quantization half is reproduced
# here, because a timing does not need the layout. Arbitrary uint8 makes the kernel
# do the same work, and whether it reads the real layout is the tests' question.
# ---------------------------------------------------------------------------


def _round_half_away(x):
    """C++ ``std::round``: ties away from zero, unlike ``torch.round``."""
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


def quantize(w, num_bits):
    """``[k, n]`` fp16 -> ``([k, n] signed int64, scales fp16)`` per output channel.

    ``symmetric_quantize``'s rules: per-column ``max|w| / 2**(bits-1)``, rounded half
    away from zero, clamped to the signed range. Verified bit-for-bit against
    DeepSpeed.
    """
    wf = w.float()
    per_col_max = wf.abs().amax(dim=0)
    scale32 = per_col_max / float(1 << (num_bits - 1))
    scales = scale32.to(torch.float16)

    safe = torch.where(scale32 > 0, scale32, torch.ones_like(scale32))
    q = _round_half_away(wf / safe[None, :])
    q = q.clamp(-(1 << (num_bits - 1)), (1 << (num_bits - 1)) - 1)
    return q.to(torch.int64), scales


# ---------------------------------------------------------------------------
# The baseline, by platform
#
# One rule, on one predicate, matching the rule ``tests/deepspeed/test_mixed_gemm.py``
# uses to pick its oracle; here the tier decides what is **timed**, so the branch is
# exclusive.
#
# 1. A backend that runs DeepSpeed's own mixed_gemm uses that operator: it is the
#    operator being ported, and it is a real kernel rather than a torch composition.
# 2. Otherwise there is no on-machine baseline. A torch reference is a correct
#    implementation, not the best one, so it is not timed; ``torch_op`` is ``None``
#    and ``--base-data`` is the only comparison. The reason is reported at import.
# ---------------------------------------------------------------------------

# Backends whose baseline is DeepSpeed. ``mixed_gemm`` ships as a CUDA op, so only a
# backend that can compile and execute one can host it.
_DEEPSPEED_BASELINE_VENDORS = {"nvidia", "hygon"}

_DEEPSPEED_UNAVAILABLE_MSG = (
    "Its kernel ships prebuilt in the `dskernels` package, and "
    "`InferenceCutlassBuilder` links `-ldeepspeedft` rather than compiling it, so that "
    "package has to carry this GPU's compute capability. Install one that does, or drop "
    "the backend from _DEEPSPEED_BASELINE_VENDORS."
)


def _load_deepspeed_mixed_gemm():
    """``(op, reason)`` -- ``op`` is ``None`` when there is not a usable one.

    The kernel is prebuilt, so a library without this GPU's compute capability
    imports cleanly and fails only when called; it is therefore called once here, on
    random bytes.
    """
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None, None

    try:
        from deepspeed.ops.op_builder import InferenceCutlassBuilder

        op = InferenceCutlassBuilder().load().mixed_gemm
    except Exception as exc:  # noqa: BLE001
        return None, f"could not be loaded: {exc!r}"

    m = k = n = 128
    try:
        hidden = torch.randn(m, k, dtype=_DTYPE, device=runtime_device.name)
        weight = torch.randint(
            0, 256, (k, n // 2), dtype=torch.uint8, device=runtime_device.name
        )
        scales = torch.rand(n, dtype=_DTYPE, device=runtime_device.name) + 0.1
        out = torch.empty(m, n, dtype=_DTYPE, device=runtime_device.name)
        op(out, hidden, weight, scales, None, 4, _ACT)
        torch_device_fn.synchronize()
    except Exception as exc:  # noqa: BLE001
        return None, f"loaded but does not run here: {exc!r}"

    return op, None


# Resolved once, at module import time.
_DEEPSPEED_ORACLE, _DEEPSPEED_REASON = _load_deepspeed_mixed_gemm()

_WHY = (
    f"{flag_train.vendor_name!r} is expected to use DeepSpeed's mixed_gemm as its "
    f"baseline, which {_DEEPSPEED_REASON}. {_DEEPSPEED_UNAVAILABLE_MSG}"
)

if _DEEPSPEED_ORACLE is not None:
    _ORACLE = "deepspeed"
    _BASELINE = "deepspeed mixed_gemm"
else:
    _ORACLE = "none"
    _BASELINE = "none (no on-machine baseline; --base-data is the only comparison)"
    if flag_train.vendor_name in _DEEPSPEED_BASELINE_VENDORS:
        warnings.warn(
            f"{_WHY} This run has no on-machine baseline; a torch composition "
            f"is not timed, so --base-data is the only comparison.",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def mixed_gemm_input_fn(shape, dtype, device):
    """The inputs every implementation shares.

    A dict **inside a tuple**: ``base`` iterates the tuple to split args from kwargs,
    and iterating a bare dict would yield its keys instead. The shape is quantized
    once per yield and ``output`` is reused, since the operator writes it in place.
    """
    m, k, n, num_bits = shape
    hidden = torch.randn(m, k, dtype=dtype, device=device)
    w = torch.randn(k, n, dtype=dtype, device=device)
    _, scales = quantize(w.cpu(), num_bits)
    # Only the buffer's length matters to a timing: the kernel reads it through its own
    # index arithmetic, and whether it reads the real layout is the tests' question.
    packed = torch.randint(
        0, 256, (k, n // (2 if num_bits == 4 else 1)), dtype=torch.uint8, device=device
    )
    scales = scales.to(device).to(dtype)
    output = torch.empty(m, n, dtype=dtype, device=device)

    yield (
        {
            "hidden": hidden,
            "weight": packed,
            "scales": scales,
            "bias": None,
            "output": output,
            "num_bits": num_bits,
        },
    )


def _call(op, **kwargs):
    """Hand the inputs to an implementation in the operator's own order.

    Every implementation goes through here, which is what keeps their signatures
    identical.
    """
    return op(
        kwargs["output"],
        kwargs["hidden"],
        kwargs["weight"],
        kwargs["scales"],
        kwargs["bias"],
        kwargs["num_bits"],
        _ACT,
    )


# ---------------------------------------------------------------------------
# Baseline and the operator under test
# ---------------------------------------------------------------------------


def _baseline(**kwargs):
    """DeepSpeed's kernel; only called when it could be loaded."""
    return _call(_DEEPSPEED_ORACLE, **kwargs)


# ``None`` is the benchmark harness's "no on-machine baseline" sentinel.
torch_op = _baseline if _DEEPSPEED_ORACLE is not None else None


def train_op(**kwargs):
    """The operator under test."""
    return _call(mixed_gemm, **kwargs)


# ---------------------------------------------------------------------------
# Benchmark body
# ---------------------------------------------------------------------------


class MixedGemmBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _SHAPES
    DEFAULT_SHAPE_DESC = "m, k, n, num_bits"

    def set_shapes(self, shape_file=None):
        self.shapes = list(_SHAPES)

    def set_more_shapes(self):
        # The base class otherwise adds 3-D and 2**28-shaped cases; ours are given.
        return []


def _peak_memory_mb():
    """Peak device memory since the last reset, in MiB.

    The framework does not report this, and saving memory is half of what weight-only
    quantization is for.
    """
    try:
        return torch.cuda.max_memory_allocated() / (1 << 20)
    except Exception:  # noqa: BLE001
        return float("nan")


def _weight_bytes(shape, num_bits):
    """``(packed_bytes, fp16_bytes)`` for the weight of ``shape``."""
    _, k, n, _ = shape
    packed = k * ((n + 1) // 2 if num_bits == 4 else n)
    return packed, k * n * 2


@pytest.mark.mixed_gemm
def test_mixed_gemm_perf():
    """Time the operator against the baseline. See the module docstring."""
    # The tier goes first, so a moved number says whether its baseline moved too.
    print(f"\nOracle: {_ORACLE}\nBaseline: {_BASELINE}")
    print(f"  dtypes={[str(d) for d in _DTYPES]}  act=IDENTITY (id 6)")
    print("  weight size per shape (int4/int8, and the fp16 it replaces):")
    for shape in _SHAPES:
        m, k, n, num_bits = shape
        packed, fp16 = _weight_bytes(shape, num_bits)
        print(
            f"    {m}x{k}x{n} int{num_bits} {packed / (1 << 20):6.2f} MiB vs "
            f"fp16 {fp16 / (1 << 20):6.2f} MiB ({fp16 / packed:.1f}x smaller)"
        )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    bench = MixedGemmBenchmark(
        input_fn=mixed_gemm_input_fn,
        op_name="mixed_gemm",
        # None when DeepSpeed is unavailable: a torch composition is not timed
        # as a baseline, so --base-data supplies the comparison.
        torch_op=torch_op,
        dtypes=_DTYPES,
    )
    bench.set_train(train_op)
    bench.run()

    peak = _peak_memory_mb()
    if peak == peak:  # not NaN
        print(f"  peak device memory: {peak:.1f} MiB")


@pytest.mark.mixed_gemm
@pytest.mark.skipif(_DEEPSPEED_ORACLE is None, reason=_DEEPSPEED_UNAVAILABLE_MSG)
def test_mixed_gemm_vs_deepspeed():
    """Report DeepSpeed's own operator on the same shapes.

    Separate from the ratio above: both sides read the same preprocessed B layout, so
    this compares kernels over identical inputs.
    """
    print(f"\nDeepSpeed oracle: {_ORACLE}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    bench = MixedGemmBenchmark(
        input_fn=mixed_gemm_input_fn,
        op_name="mixed_gemm",
        # DeepSpeed's operator takes the same seven arguments ours does, so it goes
        # through `_call` unchanged -- nothing to bind, it reads the packed buffer.
        torch_op=lambda **kw: _call(_DEEPSPEED_ORACLE, **kw),
        dtypes=_DTYPES,
    )
    bench.set_train(train_op)
    bench.run()
