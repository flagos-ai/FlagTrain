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

Run ``tests/deepspeed/test_mixed_gemm.py`` green first.

Run with::

    pytest benchmark/deepspeed/test_mixed_gemm.py -v -s      # -s to see the prints

The baseline
------------

The baseline is ``_BoundReference``: dequantize the weight on device, then cuBLAS. It
works from the *logical* quantized matrix -- the one the packed buffer was built from --
so it does no layout arithmetic, and the operator is the only side that has to get the
layout right. Both produce one ``[m, n]`` output over the same logical product, so the
only thing that differs is *where the dequantization happens*: on the device before the
GEMM, or folded into the kernel's operand fetch. That is what makes the ratio a statement
about this kernel.

The cost of quantization itself is a different question and is not answered here.
Comparing against an unquantized ``torch.mm`` would put a different amount of work
on each side of the ratio, so it would need its own benchmark under its own name,
with the difference in work stated alongside it.

A framework behaviour worth knowing
-----------------------------------

``base.py`` lines 590 and 593 hand the **same** args/kwargs to ``torch_op`` and to
``train_op``, so the two must have identical signatures. ``lamb`` guarantees that
by funnelling everything through ``_call``, and so does this file.

A measurement to reproduce on the target hardware
-------------------------------------------------

On a local RTX 4060 Laptop at 2048 cubed with int4 weights, the operator was
*slower* than the naive baseline. 2048-squared fp16 weights are only 8 MiB and fit
in L2, so the 4x smaller weight is never re-read from HBM: the saved bandwidth
buys nothing and the per-element dequantization is pure cost. Quantization only
turns saved bandwidth into saved time once the working set exceeds L2, which is
why the shape list below includes a case that large.

That number does not transfer to another part -- re-measure it here.
"""

import os

import pytest
import torch

import flag_train
from flag_train.deepspeed import mixed_gemm
from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn

from .. import base

# ``(m, k, n, num_bits)``. The framework fixes its dtype list before the shapes are
# walked, so the bit width travels with the shape rather than in a module constant --
# both precisions have to appear in one run.
#
# The set is chosen around two axes that pull in opposite directions: **how thin the
# GEMM is** (tokens against weight) and **whether the weight exceeds L2**. Thin is where
# quantization should win, because the weight is nearly all of the traffic; past L2 is
# where the saved bandwidth can first turn into saved time. Both bit widths appear on
# every axis, since int4 and int8 differ by 2x in weight size and take different
# nibble-vs-byte paths through the kernel.
_SHAPES = [
    # Thin: few tokens against a wide weight, which is the decoding shape and the one
    # weight-only quantization exists for. Here the weight dominates both the traffic and
    # the footprint, so the 4x/2x smaller buffer is most of the story.
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
    # Wide and deep: many tokens, a long reduction, and an output far wider than the
    # token count. 16384 columns is where the per-output-channel scale load and the
    # n-dimension grid start to matter.
    (8192, 4096, 4096, 4),
    (2048, 4096, 16384, 4),
    (8192, 4096, 4096, 8),
    (2048, 4096, 16384, 8),
]

_ACT = 6  # IDENTITY: keep activation time out of the GEMM measurement

# Both, because ``mixed_gemm.cu``'s ``ACT_DTYPE_SWITCH`` instantiates exactly these two
# and they take different epilogue paths; a kernel tuned against one says nothing about
# the other. The correctness tests cover both for the same reason.
_DTYPES = [torch.float16, torch.bfloat16]
_DTYPE = _DTYPES[0]  # the dtype the probe call and the byte table use


# ---------------------------------------------------------------------------
# Quantization
#
# ``weight`` reaches the operator already quantized *and* already run through
# ``preprocess_weights_for_mixed_gemm``. Only the quantization half is reproduced here,
# because a timing does not need the layout.
#
# What the operator is handed is a uint8 buffer of the right length and nothing more.
# The kernel reads it through its own index arithmetic, so arbitrary bytes make it do
# exactly the same work; building the real layout would cost minutes at 4096 cubed and
# change nothing that is being measured. Whether the operator reads that layout
# correctly is the tests' question, and they answer it at 64x64 where the transform is
# cheap.
#
# ``quantize`` itself is kept honest -- it feeds the baseline's dequantization, and the
# tests pin it against DeepSpeed's ``symmetric_quantize`` bit for bit.
# ---------------------------------------------------------------------------


def _round_half_away(x):
    """C++ ``std::round``: ties away from zero, unlike ``torch.round``."""
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


def quantize(w, num_bits):
    """``[k, n]`` fp16 -> ``([k, n] signed int64, scales fp16)``, per output channel.

    ``symmetric_quantize``'s rules: per-column ``max|w| / 2**(bits-1)``, rounded half
    away from zero, clamped to the signed range. Verified bit-for-bit against
    DeepSpeed: scales and values both, int4 and int8, 4096 elements each.
    """
    wf = w.float()
    per_col_max = wf.abs().amax(dim=0)
    scale32 = per_col_max / float(1 << (num_bits - 1))
    scales = scale32.to(torch.float16)

    safe = torch.where(scale32 > 0, scale32, torch.ones_like(scale32))
    q = _round_half_away(wf / safe[None, :])
    q = q.clamp(-(1 << (num_bits - 1)), (1 << (num_bits - 1)) - 1)
    return q.to(torch.int64), scales


def dequantize_logical(q, scales):
    """Logical quantized matrix -> the fp16 weight, for the reference baseline.

    Takes the matrix the input builder already has, so the baseline never touches the
    packed buffer and its timing is dequantize-plus-GEMM and nothing else.
    """
    return q.to(scales.dtype) * scales[None, :]


# ---------------------------------------------------------------------------
# The baseline, by platform
#
# One rule, in one order, on one predicate -- the same rule
# ``tests/deepspeed/test_mixed_gemm.py`` picks the oracle it holds the operator to,
# so a reader comparing the two files sees one statement rather than two that happen to
# resemble each other. The difference is what the choice *does*: the tests run every
# oracle they can and skip the ones they cannot, while here the tier decides what is
# **timed**, so the branch is exclusive.
#
# 1. On a backend that runs DeepSpeed's own mixed_gemm, that operator **is** the
#    baseline. It is the operator being ported, so it is the only thing that answers
#    "what did the port cost" -- and it is *required*: if it cannot be loaded this
#    module raises rather than timing the torch reference and reporting it under that
#    name, which would answer a different question.
# 2. On any other backend, the torch reference. It is not a competitor, and the ratio
#    there says how far the kernel is from *a* correct implementation rather than from
#    the best one.
#
# There is no middle tier the way the blocked-flash benchmark has one: that file falls
# back to ``flash_attn``, a real third-party competitor. Nothing comparable exists for
# a weight-only quantized GEMM at this contract.
#
# Why tier 1 may not load is its own subject -- a prebuilt kernels library that lacks
# this GPU's compute capability imports cleanly and fails only when called.
# ``_load_deepspeed_mixed_gemm`` does the lookup and the trial call, and this module
# turns its answer into a raise on the backends where the DeepSpeed baseline is not
# optional.
#
# ``FLAG_TRAIN_ALLOW_TORCH_BASELINE=1`` turns that raise into a fallback. It exists for a
# development machine that cannot host the DeepSpeed baseline at all: without it there is
# no way to obtain *any* number there, and the alternative people reach for -- editing the
# vendor set -- quietly makes the two files disagree about which baseline applies where.
# It is not a silent fallback: the reason is printed, the baseline is named everywhere it
# appears, and every result row carries it, so a number that travels out of this terminal
# cannot be mistaken for one measured against DeepSpeed.
# ---------------------------------------------------------------------------

# Backends whose baseline is DeepSpeed. ``mixed_gemm`` ships as a CUDA op, so only a
# backend that can compile and execute one can host it.
_DEEPSPEED_BASELINE_VENDORS = {"nvidia", "hygon"}

# Whether an unusable DeepSpeed baseline is fatal on those backends. Default true: a
# dedicated environment should fail loudly rather than report a different measurement.
_ALLOW_TORCH_BASELINE = os.environ.get(
    "FLAG_TRAIN_ALLOW_TORCH_BASELINE", ""
).strip().lower() in ("1", "true", "yes")

_DEEPSPEED_UNAVAILABLE_MSG = (
    "Its kernel ships prebuilt in the `dskernels` package, and "
    "`InferenceCutlassBuilder` links `-ldeepspeedft` rather than compiling it, so that "
    "package has to carry this GPU's compute capability. Install one that does, or drop "
    "the backend from _DEEPSPEED_BASELINE_VENDORS."
)


def _load_deepspeed_mixed_gemm():
    """``(op, reason)`` -- ``op`` is ``None`` when there is not a usable one.

    Import success is not enough: the kernel is prebuilt, so a library without this GPU's
    compute capability imports cleanly and fails only when called. It is therefore called
    once here, on random bytes -- the *values* mean nothing, but a signal or an exception
    is exactly the finding, and ``reason`` carries it.
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

_REQUIRED = flag_train.vendor_name in _DEEPSPEED_BASELINE_VENDORS

# Whether this platform owes a DeepSpeed baseline that is not there. Recorded rather
# than raised: raising here happens at *import*, which is collection time for pytest,
# and the repository's CI maps a changed benchmark file onto
# ``benchmark/deepspeed/<file>`` and runs it (`tools/ci_checks/select_changed_tests.py`).
# An import-time raise turns the whole perf stage into a collection error, which says
# nothing about this operator.
_BASELINE_REQUIRED_BUT_UNAVAILABLE = _REQUIRED and _DEEPSPEED_ORACLE is None

_WHY = (
    f"{flag_train.vendor_name!r} must use DeepSpeed's mixed_gemm as its baseline, "
    f"which {_DEEPSPEED_REASON}. {_DEEPSPEED_UNAVAILABLE_MSG}"
)

# The framework's own way of saying "no native baseline on this platform": the autouse
# ``configure_native_baseline`` fixture reads this marker per test and sets
# ``Config.skip_native``, which makes ``base.py`` skip the ``torch_op`` timing and
# ``consts.py`` print ``Native baseline: N/A (<reason>)`` in the report header. The
# operator under test is still timed, so the module reports what it can rather than
# failing the whole stage over a baseline the platform cannot load.
#
# Appending to ``pytestmark`` at import time is enough because the fixture runs later:
# import happens during collection, the fixture before each test. Gated on
# ``_DEEPSPEED_ORACLE is None`` rather than on the vendor, so a platform where the
# upstream kernel *does* load keeps it as the baseline.
if _BASELINE_REQUIRED_BUT_UNAVAILABLE and not _ALLOW_TORCH_BASELINE:
    _NATIVE_SKIP_REASON = (
        f"DeepSpeed's mixed_gemm {_DEEPSPEED_REASON}; {_DEEPSPEED_UNAVAILABLE_MSG}"
    )
    # Appended rather than assigned: a future module-level marker added above would
    # otherwise be silently dropped by this branch.
    pytestmark = globals().get("pytestmark", [])
    pytestmark.append(
        pytest.mark.skip_native(
            vendors=flag_train.vendor_name, reason=_NATIVE_SKIP_REASON
        )
    )


if _BASELINE_REQUIRED_BUT_UNAVAILABLE and _ALLOW_TORCH_BASELINE:
    _FALLBACK_NOTE = (
        "FLAG_TRAIN_ALLOW_TORCH_BASELINE=1 is set, so this run measures against the\n"
        "TORCH REFERENCE instead. It is NOT a competitor: it materializes the\n"
        "dequantized weight and reads it back, which is exactly the round trip this\n"
        "operator's fusion removes. Every ratio below says how far the kernel is\n"
        "from *a* correct implementation, not from the best one.\n"
        "The upstream comparison needs a compute capability the prebuilt kernels\n"
        "cover (sm_80 / sm_86) -- re-run there before quoting these numbers as a\n"
        "port's cost."
    )
    print("\n" + "!" * 78)
    for _line in (_WHY + "\n\n" + _FALLBACK_NOTE).splitlines():
        print(f"!! {_line}".rstrip())
    print("!" * 78)

if _DEEPSPEED_ORACLE is not None:
    _ORACLE = "deepspeed"
    _BASELINE = "deepspeed mixed_gemm"
else:
    _ORACLE = "torch_ref"
    _BASELINE = "torch reference (dequantize on device, then cuBLAS)"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def mixed_gemm_input_fn(shape, dtype, device):
    """The inputs every implementation shares.

    A dict **inside a tuple**: ``base`` does
    ``for item in input_tuple: ... elif isinstance(item, dict): kwargs.update(item)``,
    and iterating a bare dict yields its keys, which then arrive as positional
    arguments and every op fails on a signature mismatch.

    The shape is quantized exactly once per yield so both implementations read the
    same weight, and ``output`` is the same buffer every time -- the operator
    writes it in place, and re-allocating inside the timed region would measure
    the allocator.
    """
    m, k, n, num_bits = shape
    hidden = torch.randn(m, k, dtype=dtype, device=device)
    w = torch.randn(k, n, dtype=dtype, device=device)
    q, scales = quantize(w.cpu(), num_bits)
    # The buffer's *contents* are irrelevant to a timing: the kernel reads it through its
    # own index arithmetic, so any uint8 of the right length makes it do the same work.
    # Running the real preprocessing here would cost minutes at 4096 cubed and change
    # nothing that is being measured. Whether the operator reads the layout correctly is
    # the tests' question, and they answer it at a size where the transform is cheap.
    packed = torch.randint(
        0, 256, (k, n // (2 if num_bits == 4 else 1)), dtype=torch.uint8, device=device
    )
    scales = scales.to(device).to(dtype)
    output = torch.empty(m, n, dtype=dtype, device=device)

    # ``q`` is the logical matrix the baseline dequantizes. It is what the buffer above
    # would hold if it had been built properly, so the two sides do the same amount of
    # arithmetic even though the operator's output is meaningless on this input.
    #
    # Stored at the width it actually needs, not at the int64 ``quantize`` returns. The
    # baseline converts this to fp16 on every call, and an int64 intermediate would make
    # that conversion -- not the dequantization -- the dominant cost: 8 bytes per element
    # to read against 2 written, for a step the operator does not have at all. A real
    # quantized weight is stored the way it was quantized to, which is what int8 is here.
    yield (
        {
            "hidden": hidden,
            "weight": packed,
            "q": q.to(torch.int8).to(device),
            "scales": scales,
            "bias": None,
            "output": output,
            "num_bits": num_bits,
        },
    )


def _call(op, **kwargs):
    """Hand the inputs to an implementation in the operator's own order.

    Every implementation goes through here, which is what keeps their signatures
    identical -- ``base`` passes one argument tuple to both ``torch_op`` and
    ``train_op``.
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


class _BoundReference:
    """The baseline, holding the logical weight so its signature can stay the operator's.

    ``base`` hands one argument tuple to both ``torch_op`` and ``train_op``, so the two
    must accept exactly the same call -- the operator takes seven arguments and none of
    them is the logical matrix. Binding that matrix to the object keeps the shared call
    identical while the baseline still works from the logical values rather than from the
    packed buffer.

    It reads the *same* logical product the operator computes, so the ratio isolates the
    kernel. The dequantization stays **inside** the timed region: moving it out would
    measure a bare cuBLAS GEMM against the operator's dequantize-plus-GEMM, which is a
    different question, and one this ratio would then answer wrongly.
    """

    def __init__(self, q):
        self.q = q

    def __call__(
        self, output, hidden_states, weight, scales, bias, num_bits, activation_raw
    ):
        torch.mm(hidden_states, dequantize_logical(self.q, scales), out=output)
        if bias is not None:
            output.add_(bias)
        return output


def torch_op(**kwargs):
    """Baseline, chosen by platform. See the module docstring.

    The DeepSpeed branch reads the same preprocessed buffer the operator under test
    reads, so both sides do one ``[m, k] @ [k, n]`` product over identical inputs and
    the ratio isolates the kernel.

    The torch branch works from the logical matrix instead. ``q`` comes off the kwargs
    rather than through ``_call``: ``base`` hands one argument tuple to both ops, so the
    baseline cannot take an extra parameter, and binding it to the object keeps the
    shared call identical.
    """
    if _DEEPSPEED_ORACLE is not None:
        return _call(_DEEPSPEED_ORACLE, **kwargs)
    return _call(_BoundReference(kwargs["q"]), **kwargs)


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

    The framework does not report this and it has to be here: saving memory is one
    of the two things weight-only quantization is for, so a benchmark that reports
    only latency reports half the result.
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
    # No baseline check here: when this platform cannot load the upstream kernel the
    # module-level ``skip_native`` marker already tells the framework to leave the
    # native baseline out and report it as N/A. Failing instead would make the perf
    # stage red on a platform whose only problem is a prebuilt kernel's arch list.
    # The tier first, so a run whose numbers moved says whether the baseline it moved
    # against is the one this platform was supposed to pick.
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

    Separate from the ratio above on purpose. Both sides now read the same preprocessed
    B layout -- that is the whole point of the port -- so this compares kernels over
    identical inputs rather than mixing in a layout difference.
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
