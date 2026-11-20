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
"""Correctness tests for the LAMB operator.

Two oracles, because they fail differently:

* ``lamb_ref`` -- a plain-torch composition of the same contract. It is the
  primary check and runs on any device, so the operator is testable off NVIDIA.
  Under ``--ref cpu`` it is fed CPU operands and genuinely executes there, which
  makes it an independent oracle rather than a second reading of the same device
  arithmetic; see ``_reference_copy``.
* DeepSpeed's ``fused_lamb`` -- the operator this implementation ports. It is an
  independent implementation and its version is recorded in
  ``_DEEPSPEED_VERSION``, as tests/deepspeed/README.md asks for, but it needs the
  ``deepspeed`` package, and on a backend in ``_DEEPSPEED_BASELINE_VENDORS`` below
  it is *expected* -- if it will not load there the module warns and falls back
  to the torch reference so the suite still runs, rather than failing outright.
  On other backends it is an additional check that is skipped.

Checking both is not redundant: ``lamb_ref`` is the same arithmetic written from
the same reading of the kernel, so a shared misreading of the contract would
survive it. DeepSpeed's operator is the only oracle that can disagree with that
reading.
"""

import math
import warnings

import pytest
import torch

import flag_train
from flag_train.deepspeed import lamb

from .. import accuracy_utils as utils

# Hyper-parameters shared by both implementations, mirroring the defaults
# DeepSpeed's fused_lamb is exercised with.
_LR = 1e-3
_BETA1 = 0.9
_BETA2 = 0.999
_MAX_COEFF = 10.0
_MIN_COEFF = 0.01
_EPS = 1e-8
_GRAD_SCALE = 1.0
_DECAY = 0.01

# ---------------------------------------------------------------------------
# Reference implementation
#
# Exists so the operator can be checked against a plain-torch composition of the
# same contract on any device. The DeepSpeed oracle (``fused_lamb``) needs the
# ``deepspeed`` package and a CUDA device, so without a torch reference the
# operator would be untestable off NVIDIA.
# ---------------------------------------------------------------------------


def _reference_copy(tensor):
    """An independent copy of ``tensor``, on the device the reference must use.

    ``--ref cpu`` pushes the reference *computation* onto the CPU, not merely its
    result, so every operand ``lamb_ref`` is handed comes from here and no kernel
    of the reference runs on the accelerator. That is what makes it an
    independent oracle -- a reference evaluated by the same device arithmetic it
    is checking can only confirm the arithmetic agrees with itself.
    ``lamb_ref`` checks this rather than trusting it; the comparison helpers move
    the operator's own output across afterwards.

    ``lamb_ref`` steps its parameters in place, so the copy is not optional: on a
    non-CPU reference ``to_reference`` returns its argument unchanged, and the
    reference would then corrupt the operator's operands.
    """
    return utils.to_reference(tensor).clone()


def lamb_ref(
    p,
    p_copy,
    m,
    v,
    g,
    lr,
    beta1,
    beta2,
    max_coeff,
    min_coeff,
    eps,
    grad_scale,
    step,
    mode,
    bias_correction,
    decay,
):
    """Reference for the LAMB operator, composed from plain torch ops.

    Steps ``p``/``m``/``v`` in place and returns the trust ratio, matching the
    operator's contract including the order of operations, since the two are
    compared bit-for-bit-ish rather than through a fuzzy formula.
    """
    if utils.TO_CPU:
        # ``--ref cpu`` promises the reference *computation* is on the CPU, not
        # only its result. Asserting the precondition here rather than at the
        # call sites means a test that forgets ``_reference_copy`` fails loudly
        # instead of quietly becoming a second reading of the device arithmetic.
        for name, operand in (("p", p), ("g", g), ("m", m), ("v", v)):
            assert operand.device.type == "cpu", (
                f"--ref cpu must run the reference on the CPU; {name} is on "
                f"{operand.device}"
            )

    p_old = p.clone()
    # The operator promotes the gradient to the master dtype before any
    # arithmetic.  Reproduce that here: leaving an fp16 gradient in fp16 makes
    # ``0.001 * g * g`` underflow to zero for small gradients, which changes the
    # second moment and the trust ratio.
    scaled_grad = g.to(p.dtype) / grad_scale
    m_new = beta1 * m + (1 - beta1) * scaled_grad
    v_new = beta2 * v + (1 - beta2) * scaled_grad * scaled_grad

    if mode == 0:
        denom = torch.sqrt(v_new + eps)
    else:
        denom = torch.sqrt(v_new) + eps
    update = m_new / denom + decay * p_old

    # The trust ratio is taken from the *incoming* weights and the update built
    # from them, which is why ``p_old`` is kept rather than reusing ``p``.
    w_norm = torch.sqrt(torch.sum(p_old * p_old))
    u_norm = torch.sqrt(torch.sum(update * update))
    # Either norm being zero leaves the ratio at 1.0 rather than clamped, matching
    # lamb_cuda_kernel_part3.
    if w_norm.item() == 0.0 or u_norm.item() == 0.0:
        coeff = torch.ones((), dtype=p.dtype, device=p.device)
    else:
        coeff = torch.clamp(w_norm / u_norm, min_coeff, max_coeff)

    if bias_correction == 1:
        step_size = lr * math.sqrt(1 - beta2**step) / (1 - beta1**step)
    else:
        step_size = lr

    p_new = p_old - step_size * coeff * update
    p.copy_(p_new)
    m.copy_(m_new)
    v.copy_(v_new)
    # DeepSpeed only writes p_copy on the fp16-gradient path; its fp32/fp64
    # branches pass NULL.  Mirror that contract in the reference.
    if p_copy.numel() > 0 and g.dtype == torch.float16:
        p_copy.copy_(p_new)

    return coeff.reshape(1).to(p.dtype)


_DEEPSPEED_UNAVAILABLE_MSG = (
    "DeepSpeed's fused_lamb reference is unavailable; install the deepspeed "
    "package on a CUDA host to run this check."
)


# Backends whose reference is DeepSpeed. fused_lamb ships as a CUDA op builder, so
# only a backend that can compile and execute one can host it. On these the
# reference is expected, not mandatory: if it fails to load we warn and fall back
# to the torch reference so the rest of the suite still runs, and the warning is
# what keeps the lost check from thinning quietly.
_DEEPSPEED_BASELINE_VENDORS = {"nvidia", "hygon"}


def _load_deepspeed_lamb():
    """``(op, version)`` for DeepSpeed's fused_lamb, or ``(None, None)``.

    ``FusedLambBuilder`` JIT-compiles the CUDA source shipped inside the
    ``deepspeed`` package, then reuses the build cached under ``torch_extensions``.

    On a backend in ``_DEEPSPEED_BASELINE_VENDORS`` a load failure is warned about
    rather than raised: the DeepSpeed check is dropped but the torch reference
    still runs, so a missing fused_lamb no longer aborts the whole module.
    """
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None, None

    try:
        import deepspeed
        from deepspeed.ops.op_builder import FusedLambBuilder

        return FusedLambBuilder().load().lamb, deepspeed.__version__
    except Exception as exc:
        warnings.warn(
            f"{flag_train.vendor_name!r} is expected to use DeepSpeed's fused_lamb "
            f"as its reference, but it could not be loaded: {exc!r}. Falling back "
            f"to the torch reference only; build deepspeed to enable the "
            f"fused_lamb check.",
            stacklevel=2,
        )
        return None, None


# Resolved once, at module import time.
_deepspeed_lamb, _DEEPSPEED_VERSION = _load_deepspeed_lamb()

# The torch reference is always available, so only the DeepSpeed checks skip.
requires_deepspeed_reference = pytest.mark.skipif(
    _deepspeed_lamb is None, reason=_DEEPSPEED_UNAVAILABLE_MSG
)

requires_fp64 = pytest.mark.skipif(
    not utils.fp64_is_supported, reason="device does not support fp64"
)


def _no_p_copy(dtype, device):
    """The empty placeholder that tells every implementation to skip the copy.

    It takes the parameter's device rather than the backend's, because under
    ``--ref cpu`` the operator runs on the backend while the reference runs on
    the CPU, and each is handed the placeholder built for its own operands.
    """
    return torch.empty((0,), dtype=dtype, device=device)


def _step(
    op,
    param,
    exp_avg,
    exp_avg_sq,
    grad,
    step,
    mode,
    bias_correction,
    p_copy=None,
    decay=_DECAY,
):
    """Run one in-place step through ``op`` with the shared hyper-parameters.

    Every implementation updates its tensors in place and returns the layer's
    trust ratio, so callers pass tensors they own.

    ``decay`` is overridable because it is the only hyper-parameter that changes
    *which branch* of the contract runs rather than only its values: with weight
    decay the update is non-zero even for a zero gradient, so a test that needs
    a vanishing update (the zero-norm trust ratio) must switch it off.
    """
    if p_copy is None:
        p_copy = _no_p_copy(param.dtype, param.device)
    return op(
        param,
        p_copy,
        exp_avg,
        exp_avg_sq,
        grad,
        _LR,
        _BETA1,
        _BETA2,
        _MAX_COEFF,
        _MIN_COEFF,
        _EPS,
        _GRAD_SCALE,
        step,
        mode,
        int(bias_correction),
        decay,
    )


def _assert_step_matches(
    train_coeff,
    train_p,
    train_m,
    train_v,
    other_coeff,
    other_p,
    other_m,
    other_v,
    dtype,
    atol=1e-4,
):
    """The trust ratio and the three updated tensors must all agree.

    ``atol`` is overridable because an fp64 step still receives float32
    hyper-parameters (DeepSpeed's signature takes ``float``), so its second
    moment/update differ from a double-precision reference at roughly fp32
    epsilon rather than fp64 epsilon.
    """
    utils.train_assert_close(
        utils.to_reference(train_coeff),
        utils.to_reference(other_coeff),
        dtype,
        atol=atol,
    )
    utils.train_assert_close(
        utils.to_reference(train_p), utils.to_reference(other_p), dtype, atol=atol
    )
    utils.train_assert_close(
        utils.to_reference(train_m), utils.to_reference(other_m), dtype, atol=atol
    )
    utils.train_assert_close(
        utils.to_reference(train_v), utils.to_reference(other_v), dtype, atol=atol
    )


def _check_one_step(
    master_dtype,
    grad_dtype,
    shape,
    mode,
    bias_correction,
    with_copy=False,
    copy_dtype=torch.float16,
    expect_copy=None,
    atol=1e-4,
):
    """Compare one operator step with the torch reference (and DeepSpeed).

    ``master_dtype`` is the dtype of p/m/v; ``grad_dtype`` may be float16 when
    the master dtype is float32 (the mixed-precision contract).  When
    ``with_copy`` is set, a p_copy of ``copy_dtype`` is primed with a sentinel
    so a store that never runs is visible.  ``expect_copy`` defaults to whether
    the upstream contract writes p_copy at all: only the fp16-gradient path
    does, so fp32/fp64 steps must leave the buffer untouched.
    """
    if expect_copy is None:
        expect_copy = master_dtype == torch.float32 and grad_dtype == torch.float16
    p = torch.randn(shape, dtype=master_dtype, device=flag_train.device)
    g = torch.randn(shape, dtype=grad_dtype, device=flag_train.device)
    m = torch.zeros(shape, dtype=master_dtype, device=flag_train.device)
    v = torch.zeros(shape, dtype=master_dtype, device=flag_train.device)

    train = [p.clone(), m.clone(), v.clone()]
    train_copy = None
    if with_copy:
        train_copy = torch.full(
            shape, 123.0, dtype=copy_dtype, device=flag_train.device
        )
    train_coeff = _step(lamb, *train, g, 1, mode, bias_correction, p_copy=train_copy)

    ref = [_reference_copy(p), _reference_copy(m), _reference_copy(v)]
    ref_g = _reference_copy(g)
    ref_copy = None
    if with_copy:
        ref_copy = torch.full(shape, 123.0, dtype=copy_dtype, device=ref_g.device)
    ref_coeff = _step(lamb_ref, *ref, ref_g, 1, mode, bias_correction, p_copy=ref_copy)
    _assert_step_matches(train_coeff, *train, ref_coeff, *ref, master_dtype, atol=atol)

    if with_copy:
        if expect_copy:
            utils.train_assert_equal(
                utils.to_reference(train_copy),
                utils.to_reference(train[0].to(copy_dtype)),
            )
            utils.train_assert_equal(
                utils.to_reference(ref_copy),
                utils.to_reference(ref[0].to(copy_dtype)),
            )
        else:
            # The full-precision paths leave p_copy untouched, as DeepSpeed's
            # NULL p_copy argument does.
            utils.train_assert_equal(
                utils.to_reference(train_copy),
                utils.to_reference(torch.full_like(train_copy, 123.0)),
            )
            utils.train_assert_equal(
                utils.to_reference(ref_copy),
                utils.to_reference(torch.full_like(ref_copy, 123.0)),
            )

    if _deepspeed_lamb is not None:
        deepspeed = [p.clone(), m.clone(), v.clone()]
        deepspeed_copy = None
        if with_copy:
            deepspeed_copy = torch.full(
                shape, 123.0, dtype=copy_dtype, device=flag_train.device
            )
        deepspeed_coeff = _step(
            _deepspeed_lamb,
            *deepspeed,
            g,
            1,
            mode,
            bias_correction,
            p_copy=deepspeed_copy,
        )
        _assert_step_matches(
            train_coeff,
            *train,
            deepspeed_coeff,
            *deepspeed,
            master_dtype,
            atol=atol,
        )
        # Upstream only writes p_copy on its fp16-gradient path; the fp32 and
        # fp64 branches pass NULL, so the sentinel must survive there too.
        if with_copy:
            if expect_copy:
                utils.train_assert_equal(
                    utils.to_reference(deepspeed_copy),
                    utils.to_reference(deepspeed[0].to(copy_dtype)),
                )
            else:
                utils.train_assert_equal(
                    utils.to_reference(deepspeed_copy),
                    utils.to_reference(torch.full_like(deepspeed_copy, 123.0)),
                )


def _check_multistep(master_dtype, grad_dtype, mode, bias_correction, atol=1e-4):
    """Accumulate moments over several steps and compare params and state."""
    torch.manual_seed(0)
    train_params = [
        torch.randn(1024, dtype=master_dtype, device=flag_train.device)
        for _ in range(2)
    ]
    ref_params = [_reference_copy(param) for param in train_params]
    train_m = [torch.zeros_like(param) for param in train_params]
    ref_m = [torch.zeros_like(param) for param in ref_params]
    train_v = [torch.zeros_like(param) for param in train_params]
    ref_v = [torch.zeros_like(param) for param in ref_params]

    for step in range(1, 5):
        for i, param in enumerate(train_params):
            grad = torch.randn(param.shape, dtype=grad_dtype, device=flag_train.device)
            _step(
                lamb_ref,
                ref_params[i],
                ref_m[i],
                ref_v[i],
                _reference_copy(grad),
                step,
                mode,
                bias_correction,
            )
            _step(
                lamb,
                param,
                train_m[i],
                train_v[i],
                grad,
                step,
                mode,
                bias_correction,
            )

    for train_param, ref_param in zip(train_params, ref_params):
        utils.train_assert_close(
            utils.to_reference(train_param),
            utils.to_reference(ref_param),
            master_dtype,
            atol=atol,
        )
    for train_exp_avg, ref_exp_avg in zip(train_m, ref_m):
        utils.train_assert_close(
            utils.to_reference(train_exp_avg),
            utils.to_reference(ref_exp_avg),
            master_dtype,
            atol=atol,
        )
    for train_exp_avg_sq, ref_exp_avg_sq in zip(train_v, ref_v):
        utils.train_assert_close(
            utils.to_reference(train_exp_avg_sq),
            utils.to_reference(ref_exp_avg_sq),
            master_dtype,
            atol=atol,
        )


@pytest.mark.lamb
@pytest.mark.parametrize("shape", [(1024,), (4096,), (16384,)])
@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("bias_correction", [0, 1])
def test_lamb(shape, mode, bias_correction):
    """A single LAMB step must match the torch reference, and DeepSpeed's
    fused_lamb when it is available."""
    dtype = torch.float32

    p = torch.randn(shape, dtype=dtype, device=flag_train.device)
    g = torch.randn(shape, dtype=dtype, device=flag_train.device)
    m = torch.zeros(shape, dtype=dtype, device=flag_train.device)
    v = torch.zeros(shape, dtype=dtype, device=flag_train.device)

    # Every implementation steps in place, so each runs on its own copies.
    train = [p.clone(), m.clone(), v.clone()]
    train_coeff = _step(lamb, *train, g, 1, mode, bias_correction)

    torch_ref = [_reference_copy(p), _reference_copy(m), _reference_copy(v)]
    torch_coeff = _step(
        lamb_ref, *torch_ref, _reference_copy(g), 1, mode, bias_correction
    )
    _assert_step_matches(train_coeff, *train, torch_coeff, *torch_ref, dtype)

    if _deepspeed_lamb is not None:
        deepspeed = [p.clone(), m.clone(), v.clone()]
        deepspeed_coeff = _step(
            _deepspeed_lamb, *deepspeed, g, 1, mode, bias_correction
        )
        _assert_step_matches(train_coeff, *train, deepspeed_coeff, *deepspeed, dtype)


@pytest.mark.lamb
@requires_fp64
@pytest.mark.parametrize("shape", [(1024,), (4096,), (16384,), (65536,), (1 << 20,)])
@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("bias_correction", [0, 1])
def test_lamb_fp64(shape, mode, bias_correction):
    """Full float64 master weights, moments and gradients.

    The tolerance is looser than fp64 epsilon because the operator receives
    float32 hyper-parameters (the upstream signature takes ``float``), which
    rounds ``beta``/``eps``/``decay`` before the double-precision tensor math.
    """
    _check_one_step(
        torch.float64,
        torch.float64,
        shape,
        mode,
        bias_correction,
        atol=1e-5,
    )


@pytest.mark.lamb
@requires_fp64
@pytest.mark.parametrize("mode", [0, 1])
def test_lamb_fp64_p_copy_ignored(mode):
    """The fp64 path must not write p_copy, matching DeepSpeed's NULL argument."""
    _check_one_step(
        torch.float64,
        torch.float64,
        (4096,),
        mode,
        1,
        with_copy=True,
        copy_dtype=torch.float64,
        expect_copy=False,
        atol=1e-5,
    )


@pytest.mark.lamb
@requires_fp64
@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("bias_correction", [0, 1])
def test_lamb_fp64_multistep(mode, bias_correction):
    """Accumulated fp64 moments must stay consistent over several steps."""
    _check_multistep(torch.float64, torch.float64, mode, bias_correction, atol=1e-5)


@pytest.mark.lamb
@pytest.mark.parametrize("shape", [(1024,), (16384,)])
@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("bias_correction", [0, 1])
@pytest.mark.parametrize("with_copy", [False, True], ids=["no_copy", "copy"])
def test_lamb_mixed_fp16_grads(shape, mode, bias_correction, with_copy):
    """fp16 gradients update fp32 master weights and fp32 moments.

    The optional p_copy is fp16: it is the reduced-precision output buffer the
    mixed-precision contract exists for.
    """
    _check_one_step(
        torch.float32,
        torch.float16,
        shape,
        mode,
        bias_correction,
        with_copy=with_copy,
    )


@pytest.mark.lamb
@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("bias_correction", [0, 1])
def test_lamb_mixed_fp16_grads_multistep(mode, bias_correction):
    """fp16 gradients must keep accumulating into fp32 state correctly."""
    _check_multistep(torch.float32, torch.float16, mode, bias_correction)


@pytest.mark.lamb
def test_lamb_rejects_unsupported_dtypes():
    """The public contract only accepts the three supported combinations."""
    shape = (1024,)
    device = flag_train.device

    def _zeros(dtype):
        return torch.zeros(shape, dtype=dtype, device=device)

    # fp16 master weights are rejected by the fp16-gradient branch's
    # "parameter must be float" assertion.
    with pytest.raises(RuntimeError):
        _step(
            lamb,
            _zeros(torch.float16),
            _zeros(torch.float16),
            _zeros(torch.float16),
            _zeros(torch.float16),
            1,
            1,
            1,
        )

    if utils.bf16_is_supported:
        # bf16 is outside DeepSpeed's AT_DISPATCH_FLOATING_TYPES_AND_HALF
        # dispatch, so the failure is NotImplementedError before dtype pairing.
        with pytest.raises(NotImplementedError):
            _step(
                lamb,
                _zeros(torch.bfloat16),
                _zeros(torch.bfloat16),
                _zeros(torch.bfloat16),
                _zeros(torch.bfloat16),
                1,
                1,
                1,
            )
        # A float32 master may take fp16 gradients, but not bf16 ones.
        with pytest.raises(NotImplementedError):
            _step(
                lamb,
                _zeros(torch.float32),
                _zeros(torch.float32),
                _zeros(torch.float32),
                _zeros(torch.bfloat16),
                1,
                1,
                1,
            )

    # Moments must match the master dtype.
    if utils.fp64_is_supported:
        with pytest.raises(RuntimeError):
            _step(
                lamb,
                _zeros(torch.float32),
                _zeros(torch.float64),
                _zeros(torch.float32),
                _zeros(torch.float32),
                1,
                1,
                1,
            )
        # fp64 master requires an fp64 gradient, not fp32.
        with pytest.raises(RuntimeError):
            _step(
                lamb,
                _zeros(torch.float64),
                _zeros(torch.float64),
                _zeros(torch.float64),
                _zeros(torch.float32),
                1,
                1,
                1,
            )


@pytest.mark.lamb
@pytest.mark.parametrize("mode", [0, 1])
def test_lamb_grid_stride(mode):
    """A tensor above MAX_BLOCKS * BLOCK_SIZE exercises the grid-stride loops."""
    _check_one_step(torch.float32, torch.float32, (1 << 20,), mode, 1)


@pytest.mark.lamb
def test_lamb_rejects_non_contiguous():
    """CHECK_INPUT parity: every non-empty operand must be contiguous."""
    shape = (1024,)
    device = flag_train.device

    p = torch.randn(2 * shape[0], device=device)[::2]
    m = torch.zeros(shape, device=device)
    v = torch.zeros(shape, device=device)
    g = torch.randn(shape, device=device)
    with pytest.raises(RuntimeError):
        _step(lamb, p, m, v, g, 1, 1, 1)

    # A non-empty p_copy is checked the same way on the fp16-gradient path.
    p = torch.randn(shape, device=device)
    g = torch.randn(shape, dtype=torch.float16, device=device)
    p_copy = torch.full((2 * shape[0],), 123.0, device=device)[::2]
    with pytest.raises(RuntimeError):
        _step(lamb, p, m, v, g, 1, 1, 1, p_copy=p_copy)


@pytest.mark.lamb
def test_lamb_rejects_empty():
    """DeepSpeed fails its zero-block launch; reject an empty tensor explicitly."""
    shape = (0,)
    device = flag_train.device
    with pytest.raises(RuntimeError):
        _step(
            lamb,
            torch.empty(shape, device=device),
            torch.empty(shape, device=device),
            torch.empty(shape, device=device),
            torch.empty(shape, device=device),
            1,
            1,
            1,
        )


@pytest.mark.lamb
def test_lamb_clamp_order_matches_upstream():
    """DeepSpeed applies max first and min second; min_coeff wins when inverted."""
    shape = (1024,)
    device = flag_train.device
    p = torch.ones(shape, device=device)
    g = torch.ones(shape, device=device)
    m = torch.zeros(shape, device=device)
    v = torch.zeros(shape, device=device)
    coeff = lamb(
        p,
        torch.empty((0,), device=device),
        m,
        v,
        g,
        1e-3,
        0.9,
        0.999,
        0.5,
        2.0,
        1e-8,
        1.0,
        1,
        1,
        1,
        0.01,
    )
    assert coeff.item() == pytest.approx(2.0)


@pytest.mark.lamb
@pytest.mark.parametrize("shape", [(1024,), (4096,)])
@pytest.mark.parametrize(
    "copy_dtype", [torch.float32, torch.int32], ids=["float32", "int32"]
)
def test_lamb_p_copy_ignored_on_full_precision(shape, copy_dtype):
    """The fp32 path must not write p_copy, whatever its dtype.

    DeepSpeed's fused_lamb passes NULL for p_copy on fp32 operands ("don't
    output p_copy for fp32, it's wasted write"), even when the caller supplies a
    non-empty buffer, and it does not constrain that buffer's dtype.  The
    sentinel makes a stray store visible.
    """
    _check_one_step(
        torch.float32,
        torch.float32,
        shape,
        1,
        1,
        with_copy=True,
        copy_dtype=copy_dtype,
        expect_copy=False,
    )


@pytest.mark.lamb
@pytest.mark.parametrize("shape", [(16384,)])
def test_lamb_p_copy_reduced_precision(shape):
    """A float16 copy is written on the fp16-gradient split path.

    The copy store is exercised above the single-program size, where the
    operator runs a second kernel.  Only the fp16-gradient path writes p_copy;
    the fp32 path is covered by ``test_lamb_p_copy_ignored_on_full_precision``.

    The buffers are primed with a sentinel rather than left empty: a store that
    never runs would otherwise pass by reading whatever an earlier, correct
    result left in a recycled allocation.
    """
    dtype = torch.float32
    copy_dtype = torch.float16
    sentinel = 123.0

    p = torch.randn(shape, dtype=dtype, device=flag_train.device)
    g = torch.randn(shape, dtype=torch.float16, device=flag_train.device)
    m = torch.zeros(shape, dtype=dtype, device=flag_train.device)
    v = torch.zeros(shape, dtype=dtype, device=flag_train.device)

    torch_p, torch_m, torch_v = (
        _reference_copy(p),
        _reference_copy(m),
        _reference_copy(v),
    )
    torch_copy = torch.full(shape, sentinel, dtype=copy_dtype, device=torch_p.device)
    _step(
        lamb_ref,
        torch_p,
        torch_m,
        torch_v,
        _reference_copy(g),
        1,
        1,
        1,
        p_copy=torch_copy,
    )

    p_copy = torch.full(shape, sentinel, dtype=copy_dtype, device=flag_train.device)
    _step(lamb, p, m, v, g, 1, 1, 1, p_copy=p_copy)

    # Exactly this call's parameter rounded to float16, bit for bit: a
    # tolerance is the wrong tool here, because a value on a rounding boundary
    # can flip between the operator's and the reference's arithmetic over a 1e-7
    # difference and fail for no reason.
    #
    # Comparing exactly is also what makes the sentinel do its job on both sides:
    # a copy store that never ran still holds the sentinel and fails here, rather
    # than passing on whatever a recycled allocation happened to contain.
    utils.train_assert_equal(
        utils.to_reference(p_copy), utils.to_reference(p.to(copy_dtype))
    )
    # The oracle's own copy too, so the check above is anchored to a copy this
    # test knows was written.
    utils.train_assert_equal(
        utils.to_reference(torch_copy), utils.to_reference(torch_p.to(copy_dtype))
    )


@pytest.mark.lamb
@pytest.mark.parametrize(
    "copy_dtype",
    [torch.float32, torch.int32]
    + ([torch.bfloat16] if utils.bf16_is_supported else []),
    ids=["float32", "int32", "bfloat16"],
)
def test_lamb_mixed_rejects_non_half_p_copy(copy_dtype):
    """DeepSpeed reads p_copy as Half on the fp16-gradient path."""
    shape = (1024,)
    device = flag_train.device
    p = torch.randn(shape, dtype=torch.float32, device=device)
    g = torch.randn(shape, dtype=torch.float16, device=device)
    m = torch.zeros(shape, dtype=torch.float32, device=device)
    v = torch.zeros(shape, dtype=torch.float32, device=device)
    p_copy = torch.full(shape, 123.0, dtype=copy_dtype, device=device)

    with pytest.raises(RuntimeError, match="expected scalar type Half"):
        _step(lamb, p, m, v, g, 1, 1, 1, p_copy=p_copy)


@pytest.mark.lamb
@pytest.mark.parametrize("mode", [0, 1], ids=["eps_inside_sqrt", "eps_outside_sqrt"])
@pytest.mark.parametrize(
    "bias_correction", [True, False], ids=["bias_corr", "no_bias_corr"]
)
def test_lamb_matches_reference(mode, bias_correction):
    """Run several LAMB steps and compare against the torch reference.

    Mirrors DeepSpeed's ``test_fused_adam_matches_reference``: multiple parameter
    tensors, fresh gradients each step, accumulating first/second moments, with
    the operator under test compared to the reference at every step.
    """
    dtype = torch.float32
    torch.manual_seed(0)

    train_params = [
        torch.randn(1024, dtype=dtype, device=flag_train.device) for _ in range(3)
    ]
    # Same values as the operator's operands -- ``to_reference`` copies rather
    # than converts -- but on whichever device the reference is meant to run.
    ref_params = [_reference_copy(p) for p in train_params]
    train_m = [torch.zeros_like(p) for p in train_params]
    ref_m = [torch.zeros_like(p) for p in ref_params]
    train_v = [torch.zeros_like(p) for p in train_params]
    ref_v = [torch.zeros_like(p) for p in ref_params]

    for step in range(1, 6):
        for i in range(len(train_params)):
            grad = torch.randn_like(train_params[i])

            _step(
                lamb_ref,
                ref_params[i],
                ref_m[i],
                ref_v[i],
                _reference_copy(grad),
                step,
                mode,
                bias_correction,
            )
            _step(
                lamb,
                train_params[i],
                train_m[i],
                train_v[i],
                grad,
                step,
                mode,
                bias_correction,
            )

    for train_param, ref_param in zip(train_params, ref_params):
        utils.train_assert_close(
            utils.to_reference(train_param), utils.to_reference(ref_param), dtype
        )
    for train_exp_avg, ref_exp_avg in zip(train_m, ref_m):
        utils.train_assert_close(
            utils.to_reference(train_exp_avg), utils.to_reference(ref_exp_avg), dtype
        )
    for train_exp_avg_sq, ref_exp_avg_sq in zip(train_v, ref_v):
        utils.train_assert_close(
            utils.to_reference(train_exp_avg_sq),
            utils.to_reference(ref_exp_avg_sq),
            dtype,
        )


@pytest.mark.lamb
@requires_deepspeed_reference
@pytest.mark.parametrize("mode", [0, 1], ids=["eps_inside_sqrt", "eps_outside_sqrt"])
@pytest.mark.parametrize("shape", [(1024,), (16384,)])
def test_matches_deepspeed_oracle(mode, shape):
    """Pin the DeepSpeed oracle explicitly, so a run that quietly stopped
    reaching it (deepspeed missing) is visible rather than silently thinner."""
    dtype = torch.float32

    p = torch.randn(shape, dtype=dtype, device=flag_train.device)
    g = torch.randn(shape, dtype=dtype, device=flag_train.device)
    m = torch.zeros(shape, dtype=dtype, device=flag_train.device)
    v = torch.zeros(shape, dtype=dtype, device=flag_train.device)

    train = [p.clone(), m.clone(), v.clone()]
    train_coeff = _step(lamb, *train, g, 1, mode, 1)

    deepspeed = [p.clone(), m.clone(), v.clone()]
    deepspeed_coeff = _step(_deepspeed_lamb, *deepspeed, g, 1, mode, 1)
    _assert_step_matches(train_coeff, *train, deepspeed_coeff, *deepspeed, dtype)


@pytest.mark.lamb
@pytest.mark.parametrize("shape", [(1024,), (16384,)])
@pytest.mark.parametrize(
    "param_zero,grad_zero",
    [(False, True), (True, False), (True, True)],
    ids=["zero_grad", "zero_param", "both_zero"],
)
def test_lamb_zero_norm_trust_ratio(shape, param_zero, grad_zero):
    """Either norm vanishing must leave the trust ratio at exactly 1.0.

    ``lamb_ref`` returns 1.0 *without clamping* when ``||w||`` or ``||u||`` is
    zero, matching ``lamb_cuda_kernel_part3``. An implementation that loses one
    half of that test instead returns a clamped ratio -- 10.0 or 0.01 at these
    hyper-parameters -- and moves the parameters up to a hundred times too far.

    Reaching the branch needs an all-zero operand, which no other test provides:
    they all draw from ``randn``, so neither norm is ever zero. ``decay`` is
    switched off because weight decay keeps the update non-zero even for a zero
    gradient, so ``||u||`` would not vanish.

    The two shapes straddle the operator's single-program cutoff, so the ratio is
    checked both where one program holds the whole tensor and where it is folded
    across programs.
    """
    dtype = torch.float32

    p = (torch.zeros if param_zero else torch.randn)(
        shape, dtype=dtype, device=flag_train.device
    )
    g = (torch.zeros if grad_zero else torch.randn)(
        shape, dtype=dtype, device=flag_train.device
    )
    m = torch.zeros(shape, dtype=dtype, device=flag_train.device)
    v = torch.zeros(shape, dtype=dtype, device=flag_train.device)

    ref = [_reference_copy(p), _reference_copy(m), _reference_copy(v)]
    ref_coeff = _step(lamb_ref, *ref, _reference_copy(g), 1, 1, 1, decay=0)

    train = [p.clone(), m.clone(), v.clone()]
    train_coeff = _step(lamb, *train, g, 1, 1, 1, decay=0)

    # Exactly 1.0, unclamped. The reference is checked too: if the oracle itself
    # started clamping, this case would no longer be exercising the rule it is
    # named for.
    one = torch.ones((1,), dtype=dtype, device=flag_train.device)
    utils.train_assert_equal(utils.to_reference(ref_coeff), utils.to_reference(one))
    utils.train_assert_equal(utils.to_reference(train_coeff), utils.to_reference(one))
    _assert_step_matches(train_coeff, *train, ref_coeff, *ref, dtype)


@pytest.mark.lamb
def test_lamb_many_partial_norms():
    """More partial norms than a single reduction tile holds.

    The cross-block fold walks the partials one tile at a time. Every other shape
    here tops out at 16384 elements -- 8 blocks on the Ascend override, 16 on the
    generic kernel -- so that loop never iterates, and a defect in it (a wrong
    stride into the second half of the partial array, a fold tile that drops the
    tail) would go unnoticed. ``2**21 + 1`` elements is 257 blocks there and 2049
    here, and the odd tail also exercises the masked last block.
    """
    dtype = torch.float32
    shape = (2**21 + 1,)

    p = torch.randn(shape, dtype=dtype, device=flag_train.device)
    g = torch.randn(shape, dtype=dtype, device=flag_train.device)
    m = torch.zeros(shape, dtype=dtype, device=flag_train.device)
    v = torch.zeros(shape, dtype=dtype, device=flag_train.device)

    ref = [_reference_copy(p), _reference_copy(m), _reference_copy(v)]
    ref_coeff = _step(lamb_ref, *ref, _reference_copy(g), 1, 1, 1)

    train = [p.clone(), m.clone(), v.clone()]
    train_coeff = _step(lamb, *train, g, 1, 1, 1)

    _assert_step_matches(train_coeff, *train, ref_coeff, *ref, dtype)
