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

The reference is DeepSpeed's ``fused_lamb`` -- the operator this implementation
ports -- rather than a re-derivation of its formula in PyTorch. That keeps the
comparison anchored to the actual reference implementation and its version
(``_DEEPSPEED_VERSION``), as tests/deepspeed/README.md requires, and it catches
the documented behaviours a hand-written formula has to be told about one by one
(``mode``, ``bias_correction``, the zero-norm trust-ratio fallback).

The reference only exists on CUDA hosts with the ``deepspeed`` package
installed, so the module skips as a whole where it is unavailable.
"""

import pytest
import torch

import flag_train

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


_DEEPSPEED_UNAVAILABLE_MSG = (
    "DeepSpeed's fused_lamb reference is unavailable; install the deepspeed package "
    "(pip install deepspeed) to run the LAMB correctness tests."
)


def _load_deepspeed_lamb():
    """Return ``(op, version)`` for DeepSpeed's fused ``lamb``, or ``(None, None)``.

    The operator comes from the installed ``deepspeed`` package; ``FusedLambBuilder``
    JIT-compiles the CUDA source that ships inside it on first use, then reuses the
    build cached under ``torch_extensions``. ``None`` (rather than ``pytest.skip``)
    is returned on failure because this is called at module import time, where a
    skip aborts collection of the whole module.
    """
    try:
        import deepspeed
        from deepspeed.ops.op_builder import FusedLambBuilder

        return FusedLambBuilder().load().lamb, deepspeed.__version__
    except Exception:
        return None, None


# Resolve the reference implementation once, at module import time.
_deepspeed_lamb, _DEEPSPEED_VERSION = _load_deepspeed_lamb()

# Only the tests that read the reference need it; the p_copy contract below is
# checked against the operator's own output and runs without DeepSpeed.
requires_deepspeed_reference = pytest.mark.skipif(
    _deepspeed_lamb is None, reason=_DEEPSPEED_UNAVAILABLE_MSG
)


def _no_p_copy(dtype):
    """The empty placeholder that tells both operators to skip the weight copy."""
    return torch.empty((0,), dtype=dtype, device=flag_train.device)


def _deepspeed_step(param, exp_avg, exp_avg_sq, grad, step, mode, bias_correction):
    """One in-place step of the reference implementation.

    ``fused_lamb`` updates ``param``/``exp_avg``/``exp_avg_sq`` in place and
    returns the layer's trust ratio, so callers pass tensors they own.
    """
    return _deepspeed_lamb(
        param,
        _no_p_copy(param.dtype),
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
        _DECAY,
    )


def _train_step(param, exp_avg, exp_avg_sq, grad, step, mode, bias_correction):
    """The same step through the operator under test."""
    return flag_train.lamb(
        param,
        _no_p_copy(param.dtype),
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
        _DECAY,
    )


@pytest.mark.lamb
@requires_deepspeed_reference
@pytest.mark.parametrize("shape", [(1024,), (4096,), (16384,)])
@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("bias_correction", [0, 1])
def test_lamb(shape, mode, bias_correction):
    """A single LAMB step must match DeepSpeed's fused_lamb."""
    dtype = torch.float32

    p = torch.randn(shape, dtype=dtype, device=flag_train.device)
    g = torch.randn(shape, dtype=dtype, device=flag_train.device)
    m = torch.zeros(shape, dtype=dtype, device=flag_train.device)
    v = torch.zeros(shape, dtype=dtype, device=flag_train.device)

    # The reference steps in place, so it runs on its own copies.
    ref_p, ref_m, ref_v = p.clone(), m.clone(), v.clone()
    ref_coeff = _deepspeed_step(ref_p, ref_m, ref_v, g, 1, mode, bias_correction)

    train_coeff = _train_step(p, m, v, g, 1, mode, bias_correction)

    # The trust ratio reported by the operator must match the reference.
    utils.train_assert_close(
        utils.to_reference(train_coeff),
        utils.to_reference(ref_coeff),
        dtype,
    )
    # The updated parameters and moments must match the reference.
    utils.train_assert_close(utils.to_reference(p), utils.to_reference(ref_p), dtype)
    utils.train_assert_close(utils.to_reference(m), utils.to_reference(ref_m), dtype)
    utils.train_assert_close(utils.to_reference(v), utils.to_reference(ref_v), dtype)


@pytest.mark.lamb
@pytest.mark.parametrize("shape", [(1024,), (4096,)])
def test_lamb_p_copy(shape):
    """The optional output copy must mirror the updated parameter.

    This is the one case here with no DeepSpeed oracle: fused_lamb_cuda_kernel.cu
    passes NULL for p_copy on fp32 operands ("don't output p_copy for fp32, it's
    wasted write"), so the reference leaves the copy untouched at every precision
    these tests run at -- comparing against it would compare against garbage. The
    contract left to check is that the copy tracks the parameter it copies.
    """
    dtype = torch.float32

    p = torch.randn(shape, dtype=dtype, device=flag_train.device)
    g = torch.randn(shape, dtype=dtype, device=flag_train.device)
    m = torch.zeros(shape, dtype=dtype, device=flag_train.device)
    v = torch.zeros(shape, dtype=dtype, device=flag_train.device)
    p_copy = torch.empty(shape, dtype=dtype, device=flag_train.device)

    flag_train.lamb(
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
        1,
        1,
        1,
        _DECAY,
    )

    utils.train_assert_close(utils.to_reference(p_copy), utils.to_reference(p), dtype)


@pytest.mark.lamb
@requires_deepspeed_reference
@pytest.mark.parametrize("mode", [0, 1], ids=["eps_inside_sqrt", "eps_outside_sqrt"])
@pytest.mark.parametrize(
    "bias_correction", [True, False], ids=["bias_corr", "no_bias_corr"]
)
def test_lamb_matches_reference(mode, bias_correction):
    """Run several LAMB steps and compare against DeepSpeed's fused_lamb.

    Mirrors DeepSpeed's ``test_fused_adam_matches_reference``: multiple parameter
    tensors, fresh gradients each step, accumulating first/second moments, with
    the operator under test compared to the reference at every step.
    """
    dtype = torch.float32
    torch.manual_seed(0)

    train_params = [
        torch.randn(1024, dtype=dtype, device=flag_train.device) for _ in range(3)
    ]
    ref_params = [p.clone() for p in train_params]
    train_m = [torch.zeros_like(p) for p in train_params]
    ref_m = [torch.zeros_like(p) for p in train_params]
    train_v = [torch.zeros_like(p) for p in train_params]
    ref_v = [torch.zeros_like(p) for p in train_params]

    for step in range(1, 6):
        for i in range(len(train_params)):
            grad = torch.randn_like(train_params[i])

            _deepspeed_step(
                ref_params[i],
                ref_m[i],
                ref_v[i],
                grad,
                step,
                mode,
                bias_correction,
            )
            _train_step(
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
