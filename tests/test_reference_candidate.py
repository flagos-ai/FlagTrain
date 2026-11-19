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
"""Regression tests for reference-as-candidate selection and timing."""

from types import SimpleNamespace

import pytest
import torch

import flag_train
from flag_train.reference_candidate import (
    ReferenceContractError,
    get_candidate,
    reference_test_mode,
)


def test_candidate_defaults_to_live_operator(monkeypatch):
    candidate = lambda: "candidate"
    monkeypatch.setattr(flag_train, "megatron", SimpleNamespace(probe=candidate))
    assert get_candidate("megatron.probe") is candidate


def test_reference_mode_selects_explicit_reference():
    with reference_test_mode(True) as state:
        candidate = get_candidate("megatron.missing", reference=lambda: "reference")
        assert candidate() == "reference"
        assert state.candidate_calls == state.reference_calls == 1


@pytest.mark.parametrize("reference", [None, 42])
def test_reference_mode_rejects_invalid_reference(reference):
    with reference_test_mode(True):
        with pytest.raises(ReferenceContractError, match="No predefined reference"):
            get_candidate("megatron.missing", reference=reference)


@pytest.mark.parametrize("run_derivative", [True, False])
def test_reference_backward_requires_candidate_derivative(monkeypatch, run_derivative):
    from benchmark import base
    from benchmark.consts import BenchMode

    bench = base.Benchmark("probe", torch.square, is_backward=True)
    bench.to_bench_metrics = ["latency_base", "latency"]
    monkeypatch.setattr(
        base,
        "Config",
        SimpleNamespace(mode=BenchMode.CUDAGRAPH, skip_native=False, repetition=1),
    )
    timer_calls = iter([True, run_derivative])

    def timer(function, **kwargs):
        if next(timer_calls):
            function()
        return 1.0

    monkeypatch.setattr(base.triton.testing, "do_bench_cudagraph", timer)
    value = torch.tensor(2.0, requires_grad=True)

    with reference_test_mode(True) as state:
        if run_derivative:
            metric = bench._measure_input((value,))
            assert metric.candidate_source == "reference"
            assert state.candidate_derivative_calls == state.completed_timings == 1
        else:
            # The baseline derivative and candidate graph both ran.
            with pytest.raises(
                pytest.fail.Exception, match="candidate-side derivative"
            ):
                bench._measure_input((value,))
            assert state.candidate_calls == 1
            assert state.candidate_derivative_calls == state.completed_timings == 0
