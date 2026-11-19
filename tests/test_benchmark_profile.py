# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

ROOT = Path(__file__).resolve().parents[1]
NODE = "benchmark/test_probe.py::test_profile"
FORWARD_CASE = f"{NODE}[forward]::core::float32::1"
BACKWARD_CASE = f"{NODE}[backward]::core::float32::1"


@pytest.fixture
def suite(pytester, monkeypatch):
    benchmark = pytester.path / "benchmark"
    benchmark.mkdir()
    for filename in ("__init__.py", "base.py", "cases.py", "conftest.py", "consts.py"):
        shutil.copyfile(ROOT / "benchmark" / filename, benchmark / filename)
    tests = pytester.path / "tests"
    tests.mkdir()
    shutil.copyfile(ROOT / "tests/conftest.py", tests / "conftest.py")
    (tests / "test_unselected.py").write_text(
        "def test_unselected():\n    assert False, 'unselected correctness test ran'\n"
    )
    paths = [str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, paths)))
    pytester.makepyfile(
        events="""
from pathlib import Path

def record(event):
    with Path("events.txt").open("a") as log:
        print(event, file=log)
""",
        candidate="""
import os
from pathlib import Path

import torch
from events import record

class GradientProbe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        record("forward")
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        record("backward")
        return grad

def candidate(x):
    if x.requires_grad:
        return GradientProbe.apply(x)
    record(f"call:{int(x[0])}")
    if os.getenv("FLAGTRAIN_PROFILE_TEST_FAIL") and "enter:" in Path("events.txt").read_text():
        raise RuntimeError("capture failure")
    x.add_(1)
    return x
""",
    )
    (benchmark / "test_probe.py").write_text("""
import weakref
from types import SimpleNamespace

import pytest
import torch

from benchmark import base
from benchmark.cases import BenchmarkCasePlan
from events import record
from flag_train.reference_candidate import get_candidate

class ProbeBenchmark(base.GenericBenchmark):
    prepared_input = None

    def set_shapes(self, shape_file=None):
        self.shapes = [(1,), (2,)]

    def set_more_shapes(self):
        return []

    def get_latency(self, *args, **kwargs):
        pytest.fail("Profile must bypass the timer")

    def build_inputs(self, case):
        if self.prepared_input is not None:
            assert self.prepared_input() is None, "Warmup input is still live"
        return super().build_inputs(case)

    def unpack_to_args_kwargs(self, inputs):
        args, kwargs = super().unpack_to_args_kwargs(inputs)
        self.prepared_input = weakref.ref(args[0])
        return args, kwargs

def baseline(*args, **kwargs):
    pytest.fail("Profile must bypass the baseline")

def cases(shape, dtype):
    yield BenchmarkCasePlan(shape={"x": list(shape)}, builder_args=(shape,))

def inputs(plan, dtype, device):
    shape, = plan.builder_args
    record(f"build:{shape[0]}")
    return (torch.zeros(shape, dtype=dtype),)

@pytest.mark.parametrize("backward", [False, True], ids=["forward", "backward"])
def test_profile(backward, monkeypatch):
    monkeypatch.setattr(base, "torch_device_fn", SimpleNamespace(synchronize=lambda: record("sync")))
    bench = ProbeBenchmark(
        op_name="profile_probe", torch_op=baseline, is_backward=backward,
        dtypes=[torch.float32], case_fn=cases, build_inputs_fn=inputs,
    )
    bench.set_train(get_candidate("megatron.profile_probe"))
    bench.run()

def test_unselected():
    pytest.fail("Unselected benchmark ran")
""")

    with (benchmark / "conftest.py").open("a") as file:
        file.write("""

from contextlib import contextmanager
from events import record

@contextmanager
def pytest_flaggems_profile_scope(backend, case_id):
    assert backend == flag_train.vendor_name
    record(f"enter:{case_id}")
    try:
        yield
    finally:
        record("exit")
""")
    return pytester


def run_profile(suite, *options):
    return suite.runpytest_subprocess(
        "benchmark",
        "tests",
        "--level",
        "core",
        "--profile-only",
        "--override",
        "megatron.profile_probe:candidate.py:candidate",
        *options,
    )


@pytest.mark.parametrize("fail", [False, True])
def test_profile_warms_up_with_separate_inputs_and_closes_scope(
    suite, monkeypatch, fail
):
    if fail:
        monkeypatch.setenv("FLAGTRAIN_PROFILE_TEST_FAIL", "1")
    result = run_profile(
        suite,
        "--case-id",
        FORWARD_CASE,
        "--profile-warmup",
        "2",
        "--profile-iterations",
        "2",
        "--record",
        "json",
        "--output",
        "result.json",
        "--base-data",
        "unused-baseline.json",
    )
    result.assert_outcomes(**({"failed": 1} if fail else {"passed": 1}))
    if fail:
        result.stdout.fnmatch_lines(["*RuntimeError: capture failure*"])
    events = (suite.path / "events.txt").read_text().splitlines()
    assert events == [
        "build:2",
        "call:0",
        "call:1",
        "sync",
        "build:2",
        "sync",
        f"enter:{FORWARD_CASE}",
        "call:0",
        *([] if fail else ["call:1", "sync"]),
        "exit",
    ]
    assert not (suite.path / "result.json").exists()
    assert not (suite.path / "accuracy_result.json").exists()


@pytest.mark.parametrize("warmup", [0, 1])
def test_profile_captures_backward_after_building_graph(suite, warmup):
    run_profile(
        suite, "--case-id", BACKWARD_CASE, "--profile-warmup", str(warmup)
    ).assert_outcomes(passed=1)
    assert (suite.path / "events.txt").read_text().splitlines() == [
        *(["build:2", "forward", "backward", "sync"] if warmup else []),
        "build:2",
        "forward",
        "sync",
        f"enter:{BACKWARD_CASE}",
        "backward",
        "sync",
        "exit",
    ]


def test_profile_rejects_unknown_workload(suite):
    result = run_profile(suite, "--case-id", f"{NODE}[forward]::core::float32::99")
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(["*case selection failed: unknown=*99*"])
    assert "Traceback" not in result.stderr.str()
    assert "candidate implementation was never invoked" not in result.stderr.str()
    assert not (suite.path / "events.txt").exists()


def test_benchmark_unknown_case_retains_unused_override_check(suite):
    result = suite.runpytest_subprocess(
        f"{NODE}[forward]",
        "--level",
        "core",
        "--case-id",
        f"{NODE}[forward]::core::float32::99",
        "--override",
        "megatron.profile_probe:candidate.py:candidate",
    )
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(["*case selection failed: unknown=*99*"])
    result.stderr.fnmatch_lines(["*candidate implementation was never invoked*"])


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ([], "*requires exactly one --case-id*"),
        (
            ["--case-id", FORWARD_CASE, "--case-id", BACKWARD_CASE],
            "*requires exactly one*",
        ),
        (
            ["--case-id", FORWARD_CASE, "--list-cases"],
            "*cannot be combined*--list-cases*",
        ),
        (
            ["--case-id", FORWARD_CASE, "--profile-iterations", "0"],
            "*iterations positive*",
        ),
    ],
)
def test_profile_rejects_invalid_options(suite, options, error):
    result = run_profile(suite, *options)
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines([error])
