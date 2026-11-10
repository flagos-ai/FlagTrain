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

"""Exercise the dummy CLI through the actual correctness and benchmark runners."""

import json
import os
import shutil
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

ROOT = Path(__file__).resolve().parents[1]
OP = "megatron.dummy_probe"


@pytest.fixture
def suite(pytester, monkeypatch):
    for directory, files in (
        ("tests", ("conftest.py",)),
        (
            "benchmark",
            ("__init__.py", "base.py", "cases.py", "conftest.py", "consts.py"),
        ),
    ):
        target = pytester.path / directory
        target.mkdir()
        for file in files:
            shutil.copyfile(ROOT / directory / file, target / file)
    pythonpath = [str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, pythonpath)))
    (pytester.path / "tests/test_probe.py").write_text("""
import pytest
from flag_train.reference_candidate import get_candidate

@pytest.mark.parametrize("value", [1, 2])
def test_reached(value):
    get_candidate("megatron.dummy_probe")(value)
""")
    (pytester.path / "benchmark/test_probe_benchmark.py").write_text("""
import torch
import pytest
from benchmark.base import GenericBenchmark
from benchmark.cases import BenchmarkCasePlan
from flag_train.reference_candidate import get_candidate

class ProbeBench(GenericBenchmark):
    def set_shapes(self, shape_file=None):
        self.shapes = [(1,), (2,), (3,)]

    def set_more_shapes(self):
        return []

    def get_latency(self, *args, **kwargs):
        pytest.fail("Dummy traversal must not time either callable")

def cases(shape, dtype):
    yield BenchmarkCasePlan(shape={"x": list(shape)}, builder_args=(shape,))

def inputs(plan, dtype, device):
    shape, = plan.builder_args
    if shape == (2,):
        pytest.skip("input unavailable")
    return (torch.empty(shape, dtype=dtype),)

def test_benchmark():
    bench = ProbeBench(
        op_name="dummy_probe", torch_op=torch.neg,
        dtypes=[torch.float32], case_fn=cases, build_inputs_fn=inputs,
    )
    bench.set_train(lambda x: get_candidate("megatron.dummy_probe")(x))
    bench.run()
""")
    return pytester


def test_dummy_cli_correctness_and_benchmark(suite):
    result = suite.runpytest_subprocess(
        "tests",
        "benchmark",
        "--dummy-candidate",
        OP,
        "--dummy-report",
        "reports/dummy.json",
        "--record",
        "json",
        "--output",
        "result.json",
    )
    assert result.ret == pytest.ExitCode.OK
    assert not (suite.path / "result.json").exists()
    report = json.loads((suite.path / "reports/dummy.json").read_text())
    assert report["schema_version"] == "flaggems.dummy-candidate/v1"
    assert report["status"] == "PASSED"
    accuracy = [row for row in report["records"] if row["phase"] == "correctness"]
    benchmark = [row for row in report["records"] if row["phase"] == "benchmark"]
    assert len(accuracy) == 2
    assert all(row["status"] == "REACHED" and row["calls"] == 1 for row in accuracy)
    assert [row["status"] for row in benchmark] == ["REACHED", "SKIP", "REACHED"]
    assert [row["calls"] for row in benchmark] == [1, 0, 1]
    assert len({row["case_id"] for row in benchmark}) == 3


def test_dummy_cli_reports_bypassed_and_fixture_only_calls(suite):
    (suite.path / "tests/test_unreached.py").write_text("""
import pytest
from flag_train.reference_candidate import get_candidate

def test_bypassed():
    pass

def test_swallowed():
    with pytest.raises(Exception):
        get_candidate("megatron.dummy_probe")(1)

@pytest.fixture
def setup():
    with pytest.raises(Exception):
        get_candidate("megatron.dummy_probe")(1)

def test_fixture_only(setup):
    pass
""")
    result = suite.runpytest_subprocess(
        "tests/test_unreached.py", "--dummy-candidate", OP
    )
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    report = json.loads((suite.path / "dummy-candidate-report.json").read_text())
    assert report["status"] == "FAILED"
    assert [row["status"] for row in report["records"]] == [
        "BYPASSED",
        "POST_CALL_ERROR",
        "PRE_CALL_ERROR",
    ]
    assert report["records"][-1]["before_candidate_calls"] == 1


def test_dummy_cli_selects_exact_benchmark_case(suite):
    selected = "benchmark/test_probe_benchmark.py::test_benchmark::core::float32::2"
    result = suite.runpytest_subprocess(
        "benchmark", "--level", "core", "--dummy-candidate", OP, "--case-id", selected
    )
    assert result.ret == pytest.ExitCode.OK
    report = json.loads((suite.path / "dummy-candidate-report.json").read_text())
    (row,) = report["records"]
    assert row["case_id"] == selected
    assert row["status"] == "REACHED"


def test_dummy_report_requires_dummy_mode(suite):
    result = suite.runpytest_subprocess("tests", "--dummy-report", "unused.json")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--dummy-report requires --dummy-candidate*"])


def test_dummy_benchmark_builder_hit_is_not_candidate_evidence(suite):
    path = suite.path / "benchmark/test_probe_benchmark.py"
    path.write_text(
        path.read_text().replace(
            "    if shape == (2,):",
            """    if shape == (1,):
        with pytest.raises(Exception):
            get_candidate("megatron.dummy_probe")(1)
    if shape == (2,):""",
        )
    )
    result = suite.runpytest_subprocess("benchmark", "--dummy-candidate", OP)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    report = json.loads((suite.path / "dummy-candidate-report.json").read_text())
    assert report["status"] == "FAILED"
    assert [row["status"] for row in report["records"]] == [
        "PRE_CALL_ERROR",
        "SKIP",
        "REACHED",
    ]
    assert report["records"][0]["before_candidate_calls"] == 1


def test_dummy_conflicts_with_reference_mode(suite):
    result = suite.runpytest_subprocess(
        "tests", "--dummy-candidate", OP, "--ref-as-candidate"
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*cannot be combined with*"])


def test_dummy_cli_fails_when_requested_benchmark_case_is_skipped(suite):
    selected = "benchmark/test_probe_benchmark.py::test_benchmark::core::float32::1"
    result = suite.runpytest_subprocess(
        "tests",
        "benchmark",
        "--level",
        "core",
        "--dummy-candidate",
        OP,
        "--case-id",
        selected,
    )
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    report = json.loads((suite.path / "dummy-candidate-report.json").read_text())
    assert report["status"] == "FAILED"
    assert any(
        row["phase"] == "correctness" and row["status"] == "REACHED"
        for row in report["records"]
    )
    (row,) = [row for row in report["records"] if row.get("case_id") == selected]
    assert row["status"] == "SKIP"
    assert row["calls"] == 0


@pytest.mark.parametrize("paths", [("tests", "benchmark"), ("benchmark", "tests")])
def test_mixed_record_json_uses_default_report_paths(suite, paths):
    (suite.path / "tests/test_probe.py").write_text("""
def test_correctness():
    assert 1 + 1 == 2
""")
    path = suite.path / "benchmark/test_probe_benchmark.py"
    path.write_text(
        path.read_text()
        .replace(
            "    def get_latency(self, *args, **kwargs):\n"
            '        pytest.fail("Dummy traversal must not time either callable")',
            "    def get_latency(self, op, *args, **kwargs):\n"
            "        op(*args, **kwargs)\n"
            "        return 1.0",
        )
        .replace(
            '    if shape == (2,):\n        pytest.skip("input unavailable")\n', ""
        )
        .replace(
            'bench.set_train(lambda x: get_candidate("megatron.dummy_probe")(x))',
            "bench.set_train(torch.neg)",
        )
    )
    result = suite.runpytest_subprocess(*paths, "--record=json")
    assert result.ret == pytest.ExitCode.OK
    accuracy = json.loads((suite.path / "accuracy_result.json").read_text())
    benchmark = json.loads((suite.path / "benchmark_result.json").read_text())
    assert accuracy["tests/test_probe.py::test_correctness"]["result"] == "passed"
    assert benchmark["dummy_probe"]["details"][0]["result"][0]["latency"] == 1.0
