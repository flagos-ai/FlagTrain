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
"""Host-only tests for the opt-in reference-as-candidate infrastructure.

The host interpreter has no torch, so the shared source and the actual
benchmark base module are loaded under private/stubbed namespaces. Nothing here
imports or mutates the real ``flag_train`` package at collection time.
"""

import importlib
import importlib.util
import os
import subprocess
import sys
import textwrap
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_SOURCE = REPO_ROOT / "src" / "flag_train" / "reference_candidate.py"
HOST_PYTHON = os.environ.get(
    "FLAGTRAIN_HOST_PYTHON",
    "/Users/tao/.local/share/uv/python/cpython-3.12.12-macos-aarch64-none/bin/python3",
)
PY39_SITE = "/Users/tao/Library/Python/3.9/lib/python/site-packages"

SYNTHETIC_CONFTEST = """
import importlib.util
import os
import sys

_SOURCE = os.environ["REF_CANDIDATE_SOURCE"]
_spec = importlib.util.spec_from_file_location("refcandidate_synth", _SOURCE)
refcandidate_synth = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = refcandidate_synth
_spec.loader.exec_module(refcandidate_synth)


def pytest_addoption(parser):
    refcandidate_synth.add_reference_candidate_argument(parser)


def pytest_configure(config):
    refcandidate_synth.configure_reference_candidate(config)
"""


def _load_shared():
    name = "_flagtrain_reference_candidate_host"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, SHARED_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run_pytest(suite: Path, *args):
    code = (
        "import sys; "
        f"sys.path.insert(0, {PY39_SITE!r}); "
        "import pytest; "
        f"raise SystemExit(pytest.main({list(args)!r}))"
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PYTEST_") and key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "REF_CANDIDATE_SOURCE": str(SHARED_SOURCE),
        }
    )
    return subprocess.run(
        [HOST_PYTHON, "-B", "-c", code],
        cwd=suite,
        env=env,
        capture_output=True,
        text=True,
    )


def _write_suite(directory, source):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "conftest.py").write_text(SYNTHETIC_CONFTEST)
    (directory / "test_synth.py").write_text(textwrap.dedent(source))
    return directory


def _run_suite(tmp_path, source, *args):
    suite = _write_suite(tmp_path / "suite", source)
    return _run_pytest(suite, "-p", "no:cacheprovider", "-q", "--tb=short", *args)


# ---------------------------------------------------------------------------
# Shared selection API
# ---------------------------------------------------------------------------


def test_select_candidate_is_identity_outside_reference_mode():
    refc = _load_shared()
    candidate = object()
    assert refc.select_candidate(candidate, None) is candidate
    with refc.reference_test_mode(False):
        assert refc.select_candidate(candidate, None) is candidate


def test_select_candidate_uses_and_observes_reference():
    refc = _load_shared()

    def candidate():
        return "candidate"

    calls = []

    def reference(value):
        calls.append(value)
        return f"reference:{value}"

    with refc.reference_test_mode(True, "case") as state:
        selected = refc.select_candidate(candidate, reference, "case")
        assert selected("x") == "reference:x"
        assert calls == ["x"]
        assert state.reference_selected is True
        assert state.candidate_calls == 1
        assert state.reference_calls == 1


def test_select_candidate_rejects_missing_and_noncallable_reference():
    refc = _load_shared()
    for reference in (None, 42):
        with refc.reference_test_mode(True, "case") as state:
            with pytest.raises(
                refc.ReferenceContractError, match="No predefined reference"
            ):
                refc.select_candidate(object(), reference, "operator")
            assert state.contract_errors


def test_get_candidate_reference_mode_skips_operator_lookup(monkeypatch):
    refc = _load_shared()
    monkeypatch.delitem(sys.modules, "flag_train", raising=False)

    def reference():
        return 7

    with refc.reference_test_mode(True):
        selected = refc.get_candidate("flag_train.megatron.absent", reference=reference)
        assert selected() == 7
    assert "flag_train" not in sys.modules


def test_get_candidate_default_mode_reads_live_public_attribute(monkeypatch):
    refc = _load_shared()
    package = types.ModuleType("flag_train")
    package.__path__ = []
    megatron = types.ModuleType("flag_train.megatron")
    megatron.__path__ = []

    def original(value):
        return value

    def override(value):
        return value + 1

    megatron.fast_gelu = original
    package.megatron = megatron
    monkeypatch.setitem(sys.modules, "flag_train", package)
    monkeypatch.setitem(sys.modules, "flag_train.megatron", megatron)

    assert refc.get_candidate("megatron.fast_gelu") is original
    assert refc.get_candidate("flag_train.megatron.fast_gelu") is original

    megatron.fast_gelu = override
    selected = refc.get_candidate("flag_train.megatron.fast_gelu")
    assert selected is override
    assert selected(1) == 2


def test_get_candidate_default_mode_reports_missing_public_attribute(monkeypatch):
    refc = _load_shared()
    package = types.ModuleType("flag_train")
    package.__path__ = []
    deepspeed = types.ModuleType("flag_train.deepspeed")
    deepspeed.__path__ = []
    package.deepspeed = deepspeed
    monkeypatch.setitem(sys.modules, "flag_train", package)
    monkeypatch.setitem(sys.modules, "flag_train.deepspeed", deepspeed)

    with pytest.raises(ImportError, match="has no public candidate"):
        refc.get_candidate("deepspeed.missing")
    with pytest.raises(ValueError, match="must be qualified"):
        refc.get_candidate("missing")


def test_reference_mode_is_scoped_and_reset_on_exception():
    refc = _load_shared()
    with pytest.raises(RuntimeError):
        with refc.reference_test_mode(True, "case"):
            assert refc.reference_mode_enabled()
            raise RuntimeError("boom")
    assert not refc.reference_mode_enabled()
    assert refc.reference_execution() is None


# ---------------------------------------------------------------------------
# Actual pytest plugin on tiny synthetic suites
# ---------------------------------------------------------------------------


def test_plugin_flag_off_leaves_ordinary_tests_unchanged(tmp_path):
    result = _run_suite(
        tmp_path,
        """
        def candidate(value):
            return value + 1

        def test_one():
            assert candidate(1) == 2

        def test_two():
            assert candidate(2) == 3
        """,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout


def test_plugin_flag_on_accepts_explicit_matching_reference(tmp_path):
    result = _run_suite(
        tmp_path,
        """
        import refcandidate_synth as refc

        def reference(value):
            return value * 2

        def test_selected():
            op = refc.get_candidate(
                "flag_train.megatron.not_exported", reference=reference
            )
            assert op(3) == 6
        """,
        "--ref-as-candidate",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


def test_plugin_flag_on_rejects_missing_and_noncallable_reference(tmp_path):
    result = _run_suite(
        tmp_path,
        """
        import refcandidate_synth as refc

        def test_missing():
            refc.get_candidate("flag_train.megatron.not_exported")

        def test_noncallable():
            refc.get_candidate(
                "flag_train.megatron.not_exported", reference=object()
            )
        """,
        "--ref-as-candidate",
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "2 failed" in result.stdout
    assert "No predefined reference" in result.stdout


def test_plugin_flag_on_fails_unadapted_ordinary_test(tmp_path):
    source = """
        def candidate(value):
            return value + 1

        def test_unadapted():
            assert candidate(1) == 2
        """
    off = _run_pytest(
        _write_suite(tmp_path / "off", source),
        "-p",
        "no:cacheprovider",
        "-q",
        "--tb=short",
    )
    assert off.returncode == 0, off.stdout + off.stderr

    on = _run_pytest(
        _write_suite(tmp_path / "on", source),
        "-p",
        "no:cacheprovider",
        "-q",
        "--tb=short",
        "--ref-as-candidate",
    )
    assert on.returncode == 1, on.stdout + on.stderr
    assert "No predefined reference selected" in on.stdout


def test_plugin_counts_are_per_case_and_fixtures_do_not_satisfy(tmp_path):
    result = _run_suite(
        tmp_path,
        """
        import pytest
        import refcandidate_synth as refc

        def reference(value):
            return value

        def test_first_case():
            op = refc.get_candidate(
                "flag_train.megatron.not_exported", reference=reference
            )
            assert op(1) == 1

        def test_previous_calls_do_not_count():
            pass

        @pytest.fixture
        def fixture_only():
            op = refc.get_candidate(
                "flag_train.megatron.not_exported", reference=reference
            )
            op(1)

        def test_body_does_not_count(fixture_only):
            pass
        """,
        "--ref-as-candidate",
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert "2 failed" in result.stdout
    assert "did not complete a candidate-side call" in result.stdout


def test_plugin_fails_through_caught_error_skip_and_xfail(tmp_path):
    result = _run_suite(
        tmp_path,
        """
        import pytest
        import refcandidate_synth as refc

        def reference(value):
            return value

        def test_caught_error():
            try:
                refc.get_candidate("flag_train.megatron.not_exported")
            except refc.ReferenceContractError:
                pass

        @pytest.mark.skip(reason="would hide the missing reference")
        def test_skipped():
            refc.get_candidate(
                "flag_train.megatron.not_exported", reference=reference
            )

        @pytest.mark.xfail(reason="would hide the missing reference")
        def test_xfail():
            refc.get_candidate("flag_train.megatron.not_exported")
        """,
        "--ref-as-candidate",
    )
    assert result.returncode == 1, result.stdout + result.stderr
    # The skip is surfaced as a setup error, never as a skipped success.
    assert "2 failed, 1 error" in result.stdout
    assert "1 skipped" not in result.stdout
    assert "1 xfailed" not in result.stdout
    assert "No predefined reference" in result.stdout


def test_plugin_benchmark_node_needs_completed_candidate_timing(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "conftest.py").write_text(SYNTHETIC_CONFTEST)
    benchmarks = suite / "benchmark"
    benchmarks.mkdir()
    (benchmarks / "test_synth.py").write_text(
        textwrap.dedent(
            """
            def test_nothing_measured():
                pass
            """
        )
    )
    result = _run_pytest(
        suite, "-p", "no:cacheprovider", "-q", "--tb=short", "--ref-as-candidate"
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "candidate-side timing" in result.stdout


def test_plugin_benchmark_node_with_no_selected_cases_is_not_false_success(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "conftest.py").write_text(SYNTHETIC_CONFTEST)
    benchmarks = suite / "benchmark"
    benchmarks.mkdir()
    (benchmarks / "test_synth.py").write_text(
        textwrap.dedent(
            """
            import refcandidate_synth as refc

            def reference(value):
                return value

            def test_measured_case():
                state = refc.reference_execution()
                op = refc.get_candidate(
                    "flag_train.megatron.not_exported", reference=reference
                )
                op(1)
                state.completed_timings += 1

            def test_no_selected_case():
                refc.reference_execution().selection_empty = True
            """
        )
    )
    result = _run_pytest(
        suite, "-p", "no:cacheprovider", "-q", "--tb=short", "--ref-as-candidate"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout


def test_plugin_rejects_collect_only(tmp_path):
    result = _run_suite(
        tmp_path,
        """
        def test_anything():
            pass
        """,
        "--ref-as-candidate",
        "--collect-only",
    )
    assert result.returncode == 4, result.stdout + result.stderr
    assert "--ref-as-candidate cannot be combined with --collect-only" in (
        result.stdout + result.stderr
    )


@pytest.mark.parametrize(
    ("option", "flag"),
    [
        ("override", "--override"),
        ("override_config", "--override-config"),
        ("list_overrides", "--list-overrides"),
        ("dummy_candidate", "--dummy-candidate"),
        ("list_cases", "--list-cases"),
        ("query", "--query"),
        ("collect_marks", "--collect-marks"),
        ("parallel", "--parallel"),
        ("numprocesses", "--numprocesses"),
        ("dist", "--dist"),
    ],
)
def test_configure_rejects_conflicting_options(option, flag):
    refc = _load_shared()
    registered = []

    class PluginManager:
        def register(self, plugin, name):
            registered.append(name)

    options = SimpleNamespace(ref_as_candidate=True, **{option: "1"})
    config = SimpleNamespace(
        option=options, pluginmanager=PluginManager()
    )
    with pytest.raises(pytest.UsageError, match=flag):
        refc.configure_reference_candidate(config)
    assert registered == []


def test_configure_ignores_absent_options_and_off_flag():
    refc = _load_shared()

    class PluginManager:
        def __init__(self):
            self.names = []

        def register(self, plugin, name):
            self.names.append(name)

    off = SimpleNamespace(
        option=SimpleNamespace(ref_as_candidate=False), pluginmanager=PluginManager()
    )
    refc.configure_reference_candidate(off)
    assert off.pluginmanager.names == []

    on = SimpleNamespace(
        option=SimpleNamespace(ref_as_candidate=True), pluginmanager=PluginManager()
    )
    refc.configure_reference_candidate(on)
    assert on.pluginmanager.names == ["flag-train-reference-candidate"]
    # Registering twice is a mixed tests+benchmark run, not an error.
    refc.configure_reference_candidate(on)
    assert on.pluginmanager.names == ["flag-train-reference-candidate"]


def test_configure_accepts_xdist_default_dist_no():
    refc = _load_shared()

    class PluginManager:
        def register(self, plugin, name):
            pass

    config = SimpleNamespace(
        option=SimpleNamespace(ref_as_candidate=True, dist="no"),
        pluginmanager=PluginManager(),
    )
    refc.configure_reference_candidate(config)


# ---------------------------------------------------------------------------
# Actual benchmark selection and timing path, with controlled stubs
# ---------------------------------------------------------------------------


class _DeviceFunctions:
    @staticmethod
    def synchronize():
        pass


class _UseTrain:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _install_benchmark_stubs():
    """Install the minimal modules benchmark/base.py needs at import time."""
    torch = types.ModuleType("torch")

    class Tensor:
        pass

    class Dtype:
        pass

    torch.Tensor = Tensor
    torch.dtype = Dtype
    for name in (
        "float16",
        "float32",
        "bfloat16",
        "float64",
        "int8",
        "uint8",
        "int16",
        "int32",
        "int64",
        "bool",
        "complex64",
        "cfloat",
    ):
        setattr(torch, name, Dtype())
    torch.backends = SimpleNamespace(
        mudnn=SimpleNamespace(allow_tf32=True),
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        cudnn=SimpleNamespace(allow_tf32=True),
    )
    sys.modules["torch"] = torch

    triton = types.ModuleType("triton")
    sys.modules["triton"] = triton

    flag_train = types.ModuleType("flag_train")
    flag_train.__path__ = [str(REPO_ROOT / "src" / "flag_train")]
    flag_train.device = "cpu"
    flag_train.vendor_name = "cpu"
    flag_train.use_train = _UseTrain
    runtime = types.ModuleType("flag_train.runtime")
    runtime.torch_backend_device = SimpleNamespace(
        matmul=SimpleNamespace(allow_tf32=True)
    )
    runtime.torch_device_fn = _DeviceFunctions
    flag_train.runtime = runtime
    utils = types.ModuleType("flag_train.utils")
    utils.__path__ = []
    shape_utils = types.ModuleType("flag_train.utils.shape_utils")
    utils.shape_utils = shape_utils
    sys.modules["flag_train"] = flag_train
    sys.modules["flag_train.runtime"] = runtime
    sys.modules["flag_train.utils"] = utils
    sys.modules["flag_train.utils.shape_utils"] = shape_utils

    benchmark_pkg = types.ModuleType("benchmark")
    benchmark_pkg.__path__ = [str(REPO_ROOT / "benchmark")]
    sys.modules["benchmark"] = benchmark_pkg

    return torch, flag_train


def _install_benchmark_conftest_stub():
    module = types.ModuleType("benchmark.conftest")
    captured = {"results": [], "logs": []}

    class Config:
        mode = None
        skip_native = False
        warm_up = 0
        repetition = 0
        native_baseline_skip_reason = None
        base_data = None
        bench_level = None
        query = False
        record_json = False
        record_log = False

    module.Config = Config
    module.emit_record_logger = lambda message: captured["logs"].append(message)
    module.update_case_list = lambda data: None
    module.update_result = lambda op, data: captured["results"].append(data)
    sys.modules["benchmark.conftest"] = module
    return module, captured


@contextmanager
def _loaded_benchmark():
    prefixes = ("flag_train", "benchmark", "torch", "triton")
    saved = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name in prefixes or any(name.startswith(p + ".") for p in prefixes)
    }
    try:
        _install_benchmark_stubs()
        conftest, captured = _install_benchmark_conftest_stub()
        consts = importlib.import_module("benchmark.consts")
        base = importlib.import_module("benchmark.base")
        refc = importlib.import_module("flag_train.reference_candidate")
        conftest.Config.mode = consts.BenchMode.OPERATOR
        conftest.Config.bench_level = consts.BenchLevel.CORE
        yield SimpleNamespace(
            base=base,
            consts=consts,
            refc=refc,
            Config=conftest.Config,
            captured=captured,
        )
    finally:
        for name in list(sys.modules):
            if name in prefixes or any(name.startswith(p + ".") for p in prefixes):
                del sys.modules[name]
        sys.modules.update(saved)


def _tick():
    """Give the real OPERATOR timing loop a non-zero elapsed time."""
    sum(range(200))


def _make_bench(env, torch_op, train_op=None, metrics=("latency",)):
    bench = env.base.Benchmark(op_name="synthetic", torch_op=torch_op)
    bench.train_op = train_op
    bench.to_bench_metrics = list(metrics)
    return bench


def test_benchmark_reference_mode_times_torch_op_not_train_op():
    with _loaded_benchmark() as env:
        reference_calls = []
        train_calls = []

        def reference(value):
            reference_calls.append(value)
            _tick()
            return value

        def train_op(value):
            train_calls.append(value)
            _tick()
            return value

        bench = _make_bench(env, reference, train_op)
        with env.refc.reference_test_mode(True, "benchmark.node") as state:
            metric = bench._measure_input((1,))

        assert train_calls == []
        assert reference_calls
        assert state.reference_selected is True
        assert state.candidate_calls > 0
        assert state.completed_timings == 1
        assert metric.candidate_source == "reference"
        assert metric.latency is not None


def test_benchmark_default_path_still_prefers_train_op():
    with _loaded_benchmark() as env:
        reference_calls = []
        train_calls = []

        def reference(value):
            reference_calls.append(value)
            _tick()
            return value

        def train_op(value):
            train_calls.append(value)
            _tick()
            return value

        bench = _make_bench(
            env, reference, train_op, metrics=("latency_base", "latency")
        )
        metric = bench._measure_input((1,))

        assert train_calls
        assert reference_calls  # baseline timing still uses torch_op
        assert not hasattr(metric, "candidate_source")
        assert env.refc.reference_mode_enabled() is False


def test_benchmark_baseline_call_alone_is_not_candidate_execution():
    with _loaded_benchmark() as env:
        reference_calls = []

        def reference(value):
            reference_calls.append(value)
            _tick()
            return value

        bench = _make_bench(env, reference, metrics=("latency_base",))
        with env.refc.reference_test_mode(True, "benchmark.node") as state:
            metric = bench._measure_input((1,))

        assert reference_calls
        assert state.candidate_calls == 0
        assert state.completed_timings == 0
        assert not hasattr(metric, "candidate_source")


def test_benchmark_missing_reference_fails():
    with _loaded_benchmark() as env:
        bench = _make_bench(env, None)
        with env.refc.reference_test_mode(True, "benchmark.node") as state:
            with pytest.raises(
                pytest.fail.Exception, match="No predefined reference"
            ):
                bench._measure_input((1,))
        assert state.contract_errors


def test_benchmark_reference_mode_requires_latency_metric():
    with _loaded_benchmark() as env:
        bench = _make_bench(env, lambda value: value, metrics=("latency_base",))
        bench.init_user_config = lambda: None
        with env.refc.reference_test_mode(True, "benchmark.node"):
            with pytest.raises(
                env.refc.ReferenceContractError,
                match="requires the candidate latency metric",
            ):
                bench.run()


def test_benchmark_reference_mode_rejects_backward_timer():
    with _loaded_benchmark() as env:
        bench = env.base.Benchmark(
            op_name="synthetic_backward",
            torch_op=lambda value: value,
            is_backward=True,
        )
        bench.to_bench_metrics = ["latency"]
        with env.refc.reference_test_mode(True, "benchmark.node") as state:
            with pytest.raises(
                env.refc.ReferenceContractError, match="does not support is_backward"
            ):
                bench._measure_input((1,))
        assert state.contract_errors


def test_benchmark_report_marks_reference_source_only_when_selected():
    with _loaded_benchmark() as env:
        bench = _make_bench(env, lambda value: value)
        metric = env.consts.BenchmarkMetrics(latency=1.0, shape_detail=(1,))
        metric.candidate_source = "reference"

        with env.refc.reference_test_mode(True, "benchmark.node"):
            result = bench._emit_result("torch.float32", [metric])
        recorded = env.captured["results"][-1]
        assert recorded["result"][0]["candidate_source"] == "reference"
        assert '"candidate_source": "reference"' in result.to_json()

        default_metric = env.consts.BenchmarkMetrics(latency=1.0, shape_detail=(1,))
        default_result = bench._emit_result("torch.float32", [default_metric])
        default_record = env.captured["results"][-1]
        assert "candidate_source" not in default_record["result"][0]
        assert "candidate_source" not in default_result.to_json()
