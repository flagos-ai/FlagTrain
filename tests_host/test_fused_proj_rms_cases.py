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
"""Host-only regressions for the fused_proj_rms_backward case contract.

The benchmark tree imports torch, triton and flag_train, none of which is a
usable CPU-only dependency here.  These tests therefore install a private fake
tree for the duration of each test (never at collection time), load the real
benchmark modules under an isolated package name, and exercise the actual
``case_fn``/``build_inputs_fn``/``GenericBenchmark`` code paths.  The fake
tensors record allocation and arithmetic so input construction can be compared
with the legacy generator; nothing here claims a GPU execution or a timing
measurement.

One test drives the real pytest entrypoint in a subprocess with
``--list-cases`` and with an unknown ``--case-id``.  That exercises option
registration, collection, the case-list JSON writer and the explicit
unsupported-ID failure without materializing tensors or timing an operator.
"""

import gc
import importlib
import json
import math
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "benchmark"
PILOT_REL = "benchmark/megatron/test_fused_proj_rms_backward.py"
NODEID = f"{PILOT_REL}::test_fused_proj_rms_backward_perf"

SYNTH_BF16 = "_flagtrain_benchmark_host_bf16"
SYNTH_FP16 = "_flagtrain_benchmark_host_fp16"
_OWNED_PREFIXES = (
    "torch",
    "triton",
    "flag_train",
    "megatron",
    "_flagtrain_benchmark_host",
)

# The exact grid the pilot benchmark has always timed: ``(M, N, K)`` with
# ``x[M, K]``, ``weight[N, K]``, ``grad_proj[M, N]``, ``grad_r[M, 1]`` and
# ``r[M, 1]``.
EXPECTED_SHAPES = [
    (1, 24, 4096),
    (128, 24, 4096),
    (256, 24, 4096),
    (1024, 24, 4096),
    (4096, 24, 4096),
    (8192, 24, 8192),
    (4096, 80, 4096),
    (4096, 288, 4096),
    (4096, 1024, 4096),
    (8192, 4096, 8192),
]
EXPECTED_EPS = 1e-6


class _Calls:
    """Ordered record of every fake tensor/operator interaction."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.events = []
        self.backward_args = None

    def add(self, kind, **fields):
        self.events.append((kind, fields))


CALLS = _Calls()


class FakeDtype:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"torch.{self.name}"

    __repr__ = __str__

    def __eq__(self, other):
        return isinstance(other, FakeDtype) and other.name == self.name

    def __hash__(self):
        return hash((FakeDtype, self.name))


FLOAT16 = FakeDtype("float16")
FLOAT32 = FakeDtype("float32")
BFLOAT16 = FakeDtype("bfloat16")


class FakeTensor:
    def __init__(self, shape, dtype, device, tag=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.tag = tag
        self.requires_grad = False

    def __truediv__(self, scalar):
        CALLS.add("div", shape=self.shape, dtype=str(self.dtype), scalar=scalar)
        return FakeTensor(self.shape, self.dtype, self.device, tag=self.tag)

    def to(self, dtype):
        CALLS.add("to", shape=self.shape, dtype=str(dtype))
        return FakeTensor(self.shape, dtype, self.device, tag=self.tag)

    def size(self):
        return self.shape

    def element_size(self):
        return 4 if self.dtype == FLOAT32 else 2

    def numel(self):
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    def detach(self):
        return self

    def clone(self):
        return FakeTensor(self.shape, self.dtype, self.device, tag=self.tag)

    def requires_grad_(self, value=True):
        self.requires_grad = value
        return self


def _describe(tensor):
    return (tensor.shape, str(tensor.dtype), tensor.device, tensor.tag)


def _randn(shape, dtype=None, device=None):
    CALLS.add("randn", shape=tuple(shape), dtype=str(dtype), device=device)
    return FakeTensor(shape, dtype, device)


def _compile(fn=None, **kwargs):
    if fn is None:
        return lambda wrapped: wrapped
    return fn


def _autograd_grad(outputs, inputs, grad_outputs=None, retain_graph=False):
    CALLS.add(
        "autograd_grad",
        outputs=tuple(_describe(item) for item in outputs),
        inputs=tuple(_describe(item) for item in inputs),
        retain_graph=retain_graph,
    )
    return tuple(FakeTensor(item.shape, item.dtype, item.device) for item in inputs)


def _make_torch():
    torch = types.ModuleType("torch")
    torch.Tensor = FakeTensor
    # ``torch.dtype`` must be a class: base.py uses it in isinstance().
    torch.dtype = FakeDtype
    torch.float16 = FLOAT16
    torch.float32 = FLOAT32
    torch.bfloat16 = BFLOAT16
    torch.int16 = FakeDtype("int16")
    torch.int32 = FakeDtype("int32")
    torch.bool = FakeDtype("bool")
    torch.complex64 = FakeDtype("complex64")
    torch.cfloat = torch.complex64
    torch.int8 = FakeDtype("int8")
    torch.uint8 = FakeDtype("uint8")
    torch.int64 = FakeDtype("int64")
    torch.Size = tuple
    torch.randn = _randn
    torch.compile = _compile
    torch.is_tensor = lambda item: isinstance(item, FakeTensor)
    torch.is_floating_point = lambda item: getattr(item, "dtype", None) in (
        FLOAT16,
        FLOAT32,
        BFLOAT16,
    )

    autograd = types.ModuleType("torch.autograd")
    autograd.grad = _autograd_grad
    torch.autograd = autograd

    functorch = types.ModuleType("torch._functorch")
    functorch_config = types.ModuleType("torch._functorch.config")
    functorch_config.donated_buffer = True
    functorch.config = functorch_config
    torch._functorch = functorch

    backends = types.ModuleType("torch.backends")
    backends.mudnn = SimpleNamespace(allow_tf32=True)
    backends.cuda = SimpleNamespace(
        matmul=SimpleNamespace(allow_tf32=True),
        cudnn=SimpleNamespace(allow_tf32=True),
    )
    torch.backends = backends

    cuda = types.ModuleType("torch.cuda")
    cuda.is_available = lambda: False
    cuda.get_device_capability = lambda *args, **kwargs: (0, 0)
    torch.cuda = cuda

    return torch


def _make_triton():
    triton = types.ModuleType("triton")
    triton.__version__ = "0.0-host-stub"
    testing = types.ModuleType("triton.testing")
    testing.do_bench = lambda fn, **kwargs: 0.001
    triton.testing = testing
    return triton


def _make_flag_train():
    calls = CALLS

    def fused_proj_rms_forward(x, weight, eps=1e-6):
        calls.add(
            "forward",
            x=_describe(x),
            weight=_describe(weight),
            eps=eps,
        )
        m = x.shape[0]
        n = weight.shape[0]
        proj = FakeTensor((m, n), x.dtype, x.device, tag="proj")
        r = FakeTensor((m, 1), x.dtype, x.device, tag="forward_r")
        return proj, r

    def fused_proj_rms_backward(*args):
        calls.backward_args = args
        calls.add(
            "backward",
            args=tuple(_describe(item) for item in args),
        )
        return ("grad_x", "grad_weight")

    root = types.ModuleType("flag_train")
    root.__path__ = []
    root.device = "cuda"
    root.vendor_name = "nvidia"

    runtime = types.ModuleType("flag_train.runtime")
    runtime.__path__ = []
    runtime.device = SimpleNamespace(
        name="cuda", vendor_name="nvidia", support_bf16=True
    )
    runtime.torch_backend_device = SimpleNamespace(
        matmul=SimpleNamespace(allow_tf32=True)
    )
    runtime.torch_device_fn = SimpleNamespace(
        synchronize=lambda: None, empty_cache=lambda: None
    )
    root.runtime = runtime

    megatron = types.ModuleType("flag_train.megatron")
    megatron.fused_proj_rms_forward = fused_proj_rms_forward
    megatron.fused_proj_rms_backward = fused_proj_rms_backward
    root.megatron = megatron

    utils = types.ModuleType("flag_train.utils")
    utils.__path__ = []
    shape_utils = types.ModuleType("flag_train.utils.shape_utils")
    utils.shape_utils = shape_utils
    root.utils = utils

    cli_override = types.ModuleType("flag_train.cli_override")

    def add_override_arguments(parser):
        return None

    class _Registry:
        def restore_all(self):
            return None

    cli_override.add_override_arguments = add_override_arguments
    cli_override.apply_overrides_from_args = lambda options: _Registry()
    root.cli_override = cli_override

    modules = {
        "flag_train": root,
        "flag_train.runtime": runtime,
        "flag_train.megatron": megatron,
        "flag_train.utils": utils,
        "flag_train.utils.shape_utils": shape_utils,
        "flag_train.cli_override": cli_override,
    }
    return modules


def _install_fake_modules():
    """Install the minimal torch/triton/flag_train tree the benchmark imports."""
    torch = _make_torch()
    modules = {
        "torch": torch,
        "torch.autograd": torch.autograd,
        "torch._functorch": torch._functorch,
        "torch._functorch.config": torch._functorch.config,
        "torch.backends": torch.backends,
        "torch.cuda": torch.cuda,
        "triton": _make_triton(),
    }
    modules.update(_make_flag_train())
    sys.modules.update(modules)
    return modules


def _install_entrypoint_stubs():
    """Subprocess entrypoint: install fakes before pytest imports benchmark."""
    _install_fake_modules()


def _owned(name):
    return any(
        name == prefix or name.startswith(prefix + ".") for prefix in _OWNED_PREFIXES
    )


def _snapshot_modules():
    return {name: module for name, module in sys.modules.items() if _owned(name)}


def _drop_owned_modules():
    for name in list(sys.modules):
        if _owned(name):
            del sys.modules[name]


def _make_package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module
    return module


def _configure_config(config, conftest, level, nodeid=NODEID):
    config.mode = conftest.consts.BenchMode.KERNEL
    config.bench_level = conftest.consts.BenchLevel(level)
    config.current_nodeid = nodeid
    config.user_desired_dtypes = None
    config.user_desired_metrics = None
    config.shape_file = str(BENCHMARK_DIR / "core_shapes.yaml")
    config.query = False
    config.list_cases = False
    config.case_ids = None
    config.record_log = False
    config.record_json = False
    config.base_data = None
    config.skip_native = False
    config.native_baseline_skip_reason = None
    config.available_case_ids = set()
    config.executed_case_ids = set()
    config.parallel = 0
    config.mm_layout = None
    config.warm_up = 1
    config.repetition = 1


def _load_subject(prefix, support_bf16):
    runtime = sys.modules["flag_train.runtime"]
    previous = runtime.device.support_bf16
    runtime.device.support_bf16 = support_bf16
    try:
        _make_package(prefix, BENCHMARK_DIR)
        _make_package(f"{prefix}.megatron", BENCHMARK_DIR / "megatron")
        # The real pytest session builds Config before benchmark modules are
        # imported; reproduce that ordering so base.py binds this instance.
        conftest = importlib.import_module(f"{prefix}.conftest")
        config = conftest.BenchConfig()
        _configure_config(config, conftest, "core")
        conftest.Config = config
        base = importlib.import_module(f"{prefix}.base")
        consts = importlib.import_module(f"{prefix}.consts")
        pilot = importlib.import_module(
            f"{prefix}.megatron.test_fused_proj_rms_backward"
        )
    finally:
        runtime.device.support_bf16 = previous
    return SimpleNamespace(
        prefix=prefix,
        base=base,
        conftest=conftest,
        consts=consts,
        pilot=pilot,
    )


@pytest.fixture(scope="module")
def host_bench():
    saved = _snapshot_modules()
    _drop_owned_modules()
    _install_fake_modules()
    CALLS.reset()
    try:
        bf16 = _load_subject(SYNTH_BF16, True)
        fp16 = _load_subject(SYNTH_FP16, False)
        yield SimpleNamespace(
            bf16=bf16,
            fp16=fp16,
            calls=CALLS,
            torch=sys.modules["torch"],
            flag_train=sys.modules["flag_train"],
        )
    finally:
        CALLS.reset()
        gc.collect()
        _drop_owned_modules()
        sys.modules.update(saved)


def _make_bench(subject, level="core"):
    _configure_config(subject.conftest.Config, subject.conftest, level)
    bench = subject.pilot.FusedProjRmsBackwardBenchmark(
        case_fn=subject.pilot.fused_proj_rms_backward_case,
        build_inputs_fn=subject.pilot.fused_proj_rms_backward_build_inputs,
        op_name="fused_proj_rms_backward",
        torch_op=subject.pilot.torch_op,
        dtypes=subject.pilot._DTYPES,
    )
    bench.set_train(subject.pilot.train_op)
    return bench


def _expected_case_rows(dtype):
    rows = []
    for ordinal, (m, n, k) in enumerate(EXPECTED_SHAPES):
        rows.append(
            (
                f"{NODEID}::core::{dtype}::{ordinal}",
                ordinal,
                f"torch.{dtype}",
                {
                    "x": (m, k),
                    "weight": (n, k),
                    "grad_proj": (m, n),
                    "grad_r": (m, 1),
                    "r": (m, 1),
                },
                {"eps": EXPECTED_EPS},
            )
        )
    return rows


def _case_rows(cases):
    return [
        (
            case.case_id,
            case.ordinal,
            str(case.dtype),
            case.shape,
            case.params,
        )
        for case in cases
    ]


def test_case_plans_describe_the_exact_inputs(host_bench):
    subject = host_bench.bf16
    assert subject.pilot._BENCHMARK_SHAPES == EXPECTED_SHAPES
    assert subject.pilot._EPS == EXPECTED_EPS
    assert subject.pilot._DTYPES == [FLOAT16, BFLOAT16]
    assert subject.conftest.Config is subject.base.Config

    bench = _make_bench(subject)
    case_list = bench.list_cases(initialize=True)

    assert case_list.op_name == "fused_proj_rms_backward"
    assert case_list.level == "core"
    assert case_list.phase == "timing"
    assert case_list.schema_version == "flagtrain.benchmark-case-list/v2"
    assert len(case_list.cases) == 2 * len(EXPECTED_SHAPES)

    expected = _expected_case_rows("float16") + _expected_case_rows("bfloat16")
    assert _case_rows(case_list.cases) == expected

    for case in case_list.cases:
        assert isinstance(case.builder_args[0], subject.consts.BenchmarkCasePlan)
        plan = case.builder_args[0]
        assert plan.builder_args == (EXPECTED_SHAPES[case.ordinal],)
        assert plan.params == {"eps": EXPECTED_EPS}
        assert all(
            isinstance(dim, int) for shape in plan.shape.values() for dim in shape
        )

    payload = case_list.to_dict()
    assert payload["schema_version"] == "flagtrain.benchmark-case-list/v2"
    assert [case["case_id"] for case in payload["cases"]] == [
        case.case_id for case in case_list.cases
    ]
    for case_dict in payload["cases"]:
        assert set(case_dict) == {"case_id", "ordinal", "dtype", "shape", "params"}
        assert "builder_args" not in case_dict
    encoded = json.dumps(payload)
    assert "builder_args" not in encoded
    decoded = json.loads(encoded)
    assert decoded["cases"][0]["shape"]["x"] == [
        EXPECTED_SHAPES[0][0],
        EXPECTED_SHAPES[0][2],
    ]
    assert decoded["cases"][0]["params"]["eps"] == EXPECTED_EPS


def test_list_cases_allocates_nothing_and_runs_no_math(host_bench, monkeypatch):
    subject = host_bench.bf16
    bench = _make_bench(subject)

    def boom(*args, **kwargs):
        raise AssertionError("listing must not materialize inputs or run operators")

    monkeypatch.setattr(bench, "build_inputs", boom)
    monkeypatch.setattr(bench, "_measure_input", boom)
    monkeypatch.setattr(
        subject.pilot.flag_train.megatron, "fused_proj_rms_forward", boom
    )
    monkeypatch.setattr(
        subject.pilot.flag_train.megatron, "fused_proj_rms_backward", boom
    )
    monkeypatch.setattr(host_bench.torch, "randn", boom)
    monkeypatch.setattr(host_bench.torch.autograd, "grad", boom)
    monkeypatch.setattr(subject.pilot, "_native_proj_rms", boom)

    host_bench.calls.reset()
    case_list = bench.list_cases(initialize=True)

    assert len(case_list.cases) == 20
    assert host_bench.calls.events == []
    assert host_bench.calls.backward_args is None


def test_core_and_comprehensive_list_the_same_grid(host_bench):
    subject = host_bench.bf16
    rows_by_level = {}
    for level in ("core", "comprehensive"):
        bench = _make_bench(subject, level=level)
        case_list = bench.list_cases(initialize=True)
        assert case_list.level == level
        assert len(case_list.cases) == 20
        assert len({case.case_id for case in case_list.cases}) == 20
        rows_by_level[level] = [
            (str(case.dtype), case.ordinal, case.shape, case.params)
            for case in case_list.cases
        ]

    assert rows_by_level["core"] == rows_by_level["comprehensive"]
    core_ids = {
        case.case_id for case in _make_bench(subject, "core").list_cases(True).cases
    }
    comprehensive_ids = {
        case.case_id
        for case in _make_bench(subject, "comprehensive").list_cases(True).cases
    }
    assert core_ids.isdisjoint(comprehensive_ids)
    assert all("::core::" in case_id for case_id in core_ids)
    assert all("::comprehensive::" in case_id for case_id in comprehensive_ids)


def test_default_run_uses_the_legacy_grid_and_factory(host_bench):
    subject = host_bench.bf16
    bench = _make_bench(subject)
    bench.init_user_config()

    assert bench.supports_cases() is True
    assert bench.shapes == EXPECTED_SHAPES
    assert bench.to_bench_dtypes == [FLOAT16, BFLOAT16]
    assert (
        bench.to_bench_metrics
        == subject.pilot.FusedProjRmsBackwardBenchmark.DEFAULT_METRICS
    )

    cases = bench._collect_cases()
    assert len(cases) == 20
    assert [case.dtype for case in cases] == [FLOAT16] * 10 + [BFLOAT16] * 10

    for case in cases:
        m, n, k = EXPECTED_SHAPES[case.ordinal]

        host_bench.calls.reset()
        new_inputs = bench.build_inputs(case)
        new_events = list(host_bench.calls.events)

        assert len(new_inputs) == 5
        assert all(isinstance(item, FakeTensor) for item in new_inputs)
        assert [item.shape for item in new_inputs] == [
            (m, k),
            (n, k),
            (m, n),
            (m, 1),
            (m, 1),
        ]
        assert [item.dtype for item in new_inputs] == [case.dtype] * 5
        # ``r`` is the forward's output, not a fresh draw.
        assert new_inputs[4].tag == "forward_r"

        assert [event[0] for event in new_events] == [
            "randn",
            "randn",
            "div",
            "to",
            "randn",
            "randn",
            "forward",
        ]
        assert new_events[0][1] == {
            "shape": (m, k),
            "dtype": str(case.dtype),
            "device": "cuda",
        }
        assert new_events[1][1] == {
            "shape": (n, k),
            "dtype": "torch.float32",
            "device": "cuda",
        }
        assert new_events[2][1]["scalar"] == math.sqrt(max(k, 1))
        assert new_events[2][1]["shape"] == (n, k)
        assert new_events[3][1] == {"shape": (n, k), "dtype": str(case.dtype)}
        assert new_events[4][1]["shape"] == (m, n)
        assert new_events[5][1]["shape"] == (m, 1)
        assert new_events[6][1] == {
            "x": ((m, k), str(case.dtype), "cuda", None),
            "weight": ((n, k), str(case.dtype), "cuda", None),
            "eps": EXPECTED_EPS,
        }

        host_bench.calls.reset()
        old_inputs = next(
            subject.pilot.fused_proj_rms_backward_input_fn(
                (m, n, k), case.dtype, "cuda"
            )
        )
        assert host_bench.calls.events == new_events
        assert [(item.shape, str(item.dtype), item.tag) for item in old_inputs] == [
            (item.shape, str(item.dtype), item.tag) for item in new_inputs
        ]


def test_fp16_only_grid_when_bfloat16_is_unsupported(host_bench):
    assert host_bench.bf16.pilot._DTYPES == [FLOAT16, BFLOAT16]
    assert host_bench.fp16.pilot._DTYPES == [FLOAT16]

    bench = _make_bench(host_bench.fp16, level="comprehensive")
    case_list = bench.list_cases(initialize=True)
    assert len(case_list.cases) == len(EXPECTED_SHAPES)
    assert {str(case.dtype) for case in case_list.cases} == {"torch.float16"}
    assert [case.ordinal for case in case_list.cases] == list(
        range(len(EXPECTED_SHAPES))
    )


def test_default_run_executes_every_case(host_bench, monkeypatch):
    subject = host_bench.bf16
    bench = _make_bench(subject)
    expected_ids = [case.case_id for case in bench._collect_cases()]
    assert len(expected_ids) == 20

    built = []
    measured = []
    original_build = bench.build_inputs

    def build_spy(case):
        built.append(case.case_id)
        return original_build(case)

    def measure_spy(inputs, case_id=None):
        measured.append(case_id)
        assert len(inputs) == 5
        return subject.consts.BenchmarkMetrics()

    monkeypatch.setattr(bench, "build_inputs", build_spy)
    monkeypatch.setattr(bench, "_measure_input", measure_spy)
    monkeypatch.setattr(bench, "_record_base_comparison", lambda *args, **kwargs: None)

    results = bench.run()

    assert built == expected_ids
    assert measured == expected_ids
    assert [result.dtype for result in results] == [
        "torch.float16",
        "torch.bfloat16",
    ]
    assert subject.conftest.Config.executed_case_ids == set(expected_ids)


def test_case_id_materializes_and_measures_only_the_selected_case(
    host_bench, monkeypatch
):
    subject = host_bench.bf16
    bench = _make_bench(subject)
    cases = bench._collect_cases()
    chosen = cases[13]  # bfloat16, ordinal 3
    assert str(chosen.dtype) == "torch.bfloat16"

    built = []
    measured = []
    original_build = bench.build_inputs

    def build_spy(case):
        built.append(case.case_id)
        return original_build(case)

    def measure_spy(inputs, case_id=None):
        measured.append(case_id)
        assert len(inputs) == 5
        return subject.consts.BenchmarkMetrics()

    monkeypatch.setattr(bench, "build_inputs", build_spy)
    monkeypatch.setattr(bench, "_measure_input", measure_spy)
    monkeypatch.setattr(bench, "_record_base_comparison", lambda *args, **kwargs: None)

    subject.conftest.Config.case_ids = [chosen.case_id]
    results = bench.run(case_ids=[chosen.case_id])

    assert built == [chosen.case_id]
    assert measured == [chosen.case_id]
    assert len(results) == 1
    assert results[0].dtype == "torch.bfloat16"
    assert subject.conftest.Config.executed_case_ids == {chosen.case_id}

    lines = []
    reporter = SimpleNamespace(write_line=lines.append)
    session = SimpleNamespace(
        exitstatus=pytest.ExitCode.OK,
        config=SimpleNamespace(
            pluginmanager=SimpleNamespace(
                get_plugin=lambda name: reporter if name == "terminalreporter" else None
            )
        ),
    )
    subject.conftest.pytest_sessionfinish(session, None)
    assert session.exitstatus == pytest.ExitCode.OK
    assert not any("case selection failed" in line for line in lines)


def test_unknown_case_id_remains_an_explicit_error(host_bench):
    subject = host_bench.bf16
    bench = _make_bench(subject)
    missing = f"{NODEID}::core::float16::999"

    built = []
    original_build = bench.build_inputs

    def build_spy(case):
        built.append(case.case_id)
        return original_build(case)

    bench.build_inputs = build_spy
    try:
        subject.conftest.Config.case_ids = [missing]
        assert bench.run(case_ids=[missing]) == []
    finally:
        host_bench.calls.reset()

    assert built == []
    assert subject.conftest.Config.executed_case_ids == set()

    lines = []
    reporter = SimpleNamespace(write_line=lines.append)
    session = SimpleNamespace(
        exitstatus=pytest.ExitCode.OK,
        config=SimpleNamespace(
            pluginmanager=SimpleNamespace(
                get_plugin=lambda name: reporter if name == "terminalreporter" else None
            )
        ),
    )
    subject.conftest.pytest_sessionfinish(session, None)

    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert any("case selection failed" in line for line in lines)
    assert any(missing in line for line in lines)


def test_selected_inputs_feed_the_unchanged_operators(host_bench, monkeypatch):
    subject = host_bench.bf16
    bench = _make_bench(subject)
    case = bench._collect_cases()[0]
    inputs = bench.build_inputs(case)
    m, n, k = EXPECTED_SHAPES[0]

    host_bench.calls.reset()
    result = subject.pilot.train_op(*inputs)
    assert result == ("grad_x", "grad_weight")
    assert host_bench.calls.backward_args == inputs
    assert [event[0] for event in host_bench.calls.events] == ["backward"]

    baseline_calls = []

    def baseline_spy(*args):
        baseline_calls.append(args)
        return ("grad_x_ref", "grad_weight_ref")

    monkeypatch.setattr(subject.pilot, "_baseline", baseline_spy)
    assert subject.pilot.torch_op(*inputs) == ("grad_x_ref", "grad_weight_ref")
    assert len(baseline_calls) == 1
    assert baseline_calls[0] == inputs

    assert subject.pilot._EPS == EXPECTED_EPS
    assert bench.get_tflops(None, *inputs) == 4 * m * n * k
    assert bench.get_gbps(inputs, latency=0.001) > 0


_ENTRYPOINT_RUNNER = r"""
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1])
level = sys.argv[2]
output = Path(sys.argv[3])
for support in sys.argv[4:]:
    if support not in sys.path:
        sys.path.insert(0, support)
sys.path.insert(0, str(repo / "tests_host"))
sys.path.insert(0, str(repo))

import test_fused_proj_rms_cases as host

host._install_entrypoint_stubs()

import pytest

code = pytest.main(
    [
        str(repo / "benchmark/megatron/test_fused_proj_rms_backward.py"),
        "--level",
        level,
        "--list-cases",
        "--output",
        str(output),
        "-p",
        "no:cacheprovider",
        "-q",
    ]
)
if not output.exists():
    print("missing case list output", file=sys.stderr)
    sys.exit(2)
print("PYTEST_EXIT", int(code))
sys.exit(int(code))
"""


_ENTRYPOINT_UNKNOWN_RUNNER = r"""
import sys
from pathlib import Path

repo = Path(sys.argv[1])
for support in sys.argv[2:]:
    if support not in sys.path:
        sys.path.insert(0, support)
sys.path.insert(0, str(repo / "tests_host"))
sys.path.insert(0, str(repo))

import test_fused_proj_rms_cases as host

host._install_entrypoint_stubs()

import pytest

code = pytest.main(
    [
        str(repo / "benchmark/megatron/test_fused_proj_rms_backward.py"),
        "--case-id",
        "missing::case::id",
        "-p",
        "no:cacheprovider",
        "-q",
    ]
)
print("PYTEST_EXIT", int(code))
sys.exit(0)
"""


def _support_paths():
    paths = {str(Path(pytest.__file__).resolve().parent.parent)}
    try:
        import yaml
    except ImportError:
        return sorted(paths)
    paths.add(str(Path(yaml.__file__).resolve().parent.parent))
    return sorted(paths)


def _run_entrypoint(script, *args):
    command = [sys.executable, "-B", "-c", script, str(REPO_ROOT), *args]
    return subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("level", ["core", "comprehensive"])
def test_real_pytest_list_cases_entrypoint(tmp_path, level):
    output = tmp_path / f"cases-{level}.json"
    support = _support_paths()
    proc = _run_entrypoint(_ENTRYPOINT_RUNNER, level, str(output), *support)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PYTEST_EXIT 0" in proc.stdout

    payload = json.loads(output.read_text())
    assert payload["schema_version"] == "flagtrain.benchmark-case-list/v2"
    assert len(payload["benchmarks"]) == 1
    listed = payload["benchmarks"][0]
    assert listed["op_name"] == "fused_proj_rms_backward"
    assert listed["level"] == level
    assert listed["phase"] == "timing"
    assert len(listed["cases"]) == 2 * len(EXPECTED_SHAPES)

    cases = listed["cases"]
    expected = []
    for dtype in ("float16", "bfloat16"):
        for ordinal, (m, n, k) in enumerate(EXPECTED_SHAPES):
            expected.append(
                (
                    f"::{level}::{dtype}::{ordinal}",
                    ordinal,
                    f"torch.{dtype}",
                    [m, k],
                    [n, k],
                    [m, n],
                    [m, 1],
                    [m, 1],
                    EXPECTED_EPS,
                )
            )

    actual = [
        (
            case["case_id"][-len(expected[index][0]) :],
            case["ordinal"],
            case["dtype"],
            case["shape"]["x"],
            case["shape"]["weight"],
            case["shape"]["grad_proj"],
            case["shape"]["grad_r"],
            case["shape"]["r"],
            case["params"]["eps"],
        )
        for index, case in enumerate(cases)
    ]
    assert actual == expected
    assert len({case["case_id"] for case in cases}) == len(cases)
    assert all("test_fused_proj_rms_backward_perf" in case["case_id"] for case in cases)
    assert all("builder_args" not in case for case in cases)


def test_real_pytest_unknown_case_id_is_rejected():
    support = _support_paths()
    proc = _run_entrypoint(_ENTRYPOINT_UNKNOWN_RUNNER, *support)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    combined = proc.stdout + proc.stderr
    assert "PYTEST_EXIT 1" in combined
    assert "case selection failed" in combined
    assert "missing::case::id" in combined
