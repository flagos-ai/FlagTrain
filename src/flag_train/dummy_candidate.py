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

"""Report whether selected tests and benchmark cases reach an injected candidate."""

import json
from contextvars import ContextVar
from pathlib import Path

import pytest

from .dynamic_registry import DynamicOpOverride, OverrideResolutionError

_ACTIVE_PROBE = ContextVar("flagtrain_dummy_probe", default=None)


class _DummyReached(Exception):
    """A signal raised by the injected callable after recording its invocation."""


def get_dummy_probe():
    return _ACTIVE_PROBE.get()


def add_dummy_candidate_arguments(parser):
    for flag, help_text in (
        ("--dummy-candidate", "Check candidate invocation for this operator."),
        ("--dummy-report", "Path to the dummy candidate JSON report."),
    ):
        try:
            parser.addoption(flag, help=help_text)
        except ValueError:
            # Both conftests can be loaded in one invocation.
            pass


def configure_dummy_candidate(config):
    operator = config.getoption("dummy_candidate")
    output = config.getoption("dummy_report")
    if not operator:
        if output:
            raise pytest.UsageError("--dummy-report requires --dummy-candidate.")
        return
    if config.pluginmanager.has_plugin("flagtrain-dummy-candidate"):
        return
    for option in (
        "override",
        "override_config",
        "ref_as_candidate",
        "list_cases",
        "query",
        "collect_marks",
        "collectonly",
        "profile_only",
        "parallel",
        "numprocesses",
    ):
        if getattr(config.option, option, None):
            flag = (
                "--collect-only"
                if option == "collectonly"
                else "--" + option.replace("_", "-")
            )
            raise pytest.UsageError(
                f"--dummy-candidate cannot be combined with {flag}."
            )

    registry = DynamicOpOverride()
    try:
        target = registry.resolve_target(operator)
    except OverrideResolutionError as exc:
        raise pytest.UsageError(str(exc)) from exc
    probe = _DummyProbe(
        operator, target.attr_name, output or "dummy-candidate-report.json"
    )
    if not registry.override(operator, probe.candidate):
        raise pytest.UsageError(f"Could not inject dummy candidate for {operator}.")
    config.add_cleanup(registry.restore_all)
    config.pluginmanager.register(probe, "flagtrain-dummy-candidate")


class _DummyProbe:
    def __init__(self, operator, name, output):
        self.operator = operator
        self.name = name
        self.output = Path(output)
        self.records = []
        self.collection_errors = []
        self.case_ids = set()
        self.node = None
        self.observed = None
        self.invoking = False
        self.signal = None
        self.traversed = False
        self.reported_unexpected_calls = 0

    def _record(self, nodeid, phase, case_id=None):
        record = {
            "operator": self.operator,
            "phase": phase,
            "nodeid": nodeid,
            "status": "NOT_RUN",
            "calls": 0,
            "before_candidate_calls": 0,
            "unexpected_calls": 0,
        }
        if case_id is not None:
            record["case_id"] = case_id
        self.records.append(record)
        return record

    def candidate(self, *args, **kwargs):
        signal = _DummyReached("DUMMY_CANDIDATE_REACHED")
        record = self.observed
        if record is None:
            record = self._record("<session>", "session")
        if self.invoking:
            record["calls"] += 1
            self.signal = signal
        else:
            record["unexpected_calls"] += 1
            if not record["calls"]:
                record["before_candidate_calls"] += 1
            record.update(
                status="PRE_CALL_ERROR",
                message="Dummy candidate called outside the selected invocation.",
            )
        raise signal

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors.append(str(report.longrepr)[-2000:])

    def pytest_collection_finish(self, session):
        self.nodes = {
            item.nodeid: self._record(
                item.nodeid,
                (
                    "benchmark"
                    if Path(item.nodeid).parts[0] == "benchmark"
                    else "correctness"
                ),
            )
            for item in session.items
        }

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        token = _ACTIVE_PROBE.set(self)
        self.node = self.observed = self.nodes[item.nodeid]
        self.traversed = False
        self.reported_unexpected_calls = 0
        try:
            yield
        finally:
            _ACTIVE_PROBE.reset(token)
            self.node = self.observed = self.signal = None

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_call(self, item):
        self.invoking = self.node["phase"] == "correctness"
        try:
            yield
        finally:
            self.invoking = False

    @staticmethod
    def _fail(report, record, status, message):
        record.update(status=status, message=message)
        report.outcome = "failed"
        report.longrepr = message
        if hasattr(report, "wasxfail"):
            del report.wasxfail

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        record = self.node
        if record["unexpected_calls"] > self.reported_unexpected_calls:
            self.reported_unexpected_calls = record["unexpected_calls"]
            self._fail(report, record, record["status"], record["message"])
        elif call.when != "call":
            if report.failed or (
                report.skipped and (record["calls"] or self.traversed)
            ):
                self._fail(
                    report, record, "PYTEST_LIFECYCLE_ERROR", str(report.longrepr)
                )
            elif report.skipped:
                record.update(status="SKIP", message=str(report.longrepr))
        elif self.traversed:
            if report.failed or report.skipped:
                self._fail(report, record, "POST_CALL_ERROR", str(report.longrepr))
            elif record["status"] == "NOT_RUN":
                record["status"] = "TRAVERSED"
        elif record["calls"]:
            if report.failed and call.excinfo and call.excinfo.value is self.signal:
                record["status"] = "REACHED"
                report.outcome = "passed"
                report.longrepr = None
            else:
                self._fail(
                    report,
                    record,
                    "POST_CALL_ERROR",
                    "Dummy candidate signal did not reach pytest.",
                )
        elif report.skipped:
            record.update(status="SKIP", message=str(report.longrepr))
        elif report.failed:
            record.update(status="PRE_CALL_ERROR", message=str(report.longrepr))
        else:
            self._fail(
                report,
                record,
                "BYPASSED",
                "Selected test did not invoke the dummy candidate.",
            )

    def run_benchmark(self, bench, selected, config):
        self.traversed = True
        names = (
            {self.name, f"{self.name}_backward"} if bench.is_backward else {self.name}
        )
        if (
            bench.op_name not in names
            or not bench.supports_cases()
            or not callable(bench.train_op)
        ):
            pytest.fail(
                "Dummy probing requires the selected operator, stable cases and an explicit benchmark candidate."
            )
        cases = bench._collect_cases()
        if not cases:
            pytest.fail(f"No benchmark cases for {bench.op_name}.")
        config.available_case_ids.update(case.case_id for case in cases)
        for case in cases:
            if selected is not None and case.case_id not in selected:
                continue
            record = self._record(self.node["nodeid"], "benchmark", case.case_id)
            if case.case_id in self.case_ids:
                record.update(
                    status="DUPLICATE_CASE", message="Case ID executed more than once."
                )
                continue
            self.case_ids.add(case.case_id)
            self._run_case(bench, case, record)
            if record["status"] in {"REACHED", "SKIP"}:
                config.executed_case_ids.add(case.case_id)

    def _run_case(self, bench, case, record):
        self.observed = record
        try:
            args, kwargs = bench.unpack_to_args_kwargs(bench.build_inputs(case))
            if record["unexpected_calls"]:
                return
            self.invoking = True
            bench.train_op(*args, **kwargs)
        except pytest.skip.Exception as exc:
            if not record["unexpected_calls"]:
                status = "POST_CALL_ERROR" if self.invoking else "SKIP"
                record.update(status=status, message=str(exc))
        except (Exception, pytest.fail.Exception, pytest.xfail.Exception) as exc:
            if record["calls"] and exc is self.signal:
                record["status"] = "REACHED"
            elif not record["unexpected_calls"]:
                status = "POST_CALL_ERROR" if self.invoking else "PRE_CALL_ERROR"
                record.update(status=status, message=str(exc))
        else:
            status = "POST_CALL_ERROR" if record["calls"] else "BYPASSED"
            record.update(
                status=status,
                message="Dummy candidate signal did not reach the runner.",
            )
        finally:
            self.invoking = False
            self.observed = self.node
            self.signal = None

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_sessionfinish(self, session, exitstatus):
        yield
        records = [record for record in self.records if record["status"] != "TRAVERSED"]
        statuses = {record["status"] for record in records}
        if (
            session.exitstatus == 0
            and statuses <= {"REACHED", "SKIP"}
            and "REACHED" in statuses
        ):
            status = "PASSED"
        elif not statuses and session.exitstatus in (0, 5):
            status = "NO_CASES"
        elif statuses == {"SKIP"} and session.exitstatus == 0:
            status = "ALL_SKIP"
        else:
            status = "FAILED"
        if status != "PASSED" and session.exitstatus in (0, 5):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter:
            reporter.write_line(f"FlagTrain dummy candidate {status}: {self.output}")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(
            json.dumps(
                {
                    "schema_version": "flaggems.dummy-candidate/v1",
                    "operator": self.operator,
                    "status": status,
                    "proves": "candidate_invocation_only",
                    "records": records,
                    "collection_errors": self.collection_errors,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
