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

"""Opt-in support for running an explicit reference as the candidate.

Correctness tests opt in by calling :func:`get_candidate` with the reference
they already trust. Benchmarks opt in through their shared timing path, which
selects the benchmark's explicit ``torch_op``. Both paths call the single
:func:`select_candidate`, so there is one definition of what
``--ref-as-candidate`` changes.

This module is imported only by the pytest/benchmark integration. Importing
``flag_train`` itself must not require pytest.
"""

import importlib
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path

import pytest

_MODE = ContextVar("flag_train_reference_candidate_mode", default=None)

# Public package roots accepted in an operator path. A bare operator name is
# deliberately not accepted: it would be a second, ambiguous lookup policy.
_OPERATOR_ROOTS = ("megatron", "deepspeed", "transformer_engine")


class ReferenceContractError(ValueError):
    """A selected test did not supply or execute an explicit reference."""

    def __init__(self, message):
        super().__init__(message)
        state = _MODE.get()
        if state is not None and state.enabled:
            # Errors that a test catches must still fail the selected node.
            state.contract_errors.append(str(message))


@dataclass
class _Execution:
    enabled: bool
    nodeid: str
    reference_calls: int = 0
    candidate_calls: int = 0
    candidate_derivative_calls: int = 0
    reference_selected: bool = False
    completed_timings: int = 0
    selection_empty: bool = False
    contract_errors: list = field(default_factory=list)
    reported_contract_errors: int = 0


@contextmanager
def reference_test_mode(enabled, nodeid=""):
    """Scope the candidate choice and call counters to one pytest case."""
    state = _Execution(bool(enabled), nodeid)
    token = _MODE.set(state)
    try:
        yield state
    finally:
        _MODE.reset(token)


def reference_mode_enabled():
    state = _MODE.get()
    return state is not None and state.enabled


def reference_execution():
    """Return the current execution state for the benchmark integration."""
    return _MODE.get()


def select_candidate(candidate, reference, label=""):
    """Select the callable to run, observing reference calls in opt-in mode.

    Without ``--ref-as-candidate`` this is the identity function. With the flag
    the explicit reference wins over ``candidate``; a missing or non-callable
    reference is a contract error rather than a silent fallback.
    """
    state = _MODE.get()
    if state is None or not state.enabled:
        return candidate
    if not callable(reference):
        raise ReferenceContractError(
            f"{label or state.nodeid}: No predefined reference; explicitly "
            "define a callable reference before --ref-as-candidate."
        )
    state.reference_selected = True

    @wraps(reference)
    def invoke(*args, **kwargs):
        state.candidate_calls += 1
        state.reference_calls += 1
        return reference(*args, **kwargs)

    return invoke


def _qualified_parts(operator_path):
    parts = tuple(part for part in operator_path.split(".") if part)
    if not parts:
        raise ValueError("Operator path must not be empty.")
    if parts[0] == "flag_train":
        qualified = parts
    elif parts[0] in _OPERATOR_ROOTS:
        qualified = ("flag_train",) + parts
    else:
        raise ValueError(
            f"Operator path '{operator_path}' must be qualified as "
            "'flag_train.<component>.<name>' or '<component>.<name>'."
        )
    if len(qualified) < 3:
        raise ValueError(
            f"Operator path '{operator_path}' must name a public callable, "
            "for example 'flag_train.megatron.fast_gelu'."
        )
    return qualified


def _resolve_public_candidate(operator_path):
    """Read the live public attribute, including CLI override patches.

    Importing the package returns the already-initialized module, so an
    override installed on that module is observed. Implementation files are
    never imported directly.
    """
    parts = _qualified_parts(operator_path)
    module = importlib.import_module(parts[0])
    for part in parts[1:-1]:
        module = getattr(module, part)
    try:
        return getattr(module, parts[-1])
    except AttributeError as error:
        raise ImportError(
            f"Operator '{operator_path}' has no public candidate in flag_train."
        ) from error


def get_candidate(operator_path, *, reference=None):
    """Return the candidate, or the explicit reference in reference mode.

    In reference mode the reference is selected before any lookup, so a test
    works even when the candidate export is absent. In the default mode the
    live public attribute is returned, including any CLI override.
    """
    if reference_mode_enabled():
        return select_candidate(None, reference, operator_path)
    candidate = _resolve_public_candidate(operator_path)
    return select_candidate(candidate, reference, operator_path)


# Options that would ask the run to validate something other than a selected
# reference. They are checked only when the option is actually registered.
_CONFLICT_OPTIONS = (
    "override",
    "override_config",
    "list_overrides",
    "dummy_candidate",
    "dummy_report",
    "reference_only",
    "preflight_only",
    "profile_only",
    "list_cases",
    "query",
    "collect_marks",
    "collectonly",
    "parallel",
    "numprocesses",
    "dist",
)


def _option_flag(name):
    if name == "collectonly":
        return "--collect-only"
    return "--" + name.replace("_", "-")


def _reject_conflicts(config):
    for name in _CONFLICT_OPTIONS:
        value = getattr(config.option, name, None)
        if not value:
            continue
        if name == "dist" and value == "no":
            continue
        raise pytest.UsageError(
            f"--ref-as-candidate cannot be combined with {_option_flag(name)}."
        )


def add_reference_candidate_argument(parser):
    """Register the shared opt-in flag.

    A mixed tests+benchmark run loads both conftests, so a duplicate
    registration is expected and harmless.
    """
    try:
        parser.addoption(
            "--ref-as-candidate",
            action="store_true",
            default=False,
            help=(
                "Run the explicit reference supplied by each correctness test "
                "as the candidate, and time the benchmark's explicit torch_op "
                "as the candidate. Requires an explicit reference."
            ),
        )
    except ValueError:
        pass


def configure_reference_candidate(config):
    """Register the shared lifecycle when the opt-in flag is present."""
    if not getattr(config.option, "ref_as_candidate", False):
        return
    if getattr(config, "_reference_candidate_plugin", None) is not None:
        return
    _reject_conflicts(config)
    plugin = _ReferenceCandidatePlugin()
    config._reference_candidate_plugin = plugin
    config.pluginmanager.register(plugin, "flag-train-reference-candidate")


class _ReferenceCandidatePlugin:
    """Per-case lifecycle shared by correctness and benchmark nodes."""

    def __init__(self):
        self.records = {}
        self.collection_failed = False

    @staticmethod
    def _is_correctness(item):
        return Path(item.nodeid.split("::", 1)[0]).parts[0] != "benchmark"

    def pytest_collectreport(self, report):
        self.collection_failed = self.collection_failed or report.failed

    def pytest_collection_finish(self, session):
        for item in session.items:
            self.records[item.nodeid] = {"status": "NOT_RUN", "calls": 0}

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        with reference_test_mode(True, item.nodeid) as state:
            item._reference_execution = state
            yield

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_call(self, item):
        state = item._reference_execution
        # Fixture calls happen before this hook and must not satisfy the test.
        state.reference_calls = 0
        state.candidate_calls = 0
        state.candidate_derivative_calls = 0
        state.completed_timings = 0
        state.selection_empty = False
        outcome = yield
        interrupted = outcome.excinfo and issubclass(
            outcome.excinfo[0],
            (KeyboardInterrupt, SystemExit, pytest.exit.Exception),
        )
        if interrupted:
            return
        if state.contract_errors:
            if outcome.excinfo is None or not isinstance(
                outcome.excinfo[1], ReferenceContractError
            ):
                outcome.force_exception(
                    ReferenceContractError(state.contract_errors[0])
                )
            return
        if outcome.excinfo is not None:
            return
        if self._is_correctness(item):
            if state.candidate_calls == 0:
                if not state.reference_selected:
                    message = (
                        f"{item.nodeid}: No predefined reference selected; "
                        "explicitly pass a callable reference to get_candidate."
                    )
                else:
                    message = (
                        f"{item.nodeid}: explicit reference contract did not "
                        "complete a candidate-side call."
                    )
                outcome.force_exception(ReferenceContractError(message))
            return
        # Benchmark nodes that selected no case have nothing to time.
        if state.completed_timings == 0 and not (
            state.selection_empty and state.candidate_calls == 0
        ):
            outcome.force_exception(
                ReferenceContractError(
                    f"{item.nodeid}: explicit reference contract did not "
                    "complete a candidate-side timing."
                )
            )

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        state = getattr(item, "_reference_execution", None)
        if state is None:
            return
        record = self.records.setdefault(
            item.nodeid, {"status": "NOT_RUN", "calls": 0}
        )
        correctness = self._is_correctness(item)

        if (
            correctness
            and report.skipped
            and not state.reference_selected
            and not state.contract_errors
        ):
            # A skip (including a fixture skip or xfail) must not hide a test
            # that never declared its reference.
            ReferenceContractError(
                f"{item.nodeid}: No predefined reference selected; explicitly "
                "pass a callable reference to get_candidate."
            )

        if len(state.contract_errors) > state.reported_contract_errors:
            state.reported_contract_errors = len(state.contract_errors)
            report.outcome = "failed"
            report.longrepr = "ReferenceContractError: " + state.contract_errors[0]
            if hasattr(report, "wasxfail"):
                del report.wasxfail
            record["status"] = "FAILED"
        elif report.failed:
            record["status"] = "FAILED"
        elif report.skipped and record["status"] != "FAILED":
            record["status"] = "SKIP"
        elif (
            call.when == "call"
            and report.passed
            and record["status"] != "FAILED"
        ):
            record.update(
                status=(
                    "NOT_SELECTED"
                    if state.selection_empty and not state.candidate_calls
                    else "PASSED"
                ),
                calls=state.candidate_calls,
            )

        if call.when == "call" or state.contract_errors:
            report.reference_candidate = {
                "candidate_source": "reference",
                "reference_calls": state.reference_calls,
                "reference_candidate_calls": state.candidate_calls,
            }

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_sessionfinish(self, session, exitstatus):
        yield
        records = list(self.records.values())
        supported = any(
            record["status"] == "PASSED" and record["calls"] > 0
            for record in records
        )
        complete = all(
            record["status"] in {"PASSED", "SKIP", "NOT_SELECTED"}
            for record in records
        )
        if (
            self.collection_failed or not supported or not complete
        ) and session.exitstatus in (0, 5):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
