# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Keep FP8 parameter outcomes visible alongside the shared benchmark metrics."""

import pytest

from .. import conftest as shared


@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report):
    if (
        shared.Config is None
        or not shared.Config.record_json
        or getattr(report, "opid", None) != "fp8_gemm"
    ):
        return
    # The shared hook records the last parameter's result. For FP8, retain all
    # parameters so the four expected upstream skips cannot overwrite passes.
    if report.when == "call" or (
        report.when in ("setup", "teardown") and report.outcome != "passed"
    ):
        result = shared.TEST_RESULTS.setdefault("fp8_gemm", {})
        cases = result.setdefault("pytest_cases", {})
        previous = cases.get(report.nodeid)
        if previous is None or previous["result"] != "failed":
            cases[report.nodeid] = {
                "result": report.outcome,
                "reason": (
                    shared.get_reason(report) if report.outcome != "passed" else None
                ),
            }
        counts = {
            status: sum(case["result"] == status for case in cases.values())
            for status in ("passed", "skipped", "failed")
        }
        result["pytest_summary"] = counts
        result["result"] = (
            "failed"
            if counts["failed"]
            else "passed" if counts["passed"] else "skipped"
        )
        result["reason"] = "See pytest_cases for failures" if counts["failed"] else None


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    config = shared.Config
    if config is None or config.case_ids is None:
        return
    # Explicitly skipped FP8 IDs were found and handled. Exclude only those
    # verified IDs from the shared 'requested but not executed' error check;
    # unknown IDs still fail, and executed_case_ids remains measurements only.
    skipped = getattr(config, "fp8_skipped_case_ids", set())
    config.case_ids = [case_id for case_id in config.case_ids if case_id not in skipped]
