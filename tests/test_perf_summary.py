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

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import perf_summary  # noqa: E402


def row(shape, speedup, compared=None, miss=None):
    return {
        "shape_detail": [shape],
        "latency_base": 0.02,
        "latency": 0.01,
        "speedup": speedup,
        "compared_speedup": compared,
        "base_miss_reason": miss,
    }


def raw(rows, dtype="torch.float32", status="passed"):
    return {
        "lamb": {
            "result": status,
            "details": [{"dtype": dtype, "op_name": "lamb", "result": rows}],
        }
    }


def test_compared_speedup_overrides_local_speedup():
    parsed = perf_summary.parse_benchmark_data(
        raw([row([1024], 1.0, 2.0), row([4096], 3.0, 4.0)])
    )
    record = parsed["lamb"]

    assert record["data"]["fp32"]["speedup"] == pytest.approx(3.0)
    assert record["perf_avg"] == pytest.approx(3.0)


def test_none_is_excluded_and_zero_is_kept():
    parsed = perf_summary.parse_benchmark_data(
        raw([row([1], None), row([2], 0.0), row([3], 2.0)])
    )
    record = parsed["lamb"]

    # 0.0 is numeric and counted; None is not.
    assert record["data"]["fp32"]["speedup"] == pytest.approx(1.0)


def test_per_op_average_is_over_dtype_averages():
    payload = {
        "lamb": {
            "result": "passed",
            "details": [
                {"dtype": "torch.float16", "result": [row([1], 2.0)]},
                {"dtype": "torch.bfloat16", "result": [row([2], 4.0)]},
            ],
        }
    }
    record = perf_summary.parse_benchmark_data(payload)["lamb"]

    assert record["data"]["fp16"]["speedup"] == pytest.approx(2.0)
    assert record["data"]["bf16"]["speedup"] == pytest.approx(4.0)
    assert record["perf_avg"] == pytest.approx(3.0)


def test_misses_are_counted_and_reasons_deduplicated():
    parsed = perf_summary.parse_benchmark_data(
        raw(
            [
                row([1], 1.0, 1.1),
                row([2], 1.0, 1.2),
                row([3], 1.0, miss="no shape"),
                row([4], 1.0, miss="no shape"),
            ]
        )
    )
    record = parsed["lamb"]

    assert record["base_data_used"] is True
    assert record["matched_shapes"] == 2
    assert record["missed_shapes"] == 2
    assert record["base_miss_reasons"] == ["no shape"]


def test_without_base_data_matched_counts_are_unknown():
    record = perf_summary.parse_benchmark_data(raw([row([1], 1.5)]))["lamb"]

    assert record["base_data_used"] is False
    assert record["matched_shapes"] is None
    assert record["missed_shapes"] is None
    assert record["base_miss_reasons"] == []


def test_failed_status_has_no_performance_data():
    record = perf_summary.parse_benchmark_data(
        {"lamb": {"result": "failed", "reason": "boom"}}
    )["lamb"]

    assert record["status"] == "Failed"
    assert record["reason"] == "boom"
    assert record["perf_avg"] is None
    assert record["data"] == {}


def test_display_dtype():
    assert perf_summary.display_dtype("torch.float32") == "fp32"
    assert perf_summary.display_dtype("torch.bfloat16") == "bf16"
    assert perf_summary.display_dtype("torch.float8_e4m3fn") == "fp8_e4m3"


def test_summarize_and_write_outputs(tmp_path):
    parsed = [
        perf_summary.parse_benchmark_data(
            raw([row([1024], 1.0, 1.25), row([4096], 1.0, 1.5)])
        )
    ]
    summary = perf_summary.summarize(parsed, [tmp_path / "benchmark_lamb.json"])
    columns = perf_summary.dtype_columns(summary)

    json_path = tmp_path / "summary.json"
    csv_path = tmp_path / "summary.csv"
    md_path = tmp_path / "summary.md"
    perf_summary.write_json(json_path, summary)
    perf_summary.write_csv(csv_path, summary, columns)
    perf_summary.write_markdown(md_path, summary, columns)

    loaded = json.loads(json_path.read_text())
    perf = loaded["result"]["lamb"]["performance"]
    assert perf["perf_avg"] == pytest.approx(1.375)
    assert perf["matched_shapes"] == 2
    assert "lamb" in csv_path.read_text()
    assert "Performance summary" in md_path.read_text()
    assert columns == ["fp32"]


def test_gate_reports_below_threshold_and_missing_values():
    parsed = [
        perf_summary.parse_benchmark_data(
            raw([row([1], 0.7, 0.7), row([2], 0.8, 0.8)])
        ),
        perf_summary.parse_benchmark_data({"other": {"result": "failed"}}),
    ]
    summary = perf_summary.summarize(parsed, [])

    failures = perf_summary.gate_failures(summary, 0.8)

    assert [entry[0] for entry in failures] == ["lamb", "other"]
    assert failures[0][1] == pytest.approx(0.75)
    assert failures[1][1] is None


def test_gate_passes_above_threshold():
    parsed = [perf_summary.parse_benchmark_data(raw([row([1], 0.9, 0.95)]))]
    summary = perf_summary.summarize(parsed, [])

    assert perf_summary.gate_failures(summary, 0.8) == []


def test_gate_at_threshold_passes():
    parsed = [perf_summary.parse_benchmark_data(raw([row([1], 0.8, 0.8)]))]
    summary = perf_summary.summarize(parsed, [])

    assert perf_summary.gate_failures(summary, 0.8) == []


def test_collect_inputs_directory_uses_benchmark_json(tmp_path):
    (tmp_path / "benchmark_lamb.json").write_text("{}")
    (tmp_path / "accuracy_lamb.json").write_text("{}")

    files = perf_summary.collect_inputs([str(tmp_path)])

    assert [path.name for path in files] == ["benchmark_lamb.json"]
