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

"""Tests for the ported "--base-data" lookup and scaling helpers."""

import pytest
import torch

import benchmark.consts as consts

LAMB_SHAPE = [
    torch.Size([1024]),
    torch.Size([0]),
    torch.Size([1024]),
    torch.Size([1024]),
    torch.Size([1024]),
]


def _lamb_config():
    return {
        "inputs": {
            "p": {"shape": [1024], "dtype": "torch.float32"},
            "p_copy": {"shape": [0], "dtype": "torch.float32"},
            "m": {"shape": [1024], "dtype": "torch.float32"},
            "v": {"shape": [1024], "dtype": "torch.float32"},
            "g": {"shape": [1024], "dtype": "torch.float32"},
            "lr": {"scalar": 1e-3},
        }
    }


def _base_data(shape_key="[1024]"):
    return {
        "_scaling_factors": {
            "reference_chip": "H800",
            "chips": [
                {
                    "vendor": "nvidia",
                    "chip": "H800",
                    "measured": {
                        "bandwidth_gbps": 1.0,
                        "tensor": {"fp16": 1.0},
                        "vector": {"fp32": 1.0},
                    },
                    "nominal": {
                        "bandwidth_gbps": 1.0,
                        "tensor": {"fp16": 1.0},
                        "vector": {"fp32": 1.0},
                    },
                },
                {
                    "vendor": "nvidia",
                    "chip": "H100",
                    "measured": {
                        "bandwidth_gbps": 1.0,
                        "tensor": {"fp16": 1.04},
                        "vector": {"fp32": 0.99},
                    },
                    "nominal": {
                        "bandwidth_gbps": 1.0,
                        "tensor": {"fp16": 1.0},
                        "vector": {"fp32": 1.0},
                    },
                },
                {
                    "vendor": "hygon",
                    "chip": "BW1000",
                    "measured": {
                        "bandwidth_gbps": 0.47,
                        "tensor": {"fp16": 0.52},
                        "vector": {"fp32": 0.23},
                    },
                    "nominal": {
                        "bandwidth_gbps": 0.53,
                        "tensor": {"fp16": 0.60},
                        "vector": {"fp32": 0.25},
                    },
                },
            ],
        },
        "lamb": {
            "native_api": "ops.lamb",
            "shapes": {
                shape_key: {
                    "torch.float32": {
                        "config": _lamb_config(),
                        "latency_ms": 0.02,
                        "bottle_neck_unit": "mem",
                    },
                    "torch.bfloat16": {
                        "config": _lamb_config(),
                        "latency_ms": 0.02,
                        "bottle_neck_unit": "mem",
                    },
                }
            },
        },
    }


@pytest.fixture(autouse=True)
def _clear_signature_cache():
    consts._SIG_INDEX_CACHE.clear()
    yield
    consts._SIG_INDEX_CACHE.clear()


def test_direct_match_by_first_tensor_shape():
    record, miss = consts.lookup_base_record_ex(
        _base_data(), "lamb", torch.float32, LAMB_SHAPE
    )

    assert miss is None
    assert record["latency_ms"] == 0.02


def test_signature_fallback_when_shape_key_is_not_the_first_tensor():
    record, miss = consts.lookup_base_record_ex(
        _base_data(shape_key="semantic_lamb_key"),
        "lamb",
        torch.float32,
        LAMB_SHAPE,
    )

    assert miss is None
    assert record["latency_ms"] == 0.02


def test_missing_operator_reports_reason():
    record, miss = consts.lookup_base_record_ex(
        _base_data(), "not_an_op", torch.float32, LAMB_SHAPE
    )

    assert record is None
    assert "no operator" in miss


def test_missing_shape_reports_reason():
    shape = [torch.Size([777]), torch.Size([0])]
    record, miss = consts.lookup_base_record_ex(
        _base_data(), "lamb", torch.float32, shape
    )

    assert record is None
    assert "no matching shape" in miss


def test_missing_dtype_reports_reason():
    record, miss = consts.lookup_base_record_ex(
        _base_data(), "lamb", torch.float16, LAMB_SHAPE
    )

    assert record is None
    assert "has no dtype" in miss


def test_scaling_factor_prefers_measured():
    factor = consts.lookup_scaling_factor(
        _base_data(), "hygon", "mem", torch.float32, self_chip="Hygon BW1000"
    )

    assert factor == 0.47


def test_scaling_factor_maps_bottleneck_and_dtype():
    base = _base_data()

    assert (
        consts.lookup_scaling_factor(
            base, "hygon", "tensor", torch.float16, self_chip="Hygon BW1000"
        )
        == 0.52
    )
    assert (
        consts.lookup_scaling_factor(
            base, "hygon", "cuda", torch.float32, self_chip="Hygon BW1000"
        )
        == 0.23
    )


def test_scaling_factor_single_chip_vendor_falls_back():
    factor = consts.lookup_scaling_factor(
        _base_data(), "hygon", "mem", torch.float32, self_chip="Some Other Chip"
    )

    assert factor == 0.47


def test_scaling_factor_returns_none_for_unknown_chip_when_vendor_has_several():
    factor = consts.lookup_scaling_factor(
        _base_data(),
        "nvidia",
        "mem",
        torch.float32,
        self_chip="NVIDIA A100-SXM4-40GB",
    )

    assert factor is None


def test_scaling_factor_returns_none_without_bottleneck():
    factor = consts.lookup_scaling_factor(
        _base_data(), "hygon", None, torch.float32, self_chip="Hygon BW1000"
    )

    assert factor is None


def test_first_tensor_shape_walks_nested_args_and_kwargs():
    shape_detail = (
        [torch.Size([8, 16]), torch.Size([16])],
        {"scale": torch.Size([1]), "eps": 1e-6},
    )

    assert consts._first_tensor_shape(shape_detail) == [8, 16]


def test_base_miss_reason_is_not_a_selectable_metric():
    assert "base_miss_reason" not in consts.ALL_AVAILABLE_METRICS
