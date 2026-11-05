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

"""Regression tests for CLI overrides of operator package exports."""

from types import SimpleNamespace

import pytest

import flag_train
from flag_train.cli_override import apply_overrides_from_args
from flag_train.dynamic_registry import DynamicOpOverride, OverrideResolutionError


def _export(monkeypatch, package):
    original = lambda: "original"
    monkeypatch.setattr(package, "override_probe", original, raising=False)
    monkeypatch.setattr(package, "__all__", [*package.__all__, "override_probe"])
    return original


@pytest.mark.parametrize(
    "name",
    ("override_probe", "megatron.override_probe", "flag_train.megatron.override_probe"),
)
def test_override_replaces_and_restores_package_export(monkeypatch, name):
    package = flag_train.megatron
    original = _export(monkeypatch, package)
    registry = DynamicOpOverride()
    assert registry.override(name, lambda: "candidate")
    try:
        assert package.override_probe() == "candidate"
        assert not hasattr(flag_train, "override_probe")
    finally:
        assert registry.restore(name)
    assert package.override_probe is original


def test_override_rejects_ambiguous_bare_name(monkeypatch):
    _export(monkeypatch, flag_train.megatron)
    _export(monkeypatch, flag_train.deepspeed)
    with pytest.raises(OverrideResolutionError, match="ambiguous"):
        DynamicOpOverride().resolve_target("override_probe")


def test_cli_override_rolls_back_after_invalid_target(monkeypatch, tmp_path):
    package = flag_train.megatron
    original = _export(monkeypatch, package)
    path = tmp_path / "candidate.py"
    path.write_text('def override_probe(): return "candidate"\n', encoding="utf-8")
    registry = DynamicOpOverride()
    args = SimpleNamespace(
        override=[f"megatron.override_probe={path}", f"outside.override_probe={path}"]
    )

    with pytest.raises(SystemExit):
        apply_overrides_from_args(args, registry=registry)

    assert package.override_probe is original
    assert registry.list_overrides() == []
