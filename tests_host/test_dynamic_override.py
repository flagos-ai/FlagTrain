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

"""Host-only tests for dynamic operator override target resolution.

The shared modules are imported from the isolated synthetic package created
by ``tests_host/conftest.py``; importing this module must not touch a real or
sentinel ``flag_train`` package.
"""

import runpy
import sys
import types
from pathlib import Path

import pytest
import yaml

from _flagtrain_override_host.cli_override import (
    apply_overrides_from_args,
    load_override_config,
    parse_override_spec,
)
from _flagtrain_override_host.dynamic_registry import (
    DynamicOpOverride,
    OverrideResolutionError,
    default_impl_name,
    resolve_operator_target,
)

_HOST_DIR = Path(__file__).resolve().parent


def _original(value):
    def impl():
        return value

    impl.__name__ = "original_impl"
    return impl


def _candidate(value):
    def impl():
        return value

    impl.__name__ = "candidate_impl"
    return impl


def _write_candidate(tmp_path, func_name, value="from-file"):
    path = tmp_path / "{}.py".format(func_name)
    path.write_text(
        "def {name}():\n    return {value!r}\n".format(name=func_name, value=value)
    )
    return str(path)


def _args(override=None, override_config=None):
    return types.SimpleNamespace(
        override=override if override is not None else [],
        override_config=override_config,
    )


def _add_export(tree, package_name, op_name, value):
    package = tree.package(package_name)
    impl = _original(value)
    setattr(package, op_name, impl)
    package.__all__.append(op_name)
    return impl


def test_host_suite_import_preserves_existing_flag_train_modules():
    """Collection must not replace a real/sentinel ``flag_train`` tree."""
    sentinel_root = types.ModuleType("flag_train")
    sentinel_root.existing_operator = object()
    sentinel_child = types.ModuleType("flag_train.megatron")
    sentinel_root.megatron = sentinel_child

    sys.modules["flag_train"] = sentinel_root
    sys.modules["flag_train.megatron"] = sentinel_child
    try:
        runpy.run_path(
            str(_HOST_DIR / "conftest.py"), run_name="_host_conftest_probe"
        )
        runpy.run_path(
            str(_HOST_DIR / "test_dynamic_override.py"),
            run_name="_host_tests_probe",
        )

        assert sys.modules["flag_train"] is sentinel_root
        assert sys.modules["flag_train.megatron"] is sentinel_child
        assert sentinel_root.megatron is sentinel_child
        assert "existing_operator" in vars(sentinel_root)
    finally:
        sys.modules.pop("flag_train", None)
        sys.modules.pop("flag_train.megatron", None)


@pytest.mark.parametrize(
    "spec",
    [
        "megatron.fused_proj_rms_backward",
        "flag_train.megatron.fused_proj_rms_backward",
    ],
)
def test_explicit_qualified_paths_register_call_and_restore(operator_tree, spec):
    original = _add_export(operator_tree, "megatron", "fused_proj_rms_backward", "orig")
    candidate = _candidate("new")
    registry = DynamicOpOverride()

    assert registry.override(spec, candidate)
    assert operator_tree.package("megatron").fused_proj_rms_backward is candidate
    assert operator_tree.package("megatron").fused_proj_rms_backward() == "new"
    assert registry.list_overrides() == [
        "flag_train.megatron.fused_proj_rms_backward"
    ]

    assert registry.restore(spec)
    assert operator_tree.package("megatron").fused_proj_rms_backward is original
    assert registry.list_overrides() == []


def test_bare_name_resolves_unique_package_export(operator_tree):
    original = _add_export(operator_tree, "deepspeed", "blocked_flash", "orig")
    candidate = _candidate("new")
    registry = DynamicOpOverride()

    assert registry.override("blocked_flash", candidate)
    assert operator_tree.package("deepspeed").blocked_flash is candidate
    assert not hasattr(operator_tree.root, "blocked_flash")

    assert registry.restore("blocked_flash")
    assert operator_tree.package("deepspeed").blocked_flash is original


def test_bare_name_preserves_existing_top_level_operator(operator_tree):
    root_original = _original("root")
    operator_tree.root.softmax = root_original
    package_original = _add_export(operator_tree, "megatron", "softmax", "package")
    candidate = _candidate("new")
    registry = DynamicOpOverride()

    assert registry.override("softmax", candidate)
    assert operator_tree.root.softmax is candidate
    assert operator_tree.package("megatron").softmax is package_original

    assert registry.restore("softmax")
    assert operator_tree.root.softmax is root_original
    assert operator_tree.package("megatron").softmax is package_original


def test_explicit_flag_train_top_level_path(operator_tree):
    original = _original("orig")
    operator_tree.root.top_level_op = original
    candidate = _candidate("new")
    registry = DynamicOpOverride()

    assert registry.override("flag_train.top_level_op", candidate)
    assert operator_tree.root.top_level_op is candidate
    assert registry.list_overrides() == ["flag_train.top_level_op"]

    assert registry.restore("flag_train.top_level_op")
    assert operator_tree.root.top_level_op is original


def test_root_original_none_is_preserved_exactly(operator_tree):
    operator_tree.root.none_op = None
    candidate = _candidate("new")
    registry = DynamicOpOverride()

    assert registry.override("none_op", candidate)
    assert operator_tree.root.none_op is candidate

    assert registry.restore("none_op")
    assert "none_op" in vars(operator_tree.root)
    assert operator_tree.root.none_op is None


def test_missing_qualified_attribute_is_injected_and_removed_cleanly(operator_tree):
    untouched = _add_export(operator_tree, "deepspeed", "packbits", "orig")
    candidate = _candidate("temporary")
    registry = DynamicOpOverride()

    assert registry.override("deepspeed.brand_new_op", candidate)
    assert operator_tree.package("deepspeed").brand_new_op is candidate
    assert registry.list_overrides() == ["flag_train.deepspeed.brand_new_op"]

    assert registry.restore("deepspeed.brand_new_op")
    assert "brand_new_op" not in vars(operator_tree.package("deepspeed"))
    assert operator_tree.package("deepspeed").packbits is untouched


def test_original_none_is_preserved_exactly(operator_tree):
    package = operator_tree.package("megatron")
    package.softmax = None
    package.__all__.append("softmax")
    candidate = _candidate("new")
    registry = DynamicOpOverride()

    assert registry.override("megatron.softmax", candidate)
    assert package.__dict__["softmax"] is candidate

    assert registry.restore("megatron.softmax")
    assert "softmax" in vars(package)
    assert package.__dict__["softmax"] is None


def test_bare_name_with_multiple_package_exports_is_rejected(operator_tree):
    _add_export(operator_tree, "megatron", "shared_op", "megatron")
    _add_export(operator_tree, "deepspeed", "shared_op", "deepspeed")
    candidate = _candidate("new")
    registry = DynamicOpOverride()

    with pytest.warns(UserWarning) as records:
        assert not registry.override("shared_op", candidate)
    message = str(records[0].message)
    assert "ambiguous" in message
    assert "flag_train.megatron.shared_op" in message
    assert "flag_train.deepspeed.shared_op" in message
    assert operator_tree.package("megatron").shared_op() == "megatron"
    assert operator_tree.package("deepspeed").shared_op() == "deepspeed"
    assert registry.list_overrides() == []

    with pytest.raises(OverrideResolutionError, match="ambiguous"):
        resolve_operator_target("shared_op")

    with pytest.warns(UserWarning, match="Failed to restore"):
        assert not registry.restore("shared_op")


@pytest.mark.parametrize(
    "path",
    [
        "os.path.join",
        "torch.softmax",
        "megatron.ops.softmax",
        "flag_train.not_a_package.softmax",
        "flag_train.megatron",
        "megatron",
    ],
)
def test_unsupported_qualified_namespaces_are_rejected(operator_tree, path):
    registry = DynamicOpOverride()
    with pytest.warns(UserWarning, match="Failed to override"):
        assert not registry.override(path, _candidate("new"))
    assert registry.list_overrides() == []


@pytest.mark.parametrize(
    "spec",
    [
        "megatron..op=file.py",
        ".op=file.py",
        "op.=file.py",
        "bad-name=file.py",
        "op=file.py:",
        "op=",
    ],
)
def test_cli_parser_rejects_malformed_specs(spec):
    with pytest.raises(ValueError):
        parse_override_spec(spec)


def test_cli_parser_uses_bare_name_for_default_function():
    assert parse_override_spec("megatron.fused_proj_rms_backward=/tmp/impl.py") == (
        "megatron.fused_proj_rms_backward",
        "/tmp/impl.py",
        "fused_proj_rms_backward",
    )
    assert parse_override_spec(
        "flag_train.megatron.fused_proj_rms_backward=/tmp/impl.py:custom"
    ) == (
        "flag_train.megatron.fused_proj_rms_backward",
        "/tmp/impl.py",
        "custom",
    )
    assert default_impl_name("flag_train.megatron.fused_proj_rms_backward") == (
        "fused_proj_rms_backward"
    )


def test_cli_override_spec_applies_to_submodule_and_tracks_calls(
    operator_tree, tmp_path
):
    original = _add_export(operator_tree, "megatron", "fused_proj_rms_backward", "orig")
    filepath = _write_candidate(tmp_path, "fused_proj_rms_backward", "cli")
    registry = DynamicOpOverride()

    apply_overrides_from_args(
        _args(
            override=[
                "megatron.fused_proj_rms_backward={}".format(filepath),
            ]
        ),
        registry=registry,
    )

    overridden = operator_tree.package("megatron").fused_proj_rms_backward
    assert overridden is not original
    assert overridden() == "cli"

    # The candidate was invoked, so restore_all must not flag it as unused.
    registry.restore_all()
    assert operator_tree.package("megatron").fused_proj_rms_backward is original


def test_unused_file_candidate_is_reported_and_cleaned_up(operator_tree, tmp_path):
    original = _add_export(operator_tree, "megatron", "fast_gelu", "orig")
    filepath = _write_candidate(tmp_path, "fast_gelu", "file")
    registry = DynamicOpOverride()

    assert registry.override_from_file("megatron.fast_gelu", filepath)
    assert operator_tree.package("megatron").fast_gelu is not original

    with pytest.raises(AssertionError, match="never invoked"):
        registry.restore("flag_train.megatron.fast_gelu")

    assert operator_tree.package("megatron").fast_gelu is original
    assert registry.list_overrides() == []


def test_config_file_supports_qualified_paths_and_default_function(
    operator_tree, tmp_path
):
    original_megatron = _add_export(operator_tree, "megatron", "quick_gelu", "orig-m")
    original_deepspeed = _add_export(operator_tree, "deepspeed", "lamb", "orig-d")
    first_path = _write_candidate(tmp_path, "quick_gelu", "cfg-m")
    second_path = _write_candidate(tmp_path, "custom_lamb", "cfg-d")
    config_path = tmp_path / "overrides.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "overrides": {
                    "flag_train.megatron.quick_gelu": {"file": first_path},
                    "deepspeed.lamb": "{}:custom_lamb".format(second_path),
                }
            }
        )
    )

    parsed = load_override_config(str(config_path))
    assert parsed["flag_train.megatron.quick_gelu"] == (first_path, "quick_gelu")
    assert parsed["deepspeed.lamb"] == (second_path, "custom_lamb")

    registry = DynamicOpOverride()
    apply_overrides_from_args(
        _args(override_config=str(config_path)), registry=registry
    )

    megatron_impl = operator_tree.package("megatron").quick_gelu
    deepspeed_impl = operator_tree.package("deepspeed").lamb
    assert megatron_impl() == "cfg-m"
    assert deepspeed_impl() == "cfg-d"

    registry.restore_all()
    assert operator_tree.package("megatron").quick_gelu is original_megatron
    assert operator_tree.package("deepspeed").lamb is original_deepspeed


def test_config_file_rejects_malformed_operator_key(tmp_path):
    config_path = tmp_path / "overrides.yaml"
    config_path.write_text(
        yaml.safe_dump({"overrides": {"bad..name": str(tmp_path / "impl.py")}})
    )
    with pytest.raises(OverrideResolutionError):
        load_override_config(str(config_path))


def test_cli_ambiguous_name_exits_with_alternatives(operator_tree, capsys):
    _add_export(operator_tree, "megatron", "shared_op", "megatron")
    _add_export(operator_tree, "deepspeed", "shared_op", "deepspeed")

    with pytest.raises(SystemExit):
        apply_overrides_from_args(
            _args(override=["shared_op=/tmp/does-not-matter.py"])
        )

    stderr = capsys.readouterr().err
    assert "ambiguous" in stderr
    assert "flag_train.megatron.shared_op" in stderr
    assert "flag_train.deepspeed.shared_op" in stderr


def test_failed_cli_application_rolls_back_new_override(
    operator_tree, tmp_path, capsys
):
    original = _add_export(operator_tree, "megatron", "foo", "orig")
    filepath = _write_candidate(tmp_path, "candidate", "candidate")
    registry = DynamicOpOverride()

    with pytest.raises(SystemExit):
        apply_overrides_from_args(
            _args(
                override=[
                    "megatron.foo={}:candidate".format(filepath),
                    "outside.foo={}:candidate".format(filepath),
                ]
            ),
            registry=registry,
        )

    package = operator_tree.package("megatron")
    assert package.foo is original
    assert package.foo() == "orig"
    assert registry.list_overrides() == []
    assert registry._originals == {}
    assert registry._tracked == set()
    assert registry._call_counts == {}

    stderr = capsys.readouterr().err
    assert "outside" in stderr
    assert "never invoked" not in stderr


def test_failed_cli_application_restores_preexisting_overrides_exactly(
    operator_tree, tmp_path
):
    _add_export(operator_tree, "megatron", "foo", "orig")
    _add_export(operator_tree, "deepspeed", "bar", "orig-bar")
    old_path = _write_candidate(tmp_path, "old_foo", "old")
    new_path = _write_candidate(tmp_path, "new_foo", "new")
    registry = DynamicOpOverride()

    assert registry.override_from_file("megatron.foo", old_path, "old_foo")
    old_impl = operator_tree.package("megatron").foo
    assert old_impl() == "old"
    other_candidate = _candidate("bar-override")
    assert registry.override("deepspeed.bar", other_candidate)

    old_overrides = list(registry.list_overrides())
    old_originals = dict(registry._originals)
    old_tracked = set(registry._tracked)
    old_counts = dict(registry._call_counts)

    with pytest.raises(SystemExit), pytest.warns(
        UserWarning, match="Failed to load"
    ):
        apply_overrides_from_args(
            _args(
                override=[
                    "megatron.foo={}:new_foo".format(new_path),
                    # The later candidate names a function that does not exist.
                    "megatron.missing={}:absent".format(new_path),
                ]
            ),
            registry=registry,
        )

    package = operator_tree.package("megatron")
    assert package.foo is old_impl
    assert operator_tree.package("deepspeed").bar is other_candidate
    assert registry.list_overrides() == old_overrides
    assert registry._originals == old_originals
    assert registry._tracked == old_tracked
    assert registry._call_counts == old_counts
    assert package.foo() == "old"


def test_failed_cli_after_config_application_rolls_back_config_overrides(
    operator_tree, tmp_path
):
    original = _add_export(operator_tree, "megatron", "alpha", "orig-alpha")
    filepath = _write_candidate(tmp_path, "alpha_candidate", "cfg-alpha")
    config_path = tmp_path / "overrides.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "overrides": {
                    "megatron.alpha": {
                        "file": filepath,
                        "function": "alpha_candidate",
                    }
                }
            }
        )
    )
    registry = DynamicOpOverride()

    with pytest.raises(SystemExit):
        apply_overrides_from_args(
            _args(
                override_config=str(config_path),
                override=["outside.foo={}:alpha_candidate".format(filepath)],
            ),
            registry=registry,
        )

    package = operator_tree.package("megatron")
    assert package.alpha is original
    assert package.alpha() == "orig-alpha"
    assert registry.list_overrides() == []
    assert registry._originals == {}
    assert registry._tracked == set()
    assert registry._call_counts == {}


def test_failed_later_config_candidate_rolls_back_earlier_config_entry(
    operator_tree, tmp_path
):
    original = _add_export(operator_tree, "megatron", "alpha", "orig-alpha")
    _add_export(operator_tree, "deepspeed", "beta", "orig-beta")
    filepath = _write_candidate(tmp_path, "alpha_candidate", "cfg-alpha")
    config_path = tmp_path / "overrides.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "overrides": {
                    "megatron.alpha": {
                        "file": filepath,
                        "function": "alpha_candidate",
                    },
                    "deepspeed.beta": {
                        "file": str(tmp_path / "missing.py"),
                        "function": "beta_candidate",
                    },
                }
            }
        )
    )
    registry = DynamicOpOverride()

    with pytest.raises(SystemExit), pytest.warns(
        UserWarning, match="Failed to load"
    ):
        apply_overrides_from_args(
            _args(override_config=str(config_path)), registry=registry
        )

    assert operator_tree.package("megatron").alpha is original
    assert operator_tree.package("deepspeed").beta() == "orig-beta"
    assert registry.list_overrides() == []
    assert registry._originals == {}
    assert registry._tracked == set()
    assert registry._call_counts == {}


def test_alias_does_not_leak_first_override_as_original(operator_tree):
    original = _add_export(operator_tree, "megatron", "packbits", "orig")
    first = _candidate("first")
    second = _candidate("second")
    registry = DynamicOpOverride()

    assert registry.override("megatron.packbits", first)
    assert registry.override("flag_train.megatron.packbits", second)
    assert operator_tree.package("megatron").packbits is second
    assert registry.list_overrides() == ["flag_train.megatron.packbits"]

    assert registry.restore("megatron.packbits")
    assert operator_tree.package("megatron").packbits is original
    assert registry.list_overrides() == []


def test_failed_file_load_leaves_no_state(operator_tree, tmp_path):
    original = _add_export(operator_tree, "megatron", "fast_gelu", "orig")
    filepath = _write_candidate(tmp_path, "some_other_function", "file")
    registry = DynamicOpOverride()

    with pytest.warns(UserWarning, match="Failed to load"):
        assert not registry.override_from_file("megatron.fast_gelu", filepath)

    assert operator_tree.package("megatron").fast_gelu is original
    assert registry.list_overrides() == []
    assert registry._originals == {}
    assert registry._tracked == set()
    assert registry._call_counts == {}


def test_restore_all_handles_root_and_submodule_targets(operator_tree):
    root_original = _original("root")
    operator_tree.root.root_op = root_original
    package_original = _add_export(operator_tree, "megatron", "package_op", "pkg")
    registry = DynamicOpOverride()

    assert registry.override("root_op", _candidate("new-root"))
    assert registry.override("megatron.package_op", _candidate("new-pkg"))
    assert set(registry.list_overrides()) == {
        "flag_train.root_op",
        "flag_train.megatron.package_op",
    }

    registry.restore_all()

    assert operator_tree.root.root_op is root_original
    assert operator_tree.package("megatron").package_op is package_original
    assert registry._originals == {}
    assert registry._overrides == {}


def test_context_manager_restores_submodule_override(operator_tree):
    original = _add_export(operator_tree, "deepspeed", "unpackbits", "orig")
    candidate = _candidate("new")

    with DynamicOpOverride() as registry:
        assert registry.override("deepspeed.unpackbits", candidate)
        assert operator_tree.package("deepspeed").unpackbits is candidate

    assert operator_tree.package("deepspeed").unpackbits is original


def test_explicit_module_name_api_still_targets_that_module(operator_tree):
    original = _add_export(operator_tree, "megatron", "quick_gelu", "orig")
    candidate = _candidate("new")
    registry = DynamicOpOverride()

    assert registry.override("quick_gelu", candidate, "flag_train.megatron")
    assert operator_tree.package("megatron").quick_gelu is candidate
    assert not hasattr(operator_tree.root, "quick_gelu")

    assert registry.restore("quick_gelu", "flag_train.megatron")
    assert operator_tree.package("megatron").quick_gelu is original


def test_batch_override_resolves_each_qualified_target(operator_tree):
    original_one = _add_export(operator_tree, "megatron", "batch_one", "orig-1")
    original_two = _add_export(operator_tree, "deepspeed", "batch_two", "orig-2")
    registry = DynamicOpOverride()

    results = registry.override_batch(
        {
            "megatron.batch_one": _candidate("new-1"),
            "deepspeed.batch_two": _candidate("new-2"),
        }
    )
    assert results == {"megatron.batch_one": True, "deepspeed.batch_two": True}
    assert operator_tree.package("megatron").batch_one() == "new-1"
    assert operator_tree.package("deepspeed").batch_two() == "new-2"

    registry.restore_all()
    assert operator_tree.package("megatron").batch_one is original_one
    assert operator_tree.package("deepspeed").batch_two is original_two


def test_missing_exports_are_not_eagerly_accessed(operator_tree):
    package = operator_tree.package("megatron")
    package.__all__ = ["ghost"]

    def _raise(name):
        raise AssertionError("missing export was accessed: {}".format(name))

    package.__getattr__ = _raise
    candidate = _candidate("new")
    registry = DynamicOpOverride()

    # The bare name has no implementation anywhere; it falls back to the
    # top-level module instead of probing the package's missing export.
    assert registry.override("ghost", candidate)
    assert operator_tree.root.ghost is candidate
    assert registry.restore("ghost")
    assert "ghost" not in vars(operator_tree.root)
