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

"""Regression tests for backend operator discovery."""

import importlib
import sys
from pathlib import Path

import pytest

from flag_train.runtime import backend


@pytest.fixture
def operator_package(tmp_path, monkeypatch):
    name = "_test_backend_operator_exports"
    package = tmp_path / name
    package.mkdir()
    files = {
        "__init__.py": 'def legacy(): return "legacy"\n__all__ = ["legacy"]\n',
        "operators.py": (
            'def module_only(): return "module"\n'
            'def layered(): return "module"\n'
            'def helper(): return "helper"\n'
            '__all__ = ["module_only", "layered"]\n'
        ),
        "unlisted.py": 'def unlisted(): return "unlisted"\n',
        "_private.py": 'raise AssertionError("private module must not be imported")\n',
        "nested/__init__.py": 'def layered(): return "nested"\n',
    }
    for filename, content in files.items():
        path = package / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    operator_package = importlib.import_module(name)
    monkeypatch.setattr(backend._state, "vendor_extra_lib_imported", True)
    monkeypatch.setattr(
        backend._state, "operator_modules", {"megatron": operator_package}
    )
    monkeypatch.setattr(backend._state, "customized_ops", None)
    yield operator_package
    for module_name in list(sys.modules):
        if module_name == name or module_name.startswith(f"{name}."):
            del sys.modules[module_name]


@pytest.mark.usefixtures("operator_package")
def test_backend_module_exports_override_generic_functions(monkeypatch):
    def generic():
        return "generic"

    registry = {"module_only": generic, "unlisted": generic}
    registrar = backend.SpecOpRegistrar(registry)
    monkeypatch.setattr(registrar, "_get_specific_events", lambda: (backend._state,))

    registrar.apply()

    assert {name: fn() for name, fn in registry.items()} == {
        "module_only": "module",
        "unlisted": "generic",
        "legacy": "legacy",
        "layered": "nested",
    }


def test_failed_backend_discovery_leaves_no_partial_cache(
    monkeypatch, operator_package
):
    broken = Path(operator_package.__path__[0]) / "z_broken.py"
    broken.write_text(
        'raise ImportError("unavailable backend dependency")\n', encoding="utf-8"
    )

    with pytest.raises(ImportError, match="unavailable backend dependency"):
        backend.get_customized_ops()
    assert backend._state.customized_ops is None

    broken.unlink()
    ops = dict(backend.get_customized_ops())
    assert ops["module_only"]() == "module"
