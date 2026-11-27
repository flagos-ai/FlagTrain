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

"""Regression coverage for backend operator discovery."""

import sys
from types import ModuleType

import pytest

from flag_train.runtime import backend


def test_missing_dependency_keeps_other_backend_specializations(tmp_path, monkeypatch):
    package = ModuleType("_flagtrain_backend_registration_probe")
    package.__path__ = [str(tmp_path)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    for name in ("a_available", "c_available"):
        (tmp_path / f"{name}.py").write_text(
            f'def {name}():\n    return "specialized"\n\n__all__ = ["{name}"]\n',
            encoding="utf-8",
        )
    (tmp_path / "b_unavailable.py").write_text(
        "import _flagtrain_missing_registration_dependency\n"
        "\ndef b_unavailable():\n    return _flagtrain_missing_registration_dependency\n"
        '\n__all__ = ["b_unavailable"]\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(backend._state, "operator_modules", {"megatron": package})
    monkeypatch.setattr(backend._state, "vendor_extra_lib_imported", True)
    monkeypatch.setattr(backend._state, "customized_ops", None)

    def generic():
        return "generic"

    operators = dict.fromkeys(("a_available", "b_unavailable", "c_available"), generic)
    registrar = backend.SpecOpRegistrar(operators, vendor="probe")
    monkeypatch.setattr(registrar, "_get_specific_events", lambda: (backend._state,))

    try:
        registrar.apply()
        assert operators["a_available"]() == "specialized"
        assert operators["b_unavailable"] is generic
        assert operators["c_available"]() == "specialized"

        with pytest.raises(
            ModuleNotFoundError, match="_flagtrain_missing_registration_dependency"
        ):
            list(backend.iter_package_functions(package, include_package_exports=False))
    finally:
        for name in ("a_available", "c_available"):
            sys.modules.pop(f"{package.__name__}.{name}", None)
