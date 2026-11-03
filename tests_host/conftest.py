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

"""Host-only fixtures for the shared override infrastructure.

These tests exercise the real ``flag_train.dynamic_registry`` and
``flag_train.cli_override`` modules, but they must not import the real
``flag_train/__init__.py`` (which pulls in torch/triton/GPU runtimes). A
minimal ``flag_train`` package object pointing at the real source directory
lets Python load the two shared modules directly; the operator packages are
replaced by controlled fakes for every test.
"""

import sys
import types
from pathlib import Path

import pytest

_PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "flag_train"
_OPERATOR_PACKAGES = ("deepspeed", "megatron", "transformer_engine")
_SHARED_MODULES = {
    "flag_train",
    "flag_train.dynamic_registry",
    "flag_train.cli_override",
}


def _install_host_package_stub():
    """Put a lightweight ``flag_train`` package in ``sys.modules``.

    The stub has a ``__path__`` so submodules such as
    ``flag_train.dynamic_registry`` load from source, but its ``__init__``
    is never executed.
    """
    package = types.ModuleType("flag_train")
    package.__path__ = [str(_PACKAGE_DIR)]
    package.__package__ = "flag_train"
    package.__file__ = str(_PACKAGE_DIR / "__init__.py")
    package.__all__ = []
    sys.modules["flag_train"] = package


_install_host_package_stub()


class OperatorTree:
    """Controlled ``flag_train`` root and its three operator packages."""

    def __init__(self, root, packages):
        self.root = root
        self.packages = packages

    def package(self, name):
        return self.packages[name]


@pytest.fixture(autouse=True)
def operator_tree():
    """Provide fresh fake operator packages and restore the world afterwards."""
    root = sys.modules["flag_train"]
    root_snapshot = dict(vars(root))
    modules_snapshot = {
        name: module
        for name, module in sys.modules.items()
        if name == "flag_train" or name.startswith("flag_train.")
    }

    # Remove any previously imported operator package so each test starts
    # from the fakes installed below.
    for name in list(sys.modules):
        if name.startswith("flag_train.") and name not in _SHARED_MODULES:
            del sys.modules[name]

    packages = {}
    for short_name in _OPERATOR_PACKAGES:
        module_name = f"flag_train.{short_name}"
        module = types.ModuleType(module_name)
        module.__package__ = module_name
        module.__all__ = []
        packages[short_name] = module
        sys.modules[module_name] = module
        setattr(root, short_name, module)

    yield OperatorTree(root, packages)

    # Restore root attributes and the module table exactly.
    for name in list(vars(root)):
        if name not in root_snapshot:
            delattr(root, name)
    for name, value in root_snapshot.items():
        setattr(root, name, value)

    for name in list(sys.modules):
        if (name == "flag_train" or name.startswith("flag_train.")) and (
            name not in modules_snapshot
        ):
            del sys.modules[name]
    sys.modules.update(modules_snapshot)
