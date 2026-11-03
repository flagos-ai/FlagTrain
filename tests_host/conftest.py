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
``flag_train/__init__.py`` (which pulls in torch/triton/GPU runtimes).

The shared source files are therefore loaded under an isolated synthetic
package (``_flagtrain_override_host``) at collection time. Collection never
touches ``flag_train`` or any of its ``sys.modules`` children: a controlled
fake ``flag_train`` tree is installed only for the duration of each host test
and the previous module table is restored in ``finally``.
"""

import gc
import importlib
import sys
import types
from pathlib import Path

import pytest

_PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "flag_train"
_OPERATOR_PACKAGES = ("deepspeed", "megatron", "transformer_engine")
_ISOLATED_PACKAGE = "_flagtrain_override_host"
_SHARED_MODULES = ("dynamic_registry", "cli_override")


def _load_shared_modules():
    """Load the shared source modules under an isolated package.

    The synthetic package has a ``__path__`` pointing at the real source
    directory, so relative imports between the shared modules work, but its
    ``__init__`` is never executed. Loading is idempotent: re-running the
    collection-time code (for example through ``runpy``) must not replace an
    already loaded package or its modules.
    """
    package = sys.modules.get(_ISOLATED_PACKAGE)
    if package is None:
        package = types.ModuleType(_ISOLATED_PACKAGE)
        package.__path__ = [str(_PACKAGE_DIR)]
        package.__package__ = _ISOLATED_PACKAGE
        sys.modules[_ISOLATED_PACKAGE] = package

    for short_name in _SHARED_MODULES:
        module_name = f"{_ISOLATED_PACKAGE}.{short_name}"
        if module_name not in sys.modules:
            importlib.import_module(module_name)

    return package


_load_shared_modules()


class OperatorTree:
    """Controlled ``flag_train`` root and its three operator packages."""

    def __init__(self, root, packages):
        self.root = root
        self.packages = packages

    def package(self, name):
        return self.packages[name]


def _flag_train_modules():
    return {
        name: module
        for name, module in sys.modules.items()
        if name == "flag_train" or name.startswith("flag_train.")
    }


@pytest.fixture(autouse=True)
def operator_tree():
    """Install a private fake ``flag_train`` tree for one host test.

    Every ``flag_train`` entry is removed while the fake tree is active (so a
    mixed run cannot fall back to a real submodule) and the complete previous
    module table is restored in ``finally``.
    """
    previous_modules = _flag_train_modules()

    root = types.ModuleType("flag_train")
    root.__path__ = []
    root.__package__ = "flag_train"
    root.__all__ = []

    packages = {}
    try:
        for name in list(_flag_train_modules()):
            del sys.modules[name]
        sys.modules["flag_train"] = root
        for short_name in _OPERATOR_PACKAGES:
            module_name = f"flag_train.{short_name}"
            module = types.ModuleType(module_name)
            module.__package__ = module_name
            module.__all__ = []
            packages[short_name] = module
            sys.modules[module_name] = module
            setattr(root, short_name, module)

        yield OperatorTree(root, packages)

    finally:
        # Let registries abandoned by a test run their destructor while the
        # fake tree is still installed, so they cannot mutate a real package
        # after the module table has been restored.
        gc.collect()
        for name in list(_flag_train_modules()):
            del sys.modules[name]
        sys.modules.update(previous_modules)
