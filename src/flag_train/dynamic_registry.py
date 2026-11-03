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

"""
Dynamic operator override interface for flexible testing.

This module provides utilities to dynamically override operator implementations
in the flag_train module without modifying source code, enabling:
- Concurrent testing of different operator implementations
- Command-line driven operator override
- A/B testing of operator variants
"""

import functools
import importlib
import importlib.util
import sys
import types
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union

_MISSING = object()

_ROOT_MODULE = "flag_train"
_SUPPORTED_PACKAGES = ("deepspeed", "megatron", "transformer_engine")
_PACKAGE_MODULES = {
    package: f"{_ROOT_MODULE}.{package}" for package in _SUPPORTED_PACKAGES
}


class OverrideResolutionError(ValueError):
    """Raised when an override target cannot be resolved unambiguously."""


class _ResolvedTarget(NamedTuple):
    """Canonical location of an operator implementation."""

    module_name: str
    attr_name: str

    @property
    def full_name(self) -> str:
        return f"{self.module_name}.{self.attr_name}"


def validate_operator_name(op_name: str) -> List[str]:
    """
    Validate the shape of an operator name and return its dotted components.

    Both bare operator names (``fused_proj_rms_backward``) and qualified
    paths (``megatron.fused_proj_rms_backward`` and
    ``flag_train.megatron.fused_proj_rms_backward``) are accepted here. The
    namespace itself is checked later, once the target module is resolved.
    """
    if not isinstance(op_name, str):
        raise OverrideResolutionError(
            f"Operator name must be a string, got {type(op_name).__name__}"
        )
    if not op_name or op_name != op_name.strip():
        raise OverrideResolutionError(
            f"Malformed operator path {op_name!r}: leading/trailing whitespace "
            "is not permitted"
        )

    parts = op_name.split(".")
    if any(not part or not part.isidentifier() for part in parts):
        raise OverrideResolutionError(
            f"Malformed operator path {op_name!r}: expected a dotted Python "
            "identifier such as 'fused_proj_rms_backward', "
            "'megatron.fused_proj_rms_backward', or "
            "'flag_train.megatron.fused_proj_rms_backward'"
        )
    return parts


def default_impl_name(op_name: str) -> str:
    """Return the function name to load when an override omits one.

    The function lives next to the operator, so a qualified operator spec
    such as ``megatron.fused_proj_rms_backward`` defaults to
    ``fused_proj_rms_backward`` -- not to the dotted path itself, which is
    not a valid Python attribute name.
    """
    return op_name.rsplit(".", 1)[-1]


def _import_module(module_name: str):
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise OverrideResolutionError(
            f"Cannot import module {module_name!r} for operator override: {exc}"
        ) from exc


def _target_for_module(module, attr_name: str) -> _ResolvedTarget:
    module_name = getattr(module, "__name__", None) or str(module)
    return _ResolvedTarget(module_name, attr_name)


def _is_exported_callable(module, attr_name: str) -> bool:
    """Check an export without touching missing or lazy attributes.

    ``vars(module)`` is inspected directly so that a package with a
    ``__getattr__`` hook (or a partially initialised package) is not
    probed for names it does not define.
    """
    namespace = vars(module)
    value = namespace.get(attr_name, _MISSING)
    if value is _MISSING or not callable(value):
        return False

    exports = namespace.get("__all__")
    if exports is None:
        return True
    try:
        return attr_name in exports
    except TypeError:
        return True


def _resolve_qualified_parts(parts: List[str]) -> _ResolvedTarget:
    """Resolve a dotted path that names a package and an attribute."""
    if parts[0] == _ROOT_MODULE:
        if len(parts) == 2:
            root = _import_module(_ROOT_MODULE)
            if isinstance(vars(root).get(parts[1], _MISSING), types.ModuleType):
                raise OverrideResolutionError(
                    f"Unsupported operator path {'.'.join(parts)!r}: "
                    f"{parts[1]!r} names a module, not an operator. Use "
                    f"'flag_train.<package>.<operator>' for an operator in "
                    "a supported package."
                )
            return _ResolvedTarget(_ROOT_MODULE, parts[1])
        if len(parts) == 3 and parts[1] in _PACKAGE_MODULES:
            return _ResolvedTarget(_PACKAGE_MODULES[parts[1]], parts[2])
        raise OverrideResolutionError(
            f"Unsupported operator path {'.'.join(parts)!r}: expected "
            "'flag_train.<operator>' or one of "
            "'flag_train.deepspeed.<operator>', "
            "'flag_train.megatron.<operator>', "
            "'flag_train.transformer_engine.<operator>'"
        )

    if parts[0] in _PACKAGE_MODULES and len(parts) == 2:
        return _ResolvedTarget(_PACKAGE_MODULES[parts[0]], parts[1])

    if parts[0] in _PACKAGE_MODULES:
        raise OverrideResolutionError(
            f"Unsupported operator path {'.'.join(parts)!r}: expected "
            f"'{parts[0]}.<operator>'"
        )

    raise OverrideResolutionError(
        f"Unsupported operator namespace {parts[0]!r} in "
        f"{'.'.join(parts)!r}. Only flag_train and its operator packages "
        f"({', '.join(_SUPPORTED_PACKAGES)}) may be overridden."
    )


def _operator_packages():
    """Yield the supported operator packages already bound on ``flag_train``.

    Only modules already imported are considered. Importing ``flag_train``
    always imports its operator packages, and avoiding extra imports here
    keeps host-side resolution tests free of torch/triton side effects.
    """
    for module_name in _PACKAGE_MODULES.values():
        module = sys.modules.get(module_name)
        if module is not None:
            yield module


def resolve_operator_target(
    op_name: str,
    module_name: str = "flag_train",
) -> _ResolvedTarget:
    """
    Resolve an operator name to the module attribute it denotes.

    - A dotted path is resolved verbatim, restricted to ``flag_train`` and
      its supported operator packages (``deepspeed``, ``megatron`` and
      ``transformer_engine``).
    - A bare name first preserves an attribute that already exists at the
      ``flag_train`` top level. Otherwise it is matched against the callables
      exported by the supported packages. Exactly one match is used; zero
      matches fall back to the top-level module so a missing implementation
      can still be injected explicitly; multiple matches are rejected rather
      than silently selecting a package.
    - An explicit non-default ``module_name`` keeps its historical meaning
      for bare names: the attribute is written to that module directly.
    """
    parts = validate_operator_name(op_name)

    if len(parts) > 1:
        target = _resolve_qualified_parts(parts)
        module = _import_module(target.module_name)
        return _target_for_module(module, target.attr_name)

    if module_name != _ROOT_MODULE:
        module = _import_module(module_name)
        return _target_for_module(module, parts[0])

    root = _import_module(_ROOT_MODULE)
    existing = vars(root).get(parts[0], _MISSING)
    if existing is not _MISSING and not isinstance(existing, types.ModuleType):
        return _target_for_module(root, parts[0])

    matches = [
        _target_for_module(package, parts[0])
        for package in _operator_packages()
        if _is_exported_callable(package, parts[0])
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        alternatives = ", ".join(sorted(target.full_name for target in matches))
        raise OverrideResolutionError(
            f"Operator {op_name!r} is ambiguous: it is exported by multiple "
            f"packages ({alternatives}). Use a fully qualified operator "
            f"name such as {matches[0].full_name!r}."
        )

    if isinstance(existing, types.ModuleType):
        raise OverrideResolutionError(
            f"{op_name!r} names the flag_train module "
            f"{existing.__name__!r}, not an operator. Use a qualified "
            "operator name such as 'megatron.<operator>'."
        )

    # No exported implementation: keep the historical top-level target so an
    # explicit missing operator can be injected and later removed cleanly.
    return _target_for_module(root, parts[0])


class DynamicOpOverride:
    """
    Manage dynamic operator implementation override in flag_train module.

    This class allows runtime override of operator implementations that are
    directly called via flag_train.op_name() API, without modifying source code.

    Example:
        >>> registry = DynamicOpOverride()
        >>> # Override an implementation
        >>> registry.override("softmax", my_custom_softmax)
        >>> # Test it
        >>> result = flag_train.softmax(x)
        >>> # Restore original
        >>> registry.restore("softmax")
    """

    def __init__(self):
        self._overrides: Dict[str, Any] = {}
        self._originals: Dict[str, Any] = {}
        # Overrides loaded via override_from_file (i.e. the CLI/config path)
        # are tracked so restore()/restore_all() can flag ones that were
        # registered but never actually invoked by a test.
        self._tracked: set = set()
        self._call_counts: Dict[str, int] = {}

    def resolve_target(
        self,
        op_name: str,
        module_name: str = "flag_train",
    ) -> _ResolvedTarget:
        """Resolve ``op_name`` to its canonical module/attribute target."""
        return resolve_operator_target(op_name, module_name)

    def _apply_override(
        self,
        target: _ResolvedTarget,
        impl_func: Callable,
    ) -> None:
        """Write ``impl_func`` at ``target``, recording the true original once."""
        module = _import_module(target.module_name)
        full_name = target.full_name

        # An override registered through a different alias (e.g.
        # ``megatron.op`` and then ``flag_train.megatron.op``) must not save
        # the first override as the "original": that would leak the alias and
        # make restore() put the override back instead of the real value.
        is_new = full_name not in self._originals
        original = (
            vars(module).get(target.attr_name, _MISSING) if is_new else _MISSING
        )

        setattr(module, target.attr_name, impl_func)

        if is_new:
            self._originals[full_name] = original
        self._overrides[full_name] = impl_func

    def override(
        self,
        op_name: str,
        impl_func: Callable,
        module_name: str = "flag_train",
    ) -> bool:
        """
        Override an operator implementation in the specified module.

        Operators without a public implementation are added temporarily and
        removed by restore(). Existing attributes are restored to their exact
        original value, including None.

        Args:
            op_name: Operator name (e.g., "softmax", "rms_norm",
                "megatron.fused_proj_rms_backward", or
                "flag_train.megatron.fused_proj_rms_backward")
            impl_func: New implementation function
            module_name: Module name to override in for bare operator names
                (default: "flag_train")

        Returns:
            True if override succeeded, False otherwise
        """
        try:
            target = self.resolve_target(op_name, module_name)
            self._apply_override(target, impl_func)
            return True
        except Exception as e:
            warnings.warn(f"Failed to override '{op_name}': {e}")
            return False

    def override_batch(
        self,
        op_mapping: Dict[str, Callable],
        module_name: str = "flag_train",
    ) -> Dict[str, bool]:
        """
        Override multiple operators at once.

        Args:
            op_mapping: Dict mapping op_name to implementation function
            module_name: Module name to override in (default: "flag_train")

        Returns:
            Dict mapping op_name to override success status
        """
        results = {}
        for op_name, impl_func in op_mapping.items():
            results[op_name] = self.override(op_name, impl_func, module_name)
        return results

    def _restore_target(self, full_name: str) -> bool:
        """Restore a canonical ``module.attribute`` identity."""
        if full_name not in self._originals:
            warnings.warn(
                f"Operator '{full_name}' was not overridden by this registry"
            )
            return False

        module_name, attr_name = full_name.rsplit(".", 1)
        module = sys.modules.get(module_name)
        if module is None:
            warnings.warn(f"Module '{module_name}' not loaded")
            return False

        try:
            original = self._originals[full_name]

            if original is not _MISSING:
                setattr(module, attr_name, original)
            elif attr_name in vars(module):
                delattr(module, attr_name)

            was_unused = (
                full_name in self._tracked and self._call_counts.get(full_name, 0) == 0
            )

            # Clean up tracking
            del self._originals[full_name]
            self._overrides.pop(full_name, None)
            self._tracked.discard(full_name)
            self._call_counts.pop(full_name, None)

        except Exception as e:
            warnings.warn(f"Failed to restore '{full_name}': {e}")
            return False

        if was_unused:
            raise AssertionError(
                f"Operator '{attr_name}' was overridden via override_from_file "
                f"but its candidate implementation was never invoked"
            )

        return True

    def restore(
        self,
        op_name: str,
        module_name: str = "flag_train",
    ) -> bool:
        """
        Restore an operator to its original implementation.

        The same qualified/bare resolution rules used by :meth:`override`
        apply here, so aliases such as ``megatron.op`` and
        ``flag_train.megatron.op`` address the same canonical target.

        Args:
            op_name: Operator name to restore
            module_name: Module name for bare operator names (default: "flag_train")

        Returns:
            True if restoration succeeded, False otherwise
        """
        try:
            target = self.resolve_target(op_name, module_name)
        except Exception as e:
            warnings.warn(f"Failed to restore '{op_name}': {e}")
            return False
        return self._restore_target(target.full_name)

    def restore_all(self, module_name: str = "flag_train"):
        """
        Restore all overridden operators in the specified module.

        Targets are restored by their canonical identity, so overrides of
        submodule operators (e.g. ``flag_train.megatron.*``) are included
        when restoring the ``flag_train`` root.

        Args:
            module_name: Module name (default: "flag_train")

        Raises:
            AssertionError: If any override loaded via ``override_from_file``
                (i.e. through the CLI/config path) was registered but never
                actually invoked before being restored.
        """
        prefix = f"{module_name.rstrip('.')}."
        ops_to_restore = [
            full_name
            for full_name in list(self._originals.keys())
            if full_name.startswith(prefix)
        ]

        unused_errors = []
        for full_name in ops_to_restore:
            try:
                self._restore_target(full_name)
            except AssertionError as e:
                unused_errors.append(str(e))

        if unused_errors:
            raise AssertionError("; ".join(unused_errors))

    def list_overrides(self) -> List[str]:
        """Return list of currently overridden operator names."""
        return list(self._overrides.keys())

    def load_impl_from_file(self, filepath: str, func_name: str) -> Optional[Callable]:
        """
        Load an implementation function from a Python file.

        Args:
            filepath: Path to Python file containing the implementation
            func_name: Name of the function to load

        Returns:
            The loaded function, or None if loading failed
        """
        try:
            path = Path(filepath)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {filepath}")

            spec = importlib.util.spec_from_file_location(
                f"custom_impl_{path.stem}", filepath
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load spec from {filepath}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if not hasattr(module, func_name):
                raise AttributeError(f"Function '{func_name}' not found in {filepath}")

            return getattr(module, func_name)

        except Exception as e:
            warnings.warn(f"Failed to load implementation from {filepath}: {e}")
            return None

    def override_from_file(
        self,
        op_name: str,
        filepath: str,
        func_name: Optional[str] = None,
        module_name: str = "flag_train",
    ) -> bool:
        """
        Override an operator implementation from a Python file.

        Args:
            op_name: Operator name to override. Bare names and qualified
                paths are both accepted.
            filepath: Path to Python file containing the implementation
            func_name: Function name to load (defaults to the bare operator
                name, e.g. ``fused_proj_rms_backward`` for
                ``megatron.fused_proj_rms_backward``)
            module_name: Module name for bare operator names
                (default: "flag_train")

        Returns:
            True if override succeeded, False otherwise
        """
        if func_name is None:
            func_name = default_impl_name(op_name)

        try:
            target = self.resolve_target(op_name, module_name)
        except Exception as e:
            warnings.warn(f"Failed to override '{op_name}': {e}")
            return False

        impl_func = self.load_impl_from_file(filepath, func_name)
        if impl_func is None:
            return False

        # Call tracking uses the canonical identity so restoring through any
        # alias sees the same counter.
        wrapped = self._wrap_with_call_tracking(target.full_name, impl_func)
        try:
            self._apply_override(target, wrapped)
        except Exception as e:
            warnings.warn(f"Failed to override '{op_name}': {e}")
            return False

        self._tracked.add(target.full_name)
        self._call_counts[target.full_name] = 0
        return True

    def _wrap_with_call_tracking(self, full_name: str, impl_func: Callable) -> Callable:
        @functools.wraps(impl_func)
        def wrapper(*args, **kwargs):
            self._call_counts[full_name] += 1
            return impl_func(*args, **kwargs)

        return wrapper

    def override_batch_from_files(
        self,
        op_file_mapping: Dict[str, Union[str, tuple]],
        module_name: str = "flag_train",
    ) -> Dict[str, bool]:
        """
        Override multiple operators from files.

        Args:
            op_file_mapping: Dict mapping op_name to either:
                - A string filepath (function name same as op_name)
                - A tuple (filepath, function_name)
            module_name: Module name to override in (default: "flag_train")

        Returns:
            Dict mapping op_name to override success status
        """
        results = {}
        for op_name, file_spec in op_file_mapping.items():
            if isinstance(file_spec, str):
                filepath = file_spec
                func_name = default_impl_name(op_name)
            elif isinstance(file_spec, (tuple, list)) and len(file_spec) == 2:
                filepath, func_name = file_spec
            else:
                warnings.warn(f"Invalid file spec for '{op_name}': {file_spec}")
                results[op_name] = False
                continue

            results[op_name] = self.override_from_file(
                op_name, filepath, func_name, module_name
            )

        return results

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - restore all overrides."""
        self.restore_all()

    def __del__(self):
        """Destructor - restore on deletion."""
        self.restore_all()


# Global registry instance for convenience
_global_registry = DynamicOpOverride()


def override_op(
    op_name: str,
    impl_func: Callable,
    module_name: str = "flag_train",
) -> bool:
    """
    Convenience function to override a single operator using global registry.

    Args:
        op_name: Operator name
        impl_func: New implementation function
        module_name: Module name (default: "flag_train")

    Returns:
        True if override succeeded
    """
    return _global_registry.override(op_name, impl_func, module_name)


def restore_op(op_name: str, module_name: str = "flag_train") -> bool:
    """
    Convenience function to restore a single operator using global registry.

    Args:
        op_name: Operator name
        module_name: Module name (default: "flag_train")

    Returns:
        True if restoration succeeded
    """
    return _global_registry.restore(op_name, module_name)


def restore_all_ops(module_name: str = "flag_train"):
    """
    Convenience function to restore all operators using global registry.

    Args:
        module_name: Module name (default: "flag_train")
    """
    _global_registry.restore_all(module_name)


def get_global_registry() -> DynamicOpOverride:
    """Get the global registry instance."""
    return _global_registry
