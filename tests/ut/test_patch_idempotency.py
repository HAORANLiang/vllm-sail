# SPDX-License-Identifier: Apache-2.0
"""Patch installation must be idempotent across every entry point.

vLLM loads ``vllm.general_plugins`` in *every* process -- the API server, the
engine-core subprocess, and each worker -- and may invoke a hook more than once
within one process. The ``@patch`` decorator raises ``RuntimeError`` on a genuine
double application by design, so idempotency cannot be an emergent property of
import caching: it must be enforced by explicit guards at each layer.

This module exercises those guards on a bare CPU runner (no torch, no vLLM):

* the ``@patch`` / ``patch_value`` already-applied guard in
  ``vllm_sail/patch/utils.py`` (loaded via the ``patch_utils_module`` fixture),
* the ``_installed`` short-circuit in ``vllm_sail/patch/__init__.install()``,
* the ``_PATCHES_APPLIED`` / ``_REGISTRIES_APPLIED`` / ``_MODELS_REGISTERED``
  guards in ``vllm_sail.register_out_of_tree()``.

The two package-level guards are driven by loading the real ``__init__.py`` files
from disk into throwaway module objects (the repo's module-from-file pattern) and
stubbing their heavy imports, so the guard *logic* is executed without importing
vLLM. Class-method patching is intentionally not exercised here: it needs the
descriptor-preserving path, which ``test_patch_framework.py`` already covers.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from vllm_sail import envs

REPO_ROOT = Path(envs.__file__).resolve().parents[1]
PATCH_INIT = REPO_ROOT / "vllm_sail" / "patch" / "__init__.py"
PACKAGE_INIT = REPO_ROOT / "vllm_sail" / "__init__.py"


def _metadata() -> dict[str, str]:
    return {
        "reason": "exercise the idempotency guards",
        "affected_versions": ">=0.31.0,<0.32.0",
        "remove_when": "the guards are removed",
    }


@contextmanager
def throwaway_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """A fresh importable module, removed from ``sys.modules`` afterwards."""
    name = f"_ppu_idempotency_target_{uuid4().hex}"
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    yield module


@contextmanager
def loaded_from_path(path: Path) -> Iterator[types.ModuleType]:
    """Execute ``path`` as a private module so global guards start unset."""
    name = f"_vllm_sail_idempotency_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# @patch / patch_value already-applied guard (real utils, isolated module)
# ---------------------------------------------------------------------------
def test_double_module_patch_is_rejected(
    monkeypatch: pytest.MonkeyPatch, patch_utils_module: types.ModuleType
) -> None:
    """A second ``@patch`` on one target must raise, not silently re-wrap."""
    with throwaway_module(monkeypatch) as target:
        target.operation = lambda: "original"

        @patch_utils_module.patch(target.__name__, "operation", **_metadata())
        def first() -> str:
            return "first"

        assert target.operation is first
        full_target = f"{target.__name__}.operation"
        with pytest.raises(RuntimeError, match="already patched"):

            @patch_utils_module.patch(target.__name__, "operation", **_metadata())
            def second() -> str:
                return "second"

        # The rejected second patch left the target and registry untouched.
        assert target.operation is first
        recorded = [
            record
            for record in patch_utils_module.PATCH_REGISTRY
            if record.target == full_target
        ]
        assert len(recorded) == 1


def test_patch_marker_written_once_and_identity_stable(
    monkeypatch: pytest.MonkeyPatch, patch_utils_module: types.ModuleType
) -> None:
    """The sentinel is stamped on first install; the object never re-wraps."""
    with throwaway_module(monkeypatch) as target:

        def original() -> str:
            return "original"

        target.operation = original

        @patch_utils_module.patch(target.__name__, "operation", **_metadata())
        def replacement() -> str:
            return "replacement"

        full_target = f"{target.__name__}.operation"
        marker = getattr(replacement, patch_utils_module.PATCH_MARKER, None)
        assert isinstance(marker, dict)
        assert marker[full_target] is original

        stable_id = id(target.operation)
        for _ in range(3):
            # Re-reading the attribute never produces a new wrapper layer.
            assert target.operation is replacement
            assert id(target.operation) == stable_id
            assert not hasattr(target.operation, "__wrapped__")
        assert patch_utils_module.original_of(replacement, full_target) is original


def test_patch_value_reinstall_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, patch_utils_module: types.ModuleType
) -> None:
    """Re-installing an equal value is a no-op; a different value is an error."""
    with throwaway_module(monkeypatch) as target:
        patch_utils_module.patch_value(
            target.__name__, "TABLE", {"a": 1}, allow_missing=True, **_metadata()
        )
        assert target.TABLE == {"a": 1}
        after_first = len(patch_utils_module.PATCH_REGISTRY)

        # Equal value: clean no-op, no second registry entry, object unchanged.
        patch_utils_module.patch_value(
            target.__name__, "TABLE", {"a": 1}, allow_missing=True, **_metadata()
        )
        assert len(patch_utils_module.PATCH_REGISTRY) == after_first

        # Different value: upstream now owns the name, so the patch must go.
        with pytest.raises(RuntimeError, match="different value"):
            patch_utils_module.patch_value(
                target.__name__, "TABLE", {"a": 2}, allow_missing=True, **_metadata()
            )


# ---------------------------------------------------------------------------
# install() _installed short-circuit (real patch/__init__.py, stubbed imports)
# ---------------------------------------------------------------------------
def test_install_short_circuits_after_first_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``install()`` imports each category once no matter how often it is called."""
    with loaded_from_path(PATCH_INIT) as patch_pkg:
        assert patch_pkg.installed() is False
        categories = list(patch_pkg._CATEGORIES)
        assert categories, "no patch categories declared"

        recorded: list[str] = []
        real_import_module = importlib.import_module
        prefix = f"{patch_pkg.__name__}."

        def fake_import_module(
            name: str, *args: Any, **kwargs: Any
        ) -> types.ModuleType:
            if name.startswith(prefix):
                recorded.append(name)
                return types.ModuleType(name)
            return real_import_module(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", fake_import_module)

        patch_pkg.install()
        assert patch_pkg.installed() is True
        assert recorded == [f"{prefix}{category}" for category in categories]

        # Second and third calls are pure no-ops: no further category imports.
        patch_pkg.install()
        patch_pkg.install()
        assert len(recorded) == len(categories)


# ---------------------------------------------------------------------------
# register_out_of_tree() phase guards (real vllm_sail/__init__.py, stubbed deps)
# ---------------------------------------------------------------------------
def _stub(name: str, **attrs: Any) -> types.ModuleType:
    """A minimal stand-in module exposing only the attributes under test."""
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


def _register_stubs(monkeypatch: pytest.MonkeyPatch, counter: Counter[str]) -> None:
    """Inject counting stubs for every dependency register_out_of_tree imports.

    ``register_out_of_tree`` reaches its dependencies two ways: ``from
    vllm_sail.compat import ...`` resolves through ``sys.modules``, while ``import
    vllm_sail.patch`` followed by ``vllm_sail.patch.install()`` resolves through
    the *attribute* on the real ``vllm_sail`` package. Both are stubbed so the
    guards are exercised without importing vLLM.
    """
    import vllm_sail

    def bump(key: str) -> Any:
        def _call(*_args: Any, **_kwargs: Any) -> None:
            counter[key] += 1

        return _call

    stubs = {
        "vllm_sail.compat": _stub(
            "vllm_sail.compat", check_vllm_compatibility=bump("compat")
        ),
        "vllm_sail.patch": _stub("vllm_sail.patch", install=bump("patch")),
        "vllm_sail.ops": _stub("vllm_sail.ops"),
        "vllm_sail.registry": _stub("vllm_sail.registry", register=bump("registry")),
        "vllm_sail.profiling": _stub("vllm_sail.profiling", install=bump("profiling")),
        "vllm_sail.profiling.layerwise": _stub(
            "vllm_sail.profiling.layerwise", install=bump("layerwise")
        ),
        "vllm_sail.models": _stub("vllm_sail.models", register_model=bump("models")),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        monkeypatch.setattr(
            sys.modules.get(parent, vllm_sail), child, module, raising=False
        )


def test_register_out_of_tree_runs_each_phase_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated hook calls apply patches/registries/models exactly once."""
    counter: Counter[str] = Counter()
    _register_stubs(monkeypatch, counter)
    with loaded_from_path(PACKAGE_INIT) as pkg:
        assert pkg._PATCHES_APPLIED is False
        assert pkg._REGISTRIES_APPLIED is False
        assert pkg._MODELS_REGISTERED is False

        pkg.register_out_of_tree()
        pkg.register_out_of_tree()
        pkg.register_out_of_tree()

        assert pkg._PATCHES_APPLIED is True
        assert pkg._REGISTRIES_APPLIED is True
        assert pkg._MODELS_REGISTERED is True
        # The compatibility check is cheap and internally cached, so it may run on
        # every call; the guarded heavy phases must each run exactly once.
        assert counter["patch"] == 1
        assert counter["registry"] == 1
        assert counter["profiling"] == 1
        assert counter["layerwise"] == 1
        assert counter["models"] == 1
        assert counter["compat"] == 3


def test_register_out_of_tree_does_not_apply_patches_when_compat_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed version check must abort before any patch phase runs.

    This is the "install failure leaves no partial state" guarantee: the guard
    stays unset so a later, valid call can still apply everything.
    """
    counter: Counter[str] = Counter()
    _register_stubs(monkeypatch, counter)

    def explode(*_args: Any, **_kwargs: Any) -> None:
        counter["compat"] += 1
        raise RuntimeError("requires vLLM >=0.31,<0.32")

    compat_stub = sys.modules["vllm_sail.compat"]
    monkeypatch.setattr(compat_stub, "check_vllm_compatibility", explode)

    with loaded_from_path(PACKAGE_INIT) as pkg:
        with pytest.raises(RuntimeError, match="requires vLLM"):
            pkg.register_out_of_tree()
        assert pkg._PATCHES_APPLIED is False
        assert pkg._REGISTRIES_APPLIED is False
        assert pkg._MODELS_REGISTERED is False
        assert counter["patch"] == 0
        assert counter["registry"] == 0
        assert counter["models"] == 0
