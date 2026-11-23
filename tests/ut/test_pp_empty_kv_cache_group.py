# SPDX-License-Identifier: Apache-2.0
"""Tests for the empty PP KV cache group drop (bugfix/pp_empty_kv_cache_group).

Two-tier design like ``test_qwen4_exp_ple_quant_scheme.py``:

* **Bare-CPU import + metadata shape.** The leaf imports no vLLM at module
  scope; ``METADATA`` and the ``patch`` call carry the mandatory fields.
* **Behaviour against stubs.** ``install()`` runs against a throwaway
  stand-in of the upstream planning module whose projection reproduces the
  empty-group defect, asserting the wrapper drops only the empty groups.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tests.support.patch_inventory import scan_patch_file
from tests.support.upstream import SUPPORTED_RANGE
from vllm_sail.patch.utils import PATCH_REGISTRY, original_of, patch

MODULE_PATH = (
    Path(__file__).parents[2]
    / "vllm_sail"
    / "patch"
    / "bugfix"
    / "pp_empty_kv_cache_group.py"
)
TARGET = "vllm.v1.core.kv_cache_planning._project_kv_cache_groups_to_worker"
REQUIRED_METADATA = ("reason", "affected_versions", "remove_when")


def _load_leaf(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def leaf_module() -> Iterator[types.ModuleType]:
    """A fresh leaf-module instance per test (install state must not leak)."""
    name = f"_pp_empty_group_leaf_test_{id(object())}"
    module = _load_leaf(MODULE_PATH, name)
    registry_before = len(PATCH_REGISTRY)
    try:
        yield module
    finally:
        del PATCH_REGISTRY[registry_before:]
        sys.modules.pop(name, None)


def _register_chain(
    monkeypatch: pytest.MonkeyPatch, leaf_name: str, leaf_module: types.ModuleType
) -> None:
    parts = leaf_name.split(".")
    chain = [".".join(parts[: i + 1]) for i in range(len(parts))]
    modules: dict[str, types.ModuleType] = {}
    for name in chain[:-1]:
        existing = sys.modules.get(name)
        module = (
            existing
            if isinstance(existing, types.ModuleType)
            else types.ModuleType(name)
        )
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    modules[leaf_name] = leaf_module
    monkeypatch.setitem(sys.modules, leaf_name, leaf_module)
    for parent_name, child_name in zip(chain, chain[1:], strict=False):
        parent = modules[parent_name]
        attribute = child_name.rpartition(".")[2]
        monkeypatch.setattr(parent, attribute, modules[child_name], raising=False)


def _stub_planning(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Upstream projection with the empty-group defect, on plain namespaces."""

    def _project(global_kv_cache_groups, worker_spec):
        projected = []
        for group in global_kv_cache_groups:
            worker_layer_names = [
                layer_name
                for layer_name in group.layer_names
                if layer_name in worker_spec
            ]
            if len(worker_layer_names) == len(group.layer_names):
                projected.append(group)
                continue
            # Upstream keeps the empty group with the unfiltered global spec.
            projected.append(
                NS(
                    layer_names=worker_layer_names,
                    kv_cache_spec=group.kv_cache_spec,
                )
            )
        return projected

    planning = types.ModuleType("vllm.v1.core.kv_cache_planning")
    planning._project_kv_cache_groups_to_worker = _project  # type: ignore[attr-defined]
    _register_chain(monkeypatch, "vllm.v1.core.kv_cache_planning", planning)

    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda name: NS(  # type: ignore[attr-defined]
        warning=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
    )
    _register_chain(monkeypatch, "vllm.logger", logger_mod)
    return planning


# ---------------------------------------------------------------------------
# Bare-CPU import and metadata shape
# ---------------------------------------------------------------------------


def test_module_imports_on_bare_cpu(leaf_module) -> None:
    if "vllm" in sys.modules:
        pytest.skip("bare-CPU contract test; real vLLM is loaded (mode B)")
    assert callable(leaf_module.install)
    assert leaf_module.__doc__


def test_metadata_shape(leaf_module) -> None:
    assert leaf_module.METADATA == (
        (TARGET, leaf_module.REASON, SUPPORTED_RANGE, leaf_module.REMOVE_WHEN),
    )
    for target, reason, affected, remove_when in leaf_module.METADATA:
        assert target.startswith("vllm."), target
        for value in (reason, affected, remove_when):
            assert isinstance(value, str) and value.strip(), target
        assert ">=" in affected and "<" in affected, target


def test_no_top_level_vllm_import() -> None:
    tree = ast.parse(MODULE_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for imported in names:
            assert imported != "vllm" and not imported.startswith("vllm."), (
                f"the leaf imports {imported!r} at module level; it must stay "
                "importable without vLLM."
            )


def test_every_patch_call_carries_metadata() -> None:
    tree = ast.parse(MODULE_PATH.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "patch"
    ]
    assert len(calls) == 1
    for call in calls:
        keywords = {kw.arg for kw in call.keywords}
        missing = set(REQUIRED_METADATA) - keywords
        assert not missing, f"patch call at line {call.lineno} lacks {missing}"


def test_static_discovery_resolves_the_target() -> None:
    hooks, unresolved = scan_patch_file(MODULE_PATH)
    assert not unresolved
    assert [hook.target for hook in hooks] == [TARGET]
    assert all(hook.node is not None and not hook.additive for hook in hooks)


# ---------------------------------------------------------------------------
# Behaviour against stubs
# ---------------------------------------------------------------------------


def test_install_lands_target_and_reapplication_guard(
    monkeypatch: pytest.MonkeyPatch, leaf_module
) -> None:
    planning = _stub_planning(monkeypatch)
    original = inspect.getattr_static(
        planning, "_project_kv_cache_groups_to_worker"
    )

    registry_before = len(PATCH_REGISTRY)
    leaf_module.install()
    records = PATCH_REGISTRY[registry_before:]
    assert [record.target for record in records] == [TARGET]
    assert all(record.was_missing is False for record in records)
    assert all(record.affected_versions == SUPPORTED_RANGE for record in records)
    assert all(record.reason and record.remove_when for record in records)

    replacement = planning._project_kv_cache_groups_to_worker
    assert replacement is not original
    assert original_of(replacement, TARGET) is original

    leaf_module.install()  # the guard, not the double-patch error
    assert len(PATCH_REGISTRY) == registry_before + 1
    with pytest.raises(RuntimeError, match="already patched"):
        patch(
            "vllm.v1.core.kv_cache_planning",
            "_project_kv_cache_groups_to_worker",
            reason="test",
            affected_versions="test",
            remove_when="test",
        )(replacement)


def test_empty_groups_dropped_and_local_groups_unchanged(
    monkeypatch: pytest.MonkeyPatch, leaf_module
) -> None:
    planning = _stub_planning(monkeypatch)
    leaf_module.install()
    project = planning._project_kv_cache_groups_to_worker

    ple_group = NS(layer_names=["model.layers.1.ple"], kv_cache_spec="ple_spec")
    full_group = NS(
        layer_names=["model.layers.0", "model.layers.2"],
        kv_cache_spec="full_spec",
    )
    partial_group = NS(
        layer_names=["model.layers.0", "model.layers.1.ple"],
        kv_cache_spec="mixed_spec",
    )
    groups = [ple_group, full_group, partial_group]

    # A later PP stage owns none of the PLE group's layers.
    worker_spec = {"model.layers.0": object(), "model.layers.2": object()}
    projected = project(groups, worker_spec)
    assert [group.kv_cache_spec for group in projected] == [
        "full_spec",
        "mixed_spec",
    ]
    assert projected[0] is full_group
    assert projected[1].layer_names == ["model.layers.0"]

    # The stage that owns the PLE layer keeps every group.
    worker_spec = {
        "model.layers.0": object(),
        "model.layers.1.ple": object(),
        "model.layers.2": object(),
    }
    projected = project(groups, worker_spec)
    assert projected == groups

    # No local layers at all: an empty projection, not a list of empty groups.
    assert project(groups, {}) == []