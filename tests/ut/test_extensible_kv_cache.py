# SPDX-License-Identifier: Apache-2.0
"""Tests for the PPU extensible-KV-cache default (enhancement/extensible_kv_cache).

Two tiers:

* **Static contract.** ``scan_patch_file`` discovers the single replacement
  and its mandatory metadata without importing vLLM.
* **Behaviour against stubs.** The module is loaded against a throwaway
  ``vllm.config.vllm`` / ``vllm.platforms`` stand-in; the wrapper must force
  an unset flag to False on PPU only, leave explicit choices and non-PPU
  platforms untouched, and delegate to upstream exactly once.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tests.support.patch_inventory import scan_patch_file
from tests.support.upstream import SUPPORTED_RANGE
from vllm_sail.patch.utils import PATCH_REGISTRY, original_of

MODULE_PATH = (
    Path(__file__).parents[2]
    / "vllm_sail"
    / "patch"
    / "enhancement"
    / "extensible_kv_cache.py"
)
TARGET = "vllm.config.vllm.VllmConfig._resolve_extensible_kv_cache"


def test_static_contract() -> None:
    hooks, unresolved = scan_patch_file(MODULE_PATH)
    assert not unresolved
    assert [hook.target for hook in hooks] == [TARGET]


def _register_chain(
    monkeypatch: pytest.MonkeyPatch, name: str, module: types.ModuleType
) -> None:
    parts = name.split(".")
    chain = [".".join(parts[: i + 1]) for i in range(len(parts))]
    modules: dict[str, types.ModuleType] = {}
    for parent in chain[:-1]:
        existing = sys.modules.get(parent)
        modules[parent] = (
            existing
            if isinstance(existing, types.ModuleType)
            else types.ModuleType(parent)
        )
        monkeypatch.setitem(sys.modules, parent, modules[parent])
    modules[name] = module
    monkeypatch.setitem(sys.modules, name, module)
    for parent_name, child_name in zip(chain, chain[1:], strict=False):
        monkeypatch.setattr(
            modules[parent_name],
            child_name.rpartition(".")[2],
            modules[child_name],
            raising=False,
        )


@pytest.fixture()
def stubbed(monkeypatch: pytest.MonkeyPatch):
    """Load a fresh module instance against stub upstream modules."""
    calls: list = []

    class VllmConfig:
        def _resolve_extensible_kv_cache(self):
            calls.append(self.cache_config.enable_extensible_kv_cache)

    config_vllm = types.ModuleType("vllm.config.vllm")
    config_vllm.VllmConfig = VllmConfig  # type: ignore[attr-defined]
    _register_chain(monkeypatch, "vllm.config.vllm", config_vllm)

    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = NS(is_ppu=lambda: True)  # type: ignore[attr-defined]
    _register_chain(monkeypatch, "vllm.platforms", platforms)

    name = f"_extensible_kv_cache_test_{id(object())}"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    registry_before = len(PATCH_REGISTRY)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module, config_vllm, platforms, calls
    finally:
        del PATCH_REGISTRY[registry_before:]
        sys.modules.pop(name, None)


@pytest.mark.parametrize(
    ("is_ppu", "requested", "expected"),
    [
        (True, None, False),
        (True, True, True),
        (True, False, False),
        (False, None, None),
        (False, True, True),
        (False, False, False),
    ],
)
def test_default_resolution(stubbed, is_ppu, requested, expected) -> None:
    _, config_vllm, platforms, calls = stubbed
    platforms.current_platform.is_ppu = lambda: is_ppu
    cfg = NS(cache_config=NS(enable_extensible_kv_cache=requested))
    config_vllm.VllmConfig._resolve_extensible_kv_cache(cfg)
    assert cfg.cache_config.enable_extensible_kv_cache == expected
    # Upstream sees the settled flag and runs exactly once.
    assert calls == [expected]


def test_registry_records_original(stubbed) -> None:
    module, config_vllm, _, _ = stubbed
    record = PATCH_REGISTRY[-1]
    assert record.target == TARGET
    assert record.reason and record.remove_when
    assert record.affected_versions == SUPPORTED_RANGE
    replacement = config_vllm.VllmConfig._resolve_extensible_kv_cache
    assert original_of(replacement, TARGET) is module._upstream_resolve
