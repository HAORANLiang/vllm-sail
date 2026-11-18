# SPDX-License-Identifier: Apache-2.0
"""Bare-CPU behavioural regression for the MiniMax M3 ops quack-stub preload.

``patch/enhancement/models/minimax_m3_ops.py`` registers a module-style
``PatchRecord`` (not ``@patch``) because the CuTeDSL kernel cannot be imported
without a temporary ``quack`` stub. These tests exercise the preload mechanism
against stubs: the kernel imports under the stub, the stub is removed afterwards,
``make_fake_tensor`` is rebound to a loud error, the registration is idempotent,
and non-PPU is a no-op. Runs with no vLLM and no torch.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from vllm_sail.patch.utils import PATCH_REGISTRY

MODELS_DIR = (
    Path(__file__).parents[2] / "vllm_sail" / "patch" / "enhancement" / "models"
)
_KERNEL = "vllm.models.minimax_m3.nvidia.ops.index_decode_score"
_STUB_KEYS = (_KERNEL, "quack", "quack.compile_utils")


class _Platform:
    ppu = True

    @classmethod
    def is_ppu(cls) -> bool:
        return cls.ppu


_CREATED: list[str] = []


@pytest.fixture(autouse=True)
def _cleanup() -> None:
    before = len(PATCH_REGISTRY)
    saved = {k: sys.modules.get(k) for k in _STUB_KEYS}
    _Platform.ppu = True
    yield
    del PATCH_REGISTRY[before:]
    for key, value in saved.items():
        if value is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = value
    for name in _CREATED:
        sys.modules.pop(name, None)
    _CREATED.clear()


def _register_chain(monkeypatch: pytest.MonkeyPatch, dotted: str, leaf) -> None:
    parts = dotted.split(".")
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
    modules[dotted] = leaf
    monkeypatch.setitem(sys.modules, dotted, leaf)
    for parent_name, child_name in zip(chain, chain[1:], strict=False):
        setattr(modules[parent_name], child_name.rpartition(".")[2], modules[child_name])


def _load(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = _Platform  # type: ignore[attr-defined]
    _register_chain(monkeypatch, "vllm.platforms", platforms)
    name = f"_minimax_ops_test_{id(object())}"
    spec = importlib.util.spec_from_file_location(name, MODELS_DIR / "minimax_m3_ops.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    _CREATED.append(name)
    spec.loader.exec_module(module)
    return module


def test_install_preloads_under_stub_then_rebinds_loudly(monkeypatch) -> None:
    module = _load(monkeypatch)
    _Platform.ppu = True
    sys.modules.pop(_KERNEL, None)

    real_import = importlib.import_module
    created: dict[str, types.ModuleType] = {}

    def fake_import(name, *args, **kwargs):
        if name == _KERNEL:
            # The temporary quack stub must be in place during the preload.
            assert "quack" in sys.modules
            assert "quack.compile_utils" in sys.modules
            kernel = types.ModuleType(_KERNEL)
            kernel.make_fake_tensor = lambda *a, **k: "original"  # type: ignore[attr-defined]
            sys.modules[_KERNEL] = kernel
            created["kernel"] = kernel
            return kernel
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    before = len(PATCH_REGISTRY)
    module.install()

    kernel = created["kernel"]
    with pytest.raises(NotImplementedError, match="not supported on PPU"):
        kernel.make_fake_tensor(object())
    # The temporary stub is removed once the kernel module is loaded.
    assert "quack" not in sys.modules
    assert "quack.compile_utils" not in sys.modules
    assert len(PATCH_REGISTRY) == before + 1
    record = PATCH_REGISTRY[-1]
    assert record.target == module._TARGET
    assert record.kind == "module"
    assert record.reason.strip() and record.remove_when.strip()

    # Idempotent: the kernel is already loaded, so a second install is a no-op.
    module.install()
    assert len(PATCH_REGISTRY) == before + 1


def test_install_is_noop_on_non_ppu(monkeypatch) -> None:
    module = _load(monkeypatch)
    _Platform.ppu = False
    sys.modules.pop(_KERNEL, None)

    real_import = importlib.import_module
    called: list[str] = []

    def fake_import(name, *args, **kwargs):
        called.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    before = len(PATCH_REGISTRY)
    module.install()
    assert _KERNEL not in called
    assert len(PATCH_REGISTRY) == before
    assert "quack" not in sys.modules
    _Platform.ppu = True