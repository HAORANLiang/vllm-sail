# SPDX-License-Identifier: Apache-2.0
"""Public lifecycle tests for the installed vLLM plugin entry point."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by Python 3.10 CI
    import tomli as tomllib

import vllm_sail
import vllm_sail.registry as registry_pkg
from vllm_sail import envs


def _module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


def test_ppu_allowlist_name_enables_the_complete_general_hook() -> None:
    """``VLLM_PLUGINS=ppu`` must not select only the platform half."""
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    entry_points = tomllib.loads(pyproject.read_text())["project"]["entry-points"]

    assert entry_points["vllm.platform_plugins"] == {"ppu": "vllm_sail:register"}
    assert entry_points["vllm.general_plugins"] == {
        "ppu": "vllm_sail:register_out_of_tree"
    }


def test_platform_hook_bootstraps_empty_vllm_before_flash_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    bootstrap = _module(
        "vllm_sail.native.bootstrap", install=lambda: events.append("native")
    )
    flash = _module(
        "vllm_sail.attention.flash_attn_shim", install=lambda: events.append("flash")
    )
    monkeypatch.setitem(sys.modules, bootstrap.__name__, bootstrap)
    monkeypatch.setitem(sys.modules, flash.__name__, flash)
    monkeypatch.setattr(vllm_sail, "_SHIMS_INSTALLED", False)

    assert vllm_sail.register() == "vllm_sail.platform.PPUPlatform"
    assert vllm_sail.register() == "vllm_sail.platform.PPUPlatform"

    assert events == ["native", "flash"]


@pytest.fixture
def general_hook(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    for name in dir(envs):
        monkeypatch.delenv(name, raising=False)
    events: list[str] = []

    compat = _module(
        "vllm_sail.compat",
        check_vllm_compatibility=lambda: events.append("compat"),
    )
    patch = _module("vllm_sail.patch", install=lambda: events.append("patch"))
    registry = _module("vllm_sail.registry", register=lambda: events.append("registry"))
    models = _module("vllm_sail.models", register_model=lambda: events.append("models"))
    profiling = _module(
        "vllm_sail.profiling", install=lambda: events.append("profiling")
    )
    layerwise = _module(
        "vllm_sail.profiling.layerwise", install=lambda: events.append("layerwise")
    )
    monkeypatch.setitem(sys.modules, layerwise.__name__, layerwise)
    ops = _module("vllm_sail.ops")

    for module in (compat, patch, registry, models, profiling, ops):
        monkeypatch.setitem(sys.modules, module.__name__, module)
        child = module.__name__.removeprefix("vllm_sail.")
        monkeypatch.setattr(vllm_sail, child, module, raising=False)

    monkeypatch.setattr(vllm_sail, "_PATCHES_APPLIED", False)
    monkeypatch.setattr(vllm_sail, "_REGISTRIES_APPLIED", False)
    monkeypatch.setattr(vllm_sail, "_MODELS_REGISTERED", False)
    return events


def test_general_hook_runs_the_complete_lifecycle_once(general_hook) -> None:
    """A cached registry import must not suppress backend registration."""
    # The module is deliberately present in sys.modules before the hook. The
    # hook must call its public register() function rather than relying on a
    # package-import side effect.
    assert sys.modules["vllm_sail.registry"] is vllm_sail.registry

    vllm_sail.register_out_of_tree()
    vllm_sail.register_out_of_tree()

    assert general_hook == [
        "compat",
        "patch",
        "registry",
        "layerwise",
        "profiling",
        "models",
        "compat",
    ]


def test_general_hook_keeps_step5_reads_lazy(monkeypatch, general_hook):
    name = "VLLM_SAIL_STEP5_DSA_SCORES"
    monkeypatch.setenv(name, "v3")

    def install():
        assert envs.VLLM_SAIL_STEP5_DSA_SCORES == "v3"
        monkeypatch.setenv(name, "v0")
        general_hook.append("patch")

    monkeypatch.setattr(vllm_sail.patch, "install", install)
    vllm_sail.register_out_of_tree()
    assert envs.VLLM_PPU_STEP5_DSA_SCORES == "v0"
    assert envs.snapshot()[name] == "v0"
    monkeypatch.setenv(name, "v1")
    vllm_sail.register_out_of_tree()
    assert envs.VLLM_PPU_STEP5_DSA_SCORES == "v1"
    assert envs.snapshot()[name] == "v1"
    assert general_hook.count("patch") == 1


def test_general_hook_does_not_eagerly_validate_unused_step5_settings(
    monkeypatch, general_hook
):
    name = "VLLM_SAIL_STEP5_DSA_SCORE_DECODE"
    monkeypatch.setenv(name, "v3")
    vllm_sail.register_out_of_tree()
    assert general_hook.count("patch") == 1
    assert vllm_sail._PATCHES_APPLIED
    with pytest.raises(ValueError, match=name):
        getattr(envs, name)
    monkeypatch.setenv(name, "v2")
    assert getattr(envs, name) == "v2"


def test_registry_registration_fails_terminally_after_partial_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mutated global registry must not be presented as safely retryable."""
    events: list[str] = []
    fail_linear = True

    def register_linear() -> None:
        events.append("linear")
        if fail_linear:
            raise RuntimeError("synthetic linear registration failure")

    children = {
        "tuned_configs": _module(
            "vllm_sail.registry.tuned_configs",
            register=lambda: events.append("configs"),
        ),
        "linear_kernels": _module(
            "vllm_sail.registry.linear_kernels", register=register_linear
        ),
        "quant_config": _module(
            "vllm_sail.registry.quant_config",
            register=lambda: events.append("quant"),
        ),
        "moe_backends": _module(
            "vllm_sail.registry.moe_backends",
            register=lambda: events.append("moe"),
        ),
    }
    for name, module in children.items():
        monkeypatch.setattr(registry_pkg, name, module, raising=False)
    monkeypatch.setattr(registry_pkg, "_registered", False)
    monkeypatch.setattr(registry_pkg, "_registration_error", None)

    with pytest.raises(RuntimeError, match="restart this process"):
        registry_pkg.register()

    fail_linear = False
    with pytest.raises(RuntimeError, match="previously failed"):
        registry_pkg.register()

    assert events == ["configs", "linear"]
