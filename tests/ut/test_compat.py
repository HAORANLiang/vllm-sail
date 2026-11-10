# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm_sail.compat.

``vllm_sail.compat`` imports ``vllm.logger``, which is not installed in the
CPU-only test environment, so a minimal stub is installed for the duration of
each test rather than at collection time. The stub is a real module object with
exactly the one attribute compat needs — not a MagicMock — so an unexpected new
dependency on vLLM surfaces as an AttributeError here instead of being silently
absorbed.
"""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
import types
from collections.abc import Iterator

import pytest


@pytest.fixture
def compat(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """Import ``vllm_sail.compat`` against a stubbed ``vllm.logger``."""
    vllm = sys.modules.get("vllm")
    if vllm is None:
        vllm = types.ModuleType("vllm")
        vllm.__path__ = []  # mark as a package so submodule imports resolve
        monkeypatch.setitem(sys.modules, "vllm", vllm)

    if "vllm.logger" not in sys.modules:
        logger_mod = types.ModuleType("vllm.logger")
        logger_mod.init_logger = logging.getLogger
        monkeypatch.setitem(sys.modules, "vllm.logger", logger_mod)

    monkeypatch.delitem(sys.modules, "vllm_sail.compat", raising=False)
    module = importlib.import_module("vllm_sail.compat")
    # The range-policy tests below exercise release-tag manifests. Exact main
    # pins have a separate commit-identity contract.
    from vllm_sail.native import manifest

    monkeypatch.setattr(
        manifest, "load", lambda: types.SimpleNamespace(upstream_ref="v0.33.0")
    )
    yield module
    # Leave no cached copy behind: the next test may want a different stub.
    sys.modules.pop("vllm_sail.compat", None)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.31.0", (0, 31, 0)),
        ("v0.31.0", (0, 31, 0)),
        ("0.31", (0, 31)),
        ("0.31.1+cu128", (0, 31, 1)),
        ("0.31.0rc1", (0, 31, 0)),
        # Numeric parsing stops at the first non-numeric chunk, so a setuptools-scm
        # dev version yields a misleadingly low key -- which is exactly why
        # is_dev_build() exists.
        ("0.1.dev1+g4bdc8a788", (0, 1)),
    ],
)
def test_version_key_parses(compat, version: str, expected: tuple[int, ...]) -> None:
    assert compat._version_key(version) == expected


@pytest.mark.parametrize(
    "version",
    [
        "0.1.dev1+g4bdc8a788",
        "0.33.0.dev5",
        "0.31.1+g1234567",
        "0.31.0rc2.dev3+gabcdef0",
    ],
)
def test_is_dev_build(compat, version: str) -> None:
    assert compat.is_dev_build(version)


@pytest.mark.parametrize("version", ["0.31.0", "v0.31.1", "0.31.0rc1", "0.31.1+cu128"])
def test_is_not_dev_build(compat, version: str) -> None:
    assert not compat.is_dev_build(version)


def test_check_rejects_unverifiable_shallow_clone_dev_build(
    compat, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shallow git clone reports 0.1.devN+g<hash>, which parses to (0, 1).

    That prefix cannot distinguish a supported v0.31 checkout from an unsupported
    checkout, so the caller must provide an authoritative VLLM_VERSION override.
    """
    monkeypatch.delenv("VLLM_VERSION", raising=False)
    monkeypatch.setattr(compat, "installed_vllm_version", lambda: "0.1.dev1+g4bdc8a788")
    with pytest.raises(RuntimeError, match="cannot verify compatibility"):
        compat.check_vllm_compatibility(force=True)


def test_check_accepts_supported_dev_build_with_meaningful_version(
    compat, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A v0.33-based source build remains usable without an override."""
    monkeypatch.delenv("VLLM_VERSION", raising=False)
    monkeypatch.setattr(
        compat, "installed_vllm_version", lambda: "0.33.1.dev5+g1234567"
    )
    with caplog.at_level(logging.WARNING):
        compat.check_vllm_compatibility(force=True)
    assert "development build" in caplog.text


@pytest.mark.parametrize(
    "version",
    [
        "0.31.1.dev5+gabcdef0",
        "0.34.1.dev58+g40cf03ae64",
        "0.34.0.dev1+gabcdef0",
    ],
)
def test_check_rejects_unsupported_dev_build_without_override(
    compat, monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    monkeypatch.delenv("VLLM_VERSION", raising=False)
    monkeypatch.setattr(compat, "installed_vllm_version", lambda: version)
    with pytest.raises(RuntimeError, match="requires vLLM"):
        compat.check_vllm_compatibility(force=True)


def test_explicit_vllm_version_is_taken_literally(
    compat, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the user sets VLLM_VERSION, they asked to be judged on it."""
    monkeypatch.setenv("VLLM_VERSION", "0.26.0.dev1")
    with pytest.raises(RuntimeError, match="requires vLLM"):
        compat.check_vllm_compatibility(force=True)


@pytest.mark.parametrize("version", ["unknown", "", "0", "dev", "abc"])
def test_version_key_unparseable(compat, version: str) -> None:
    """Unparseable versions return None so callers can warn instead of failing."""
    assert compat._version_key(version) is None


def test_installed_version_env_override(
    compat, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_VERSION", " 0.31.0 ")
    assert compat.installed_vllm_version() == "0.31.0"


def test_vllm_version_is(compat, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_VERSION", "0.31.0+cu128")
    assert compat.vllm_version_is("0.31.0")
    assert compat.vllm_version_is("v0.31.0")
    assert not compat.vllm_version_is("0.31.1")


def test_vllm_version_at_least(compat, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_VERSION", "0.31.1")
    assert compat.vllm_version_at_least("0.31.0")
    assert compat.vllm_version_at_least("0.31.1")
    assert not compat.vllm_version_at_least("0.33.0")


def test_vllm_version_at_least_unparseable_is_false(
    compat, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unjudgeable version must not be reported as satisfying a bound."""
    monkeypatch.setenv("VLLM_VERSION", "unknown")
    assert not compat.vllm_version_at_least("0.31.0")


@pytest.mark.parametrize("version", ["0.33.0", "0.33.9", "0.33.1+cu128"])
def test_check_accepts_supported_range(
    compat, monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    monkeypatch.setenv("VLLM_VERSION", version)
    compat.check_vllm_compatibility(force=True)  # must not raise


@pytest.mark.parametrize(
    "version", ["0.27.0", "0.30.0", "0.31.0", "0.32.0", "0.34.0", "1.0.0"]
)
def test_check_rejects_out_of_range(
    compat, monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    monkeypatch.setenv("VLLM_VERSION", version)
    with pytest.raises(RuntimeError, match="requires vLLM"):
        compat.check_vllm_compatibility(force=True)


def test_check_rejects_unparseable(compat, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unjudgeable version must fail closed before patches are installed."""
    monkeypatch.setenv("VLLM_VERSION", "dev")
    with pytest.raises(RuntimeError, match="cannot verify compatibility"):
        compat.check_vllm_compatibility(force=True)


def test_failed_check_is_not_cached(compat, monkeypatch: pytest.MonkeyPatch) -> None:
    """A caught failure must not make the next no-force check skip validation."""
    monkeypatch.setenv("VLLM_VERSION", "0.34.0")
    with pytest.raises(RuntimeError, match="requires vLLM"):
        compat.check_vllm_compatibility(force=True)

    monkeypatch.setenv("VLLM_VERSION", "0.33.0")
    compat.check_vllm_compatibility()
    assert compat._checked is True


def test_check_is_cached_without_force(compat, monkeypatch: pytest.MonkeyPatch) -> None:
    """The check runs once per process; register_out_of_tree relies on that."""
    monkeypatch.setenv("VLLM_VERSION", "0.33.0")
    compat.check_vllm_compatibility(force=True)

    # Now an unsupported version would raise -- but the cached flag short-circuits.
    monkeypatch.setenv("VLLM_VERSION", "0.99.0")
    compat.check_vllm_compatibility()

    with pytest.raises(RuntimeError):
        compat.check_vllm_compatibility(force=True)
    assert compat._checked is False

    # A failed forced refresh must not leave the old successful result cached.
    monkeypatch.setenv("VLLM_VERSION", "0.33.0")
    compat.check_vllm_compatibility()
    assert compat._checked is True


PIN = "528772a4bf589fd52eec40bc8ca79c84d99206b5"


@pytest.mark.parametrize("dirty", [False, True])
def test_exact_pin_accepts_generated_scm_version(compat, monkeypatch, tmp_path, dirty):
    """Exercise upstream's full-hash SCM command, including dirty build inputs."""
    from setuptools_scm import get_version

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(tmp_path), *args], text=True
        ).strip()

    git("init", "-q")
    git("config", "user.name", "SCM test")
    git("config", "user.email", "scm@example.com")
    source = tmp_path / "source.py"
    source.write_text("initial\n")
    git("add", ".")
    git("-c", "commit.gpgsign=false", "commit", "-qm", "initial")
    git("tag", "v0.33.0")
    source.write_text("checkpoint\n")
    git("add", ".")
    git("-c", "commit.gpgsign=false", "commit", "-qm", "checkpoint")
    pin = git("rev-parse", "HEAD")
    if dirty:
        source.write_text("prepared build inputs\n")
    for name in (
        "VLLM_VERSION",
        "SETUPTOOLS_SCM_PRETEND_VERSION",
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM",
    ):
        monkeypatch.delenv(name, raising=False)
    version = get_version(
        root=str(tmp_path),
        git_describe_command=[
            "git",
            "describe",
            "--dirty",
            "--tags",
            "--long",
            "--abbrev=40",
            "--match",
            "v[0-9]*",
        ],
    )
    monkeypatch.setattr(compat, "_verified_upstream_commit", lambda: pin)
    monkeypatch.setattr(compat, "installed_vllm_version", lambda: version)
    compat.check_vllm_compatibility(force=True)
    assert compat._checked


def test_exact_main_pin_reads_the_packaged_kernel_metadata(compat, monkeypatch):
    from vllm_sail.native import manifest

    monkeypatch.setattr(
        manifest, "load", lambda: types.SimpleNamespace(upstream_ref=PIN)
    )
    assert compat._verified_upstream_commit() == PIN


def test_exact_main_pin_accepts_its_real_scm_version(compat, monkeypatch):
    monkeypatch.delenv("VLLM_VERSION", raising=False)
    monkeypatch.setattr(compat, "_verified_upstream_commit", lambda: PIN)
    monkeypatch.setattr(
        compat,
        "installed_vllm_version",
        lambda: f"0.30.1rc1.dev672+g{PIN}.d20261024.empty",
    )
    compat.check_vllm_compatibility(force=True)
    assert compat._checked


@pytest.mark.parametrize(
    "version",
    ["0.31.0", "0.31.1.dev1+g528772a", "0.31.1.dev1+g" + "a" * 40],
)
def test_exact_main_pin_rejects_missing_or_different_commit(
    compat, monkeypatch, version
):
    monkeypatch.delenv("VLLM_VERSION", raising=False)
    monkeypatch.setattr(compat, "_verified_upstream_commit", lambda: PIN)
    monkeypatch.setattr(compat, "installed_vllm_version", lambda: version)
    with pytest.raises(RuntimeError, match="requires verified vLLM commit"):
        compat.check_vllm_compatibility(force=True)
    assert not compat._checked


def test_exact_main_pin_cannot_be_replaced_with_a_numeric_override(compat, monkeypatch):
    monkeypatch.setattr(compat, "_verified_upstream_commit", lambda: PIN)
    monkeypatch.setenv("VLLM_VERSION", "0.31.0")
    with pytest.raises(RuntimeError, match="requires verified vLLM commit"):
        compat.check_vllm_compatibility(force=True)
