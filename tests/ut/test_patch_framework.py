# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the patch contract without importing vLLM or torch."""

from __future__ import annotations

import inspect
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

import pytest


@contextmanager
def throwaway_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    name = f"_ppu_patch_target_{uuid4().hex}"
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    yield module


def test_module_fixture_uses_imported_package_python(modules):
    from pathlib import Path

    from tests.support.modules import load_patch
    from tests.support.source import PACKAGE_ROOT

    # The simulated module tree must execute the wheel's Python too, not just
    # check the paths of a normal import.
    bodies = sys.modules["vllm_sail.patch.bodies"]
    assert (
        Path(bodies.bind_body.__code__.co_filename)
        .resolve()
        .is_relative_to(PACKAGE_ROOT)
    )
    package = modules("vllm_sail")
    assert package.__path__ == [str(PACKAGE_ROOT / "vllm_sail")]
    module = load_patch("vllm_sail/envs.py")
    assert Path(module.__file__).resolve() == PACKAGE_ROOT / "vllm_sail/envs.py"
    assert module.VLLM_SAIL_STEP5_DSA_PREFILL_BACKEND == "flashinfer"


def metadata() -> dict[str, str]:
    return {
        "reason": "exercise the patch contract",
        "affected_versions": ">=0.27.0,<0.28.0",
        "remove_when": "the upstream extension point exists",
    }


def test_module_attribute_patch_and_registry(
    monkeypatch: pytest.MonkeyPatch, patch_utils_module: types.ModuleType
) -> None:
    with throwaway_module(monkeypatch) as target:

        def original() -> str:
            return "original"

        target.operation = original

        @patch_utils_module.patch(target.__name__, "operation", **metadata())
        def replacement() -> str:
            return "replacement"

        assert target.operation is replacement
        assert target.operation() == "replacement"
        record = patch_utils_module.PATCH_REGISTRY[-1]
        assert record.target == f"{target.__name__}.operation"
        assert record.kind == "module"
        assert record.was_missing is False
        assert patch_utils_module.original_of(replacement, record.target) is original


def test_plain_class_method_patch(
    monkeypatch: pytest.MonkeyPatch, patch_utils_module: types.ModuleType
) -> None:
    with throwaway_module(monkeypatch) as target:

        class Target:
            def operation(self) -> str:
                return "original"

        target.Target = Target

        @patch_utils_module.patch(target.__name__, "Target.operation", **metadata())
        def replacement(self: Any) -> str:
            del self
            return "replacement"

        assert Target().operation() == "replacement"
        assert inspect.getattr_static(Target, "operation") is replacement
        assert patch_utils_module.PATCH_REGISTRY[-1].kind == "class"


def test_class_descriptors_are_preserved(
    monkeypatch: pytest.MonkeyPatch, patch_utils_module: types.ModuleType
) -> None:
    with throwaway_module(monkeypatch) as target:

        class Target:
            @staticmethod
            def static() -> str:
                return "old-static"

            @classmethod
            def class_method(cls) -> str:
                return cls.__name__

            @property
            def value(self) -> str:
                return "old-property"

        target.Target = Target

        @patch_utils_module.patch(target.__name__, "Target.static", **metadata())
        @staticmethod
        def new_static() -> str:
            return "new-static"

        @patch_utils_module.patch(target.__name__, "Target.class_method", **metadata())
        @classmethod
        def new_class_method(cls: type[Any]) -> str:
            return f"new-{cls.__name__}"

        @patch_utils_module.patch(target.__name__, "Target.value", **metadata())
        @property
        def new_value(self: Any) -> str:
            del self
            return "new-property"

        assert isinstance(inspect.getattr_static(Target, "static"), staticmethod)
        assert isinstance(inspect.getattr_static(Target, "class_method"), classmethod)
        assert isinstance(inspect.getattr_static(Target, "value"), property)
        assert Target.static() == "new-static"
        assert Target.class_method() == "new-Target"
        assert Target().value == "new-property"


def test_allow_missing_adds_attribute_and_original_of_rejects_additive_patch(
    monkeypatch: pytest.MonkeyPatch, patch_utils_module: types.ModuleType
) -> None:
    with throwaway_module(monkeypatch) as target:

        @patch_utils_module.patch(
            target.__name__, "new_operation", allow_missing=True, **metadata()
        )
        def replacement() -> str:
            return "added"

        full_target = f"{target.__name__}.new_operation"
        assert target.new_operation is replacement
        assert patch_utils_module.PATCH_REGISTRY[-1].was_missing is True
        with pytest.raises(KeyError, match="did not previously exist"):
            patch_utils_module.original_of(replacement, full_target)


def test_double_patch_raises(
    monkeypatch: pytest.MonkeyPatch, patch_utils_module: types.ModuleType
) -> None:
    with throwaway_module(monkeypatch) as target:
        target.operation = lambda: "original"

        @patch_utils_module.patch(target.__name__, "operation", **metadata())
        def first() -> str:
            return "first"

        with pytest.raises(RuntimeError, match="already patched"):

            @patch_utils_module.patch(target.__name__, "operation", **metadata())
            def second() -> str:
                return "second"


def test_additive_patch_shadows_inherited_patched_attribute(
    monkeypatch: pytest.MonkeyPatch, patch_utils_module: types.ModuleType
) -> None:
    """The SparseAttnIndexer/CustomOp.forward_ppu collision, generalised.

    An additive patch on a subclass must not trip the double-patch guard just
    because the subclass inherits an attribute that a different patch already
    installed on the base class; it shadows it in the subclass namespace.
    """
    with throwaway_module(monkeypatch) as target:

        class Base:
            pass

        class Child(Base):
            pass

        target.Base = Base
        target.Child = Child

        @patch_utils_module.patch(
            target.__name__, "Base.forward_ppu", allow_missing=True, **metadata()
        )
        def base_forward(self: Any) -> str:
            del self
            return "base"

        @patch_utils_module.patch(
            target.__name__, "Child.forward_ppu", allow_missing=True, **metadata()
        )
        def child_forward(self: Any) -> str:
            del self
            return "child"

        assert Base.forward_ppu is base_forward
        assert Child.forward_ppu is child_forward
        assert Child().forward_ppu() == "child"
        assert Base().forward_ppu() == "base"
        record = patch_utils_module.PATCH_REGISTRY[-1]
        assert record.was_missing is True
        # Reapplying the same additive patch on the subclass still raises:
        # the attribute now exists on Child itself and carries the marker.
        with pytest.raises(RuntimeError, match="already patched"):

            @patch_utils_module.patch(
                target.__name__, "Child.forward_ppu", allow_missing=True, **metadata()
            )
            def duplicate(self: Any) -> str:
                del self
                return "duplicate"


def test_missing_target_without_allow_missing_raises(
    monkeypatch: pytest.MonkeyPatch, patch_utils_module: types.ModuleType
) -> None:
    with throwaway_module(monkeypatch) as target:
        with pytest.raises(AttributeError, match="does not exist"):

            @patch_utils_module.patch(target.__name__, "missing", **metadata())
            def replacement() -> None:
                return None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reason", ""),
        ("reason", None),
        ("affected_versions", "   "),
        ("affected_versions", None),
        ("remove_when", ""),
        ("remove_when", None),
    ],
)
def test_missing_or_blank_metadata_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
    patch_utils_module: types.ModuleType,
    field: str,
    value: str | None,
) -> None:
    with throwaway_module(monkeypatch) as target:
        target.operation = lambda: "original"
        values: dict[str, Any] = metadata()
        values[field] = value
        with pytest.raises(ValueError, match=field):

            @patch_utils_module.patch(target.__name__, "operation", **values)
            def replacement() -> None:
                return None
