# SPDX-License-Identifier: Apache-2.0
"""Bare-CPU behavioural regression for the Kimi-K2.5 processor flag shim.

``patch/enhancement/models/kimi_k25_processor.py`` shims
``image_processor.preprocess`` for the duration of ``KimiK25Processor.__call__``
to drop ``do_rescale``/``do_normalize`` flags the checkpoint processor rejects.
These tests exercise the shim against a stub processor: rejected flags are
dropped and the call succeeds, unrelated TypeErrors propagate, accepted flags
pass through, and the instance's ``preprocess`` is restored afterwards. Runs
with no vLLM and no torch.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from vllm_sail.patch.utils import PATCH_REGISTRY

MODELS_DIR = (
    Path(__file__).parents[2] / "vllm_sail" / "patch" / "enhancement" / "models"
)
_TARGET_MODULE = "vllm.transformers_utils.processors.kimi_k25"
_CREATED: list[str] = []


@pytest.fixture(autouse=True)
def _cleanup() -> None:
    before = len(PATCH_REGISTRY)
    yield
    del PATCH_REGISTRY[before:]
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


def _load(monkeypatch: pytest.MonkeyPatch):
    class FakeImageProcessor:
        def preprocess(self, medias, return_tensors=None, **kwargs):
            rejected = [k for k in ("do_rescale", "do_normalize") if k in kwargs]
            if rejected:
                raise TypeError(
                    "preprocess() got an unexpected keyword argument "
                    f"'{rejected[0]}'"
                )
            return ("processed", medias, return_tensors)

    class KimiK25Processor:
        def __init__(self):
            self.image_processor = FakeImageProcessor()

        def __call__(self, *args, **kwargs):
            return self.image_processor.preprocess(*args, **kwargs)

    stub = types.ModuleType(_TARGET_MODULE)
    stub.KimiK25Processor = KimiK25Processor  # type: ignore[attr-defined]
    _register_chain(monkeypatch, _TARGET_MODULE, stub)

    name = f"_kimi_k25_proc_test_{id(object())}"
    spec = importlib.util.spec_from_file_location(
        name, MODELS_DIR / "kimi_k25_processor.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    _CREATED.append(name)
    spec.loader.exec_module(module)
    return module, stub


def test_call_drops_rejected_flags_and_restores(monkeypatch) -> None:
    _, stub = _load(monkeypatch)
    processor = stub.KimiK25Processor()
    result = processor(
        ["img"], return_tensors="pt", do_rescale=False, do_normalize=False
    )
    assert result == ("processed", ["img"], "pt")
    # The shim was an instance attribute; it is removed afterwards so the class
    # method is used again (the processor had no own preprocess before the call).
    assert "preprocess" not in vars(processor.image_processor)


def test_call_reraises_unrelated_typeerror_and_restores_own(monkeypatch) -> None:
    _, stub = _load(monkeypatch)
    processor = stub.KimiK25Processor()

    def boom(*args, **kwargs):
        raise TypeError("something else entirely")

    processor.image_processor.preprocess = boom
    with pytest.raises(TypeError, match="something else entirely"):
        processor(["img"], do_rescale=False)
    # An own preprocess is restored exactly, not deleted.
    assert processor.image_processor.preprocess is boom


def test_call_passes_accepted_flags_through(monkeypatch) -> None:
    _, stub = _load(monkeypatch)
    processor = stub.KimiK25Processor()

    def accept(*args, **kwargs):
        return ("ok", kwargs)

    processor.image_processor.preprocess = accept
    result = processor(["img"], do_rescale=False, do_normalize=False)
    assert result == ("ok", {"do_rescale": False, "do_normalize": False})
    assert processor.image_processor.preprocess is accept