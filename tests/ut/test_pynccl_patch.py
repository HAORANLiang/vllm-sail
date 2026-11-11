# SPDX-License-Identifier: Apache-2.0
"""Deterministic optional-symbol coverage without torch, vLLM, or a device."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OPTIONAL = ("ncclCommSuspend", "ncclCommResume")


@pytest.fixture
def runtime(monkeypatch, patch_utils_module):
    modules = {}

    def add(name, **attrs):
        if name not in modules:
            module = types.ModuleType(name)
            module.__path__ = []
            modules[name] = module
            monkeypatch.setitem(sys.modules, name, module)
            parent, _, leaf = name.rpartition(".")
            if parent:
                setattr(add(parent), leaf, module)
        modules[name].__dict__.update(attrs)
        return modules[name]

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    add("vllm_sail.patch.utils", patch=patch_utils_module.patch)
    helpers = load("_pynccl_bodies", ROOT / "vllm_sail/patch/bodies.py")
    add("vllm_sail.patch.bodies", bind_body=helpers.bind_body)
    platform = types.SimpleNamespace(backend="ppu")
    platform.is_ppu = lambda: platform.backend == "ppu"
    platform.is_rocm = lambda: platform.backend == "rocm"
    add("vllm.platforms", current_platform=platform)
    libraries, opened, delegated, warnings = {}, [], [], []

    def cdll(path):
        opened.append(path)
        library = libraries[path]
        if isinstance(library, Exception):
            raise library
        return library

    logger = types.SimpleNamespace(
        error=lambda *args: None,
        warning_once=lambda *args: warnings.append(args),
    )

    class Library:
        exported_functions = [
            types.SimpleNamespace(name=name, restype=None, argtypes=[])
            for name in ("ncclAllReduce", *OPTIONAL, "ncclCommQueryProperties")
        ]
        path_to_library_cache = {}
        path_to_dict_mapping = {}

        def __init__(self, so_file=None):
            raise AssertionError("unpatched loader")

        def has_symbol(self, name):
            return name in self._funcs

    gathers = []

    class Communicator:
        def all_gather(self, output, input_, stream=None):
            gathers.append((output, input_, None, stream))

        def all_gatherv(self, output, input_, sizes, stream=None):
            gathers.append((output, input_, sizes, stream))

        def suspend(self):
            delegated.append(self)
            self._suspended = True

    wrapper = add(
        "vllm.distributed.device_communicators.pynccl_wrapper",
        NCCLLibrary=Library,
        ctypes=types.SimpleNamespace(CDLL=cdll),
        current_platform=platform,
        envs=types.SimpleNamespace(VLLM_USE_NCCL_SYMM_MEM=False),
        logger=logger,
        platform=types.SimpleNamespace(platform=lambda: "test"),
        find_nccl_library=lambda: "default.so",
    )
    add(
        "vllm.distributed.device_communicators.pynccl",
        PyNcclCommunicator=Communicator,
        NCCLLibrary=Library,
        logger=logger,
    )
    original_init, original_suspend = Library.__init__, Communicator.suspend
    load("_pynccl_patch", ROOT / "vllm_sail/patch/enhancement/pynccl.py")
    assert Library.__init__ is not original_init
    assert Communicator.suspend is not original_suspend
    assert len(patch_utils_module.PATCH_REGISTRY) == 4

    def library(path="default.so", symbols=OPTIONAL):
        libraries[path] = types.SimpleNamespace(
            **{name: types.SimpleNamespace() for name in ("ncclAllReduce", *symbols)}
        )
        return Library(path)

    return types.SimpleNamespace(
        Library=Library,
        Communicator=Communicator,
        platform=platform,
        libraries=libraries,
        opened=opened,
        delegated=delegated,
        warnings=warnings,
        library=library,
        wrapper=wrapper,
        gathers=gathers,
    )


@pytest.mark.parametrize("backend", ["ppu", "cuda", "rocm"])
@pytest.mark.parametrize("symbols", [(), OPTIONAL[:1], OPTIONAL[1:], OPTIONAL])
def test_optional_symbol_policy(runtime, backend, symbols):
    runtime.platform.backend = backend
    if backend == "cuda" and len(symbols) != 2:
        with pytest.raises(AttributeError, match="ncclComm"):
            runtime.library(symbols=symbols)
    else:
        library = runtime.library(symbols=symbols)
        assert library.has_symbol("ncclAllReduce")
        assert {name for name in OPTIONAL if library.has_symbol(name)} == set(symbols)


@pytest.mark.parametrize("symbols", [(), OPTIONAL[:1], OPTIONAL[1:], OPTIONAL])
def test_ppu_suspends_only_with_both_apis(runtime, symbols):
    communicator = runtime.Communicator()
    communicator.nccl = runtime.library(symbols=symbols)
    communicator.disabled = communicator._suspended = False
    communicator.suspend()
    assert bool(runtime.delegated) == (len(symbols) == 2)
    assert communicator._suspended == (len(symbols) == 2)
    assert bool(runtime.warnings) == (len(symbols) != 2)


def test_missing_required_symbol_and_library_load_errors_propagate(runtime):
    runtime.libraries["required.so"] = types.SimpleNamespace()
    with pytest.raises(AttributeError, match="ncclAllReduce"):
        runtime.Library("required.so")
    runtime.libraries["broken.so"] = OSError("invalid ELF header")
    with pytest.raises(OSError, match="invalid ELF header"):
        runtime.Library("broken.so")


def test_repeated_and_distinct_libraries_keep_separate_symbol_caches(runtime):
    alias = runtime.Library
    absent = runtime.library("absent.so", symbols=())
    cached = alias("absent.so")
    present = runtime.library("present.so")
    assert absent._funcs is cached._funcs
    assert present._funcs is not absent._funcs
    assert not absent.has_symbol("ncclCommSuspend")
    assert present.has_symbol("ncclCommSuspend")
    assert runtime.opened == ["absent.so", "present.so"]
    assert [f.name for f in alias.exported_functions] == [
        "ncclAllReduce",
        *OPTIONAL,
        "ncclCommQueryProperties",
    ]


@pytest.mark.parametrize("backend", ["ppu", "cuda"])
@pytest.mark.parametrize("variable", [False, True])
@pytest.mark.parametrize("dtype", ["uint16", "float32"])
def test_gather_transports_packed_scales_without_casting(
    runtime, monkeypatch, backend, variable, dtype
):
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(uint16="uint16", bfloat16="bf16")
    )

    class Tensor:
        def __init__(self, dtype, storage):
            self.dtype, self.storage = dtype, storage

        def view(self, dtype):
            return Tensor(dtype, self.storage)

    runtime.platform.backend = backend
    source = Tensor(dtype, [0, 32768, 65535])
    target = Tensor(dtype, [0] * 6)
    communicator = runtime.Communicator()
    if variable:
        communicator.all_gatherv(target, source, [3, 3], stream="stream")
    else:
        communicator.all_gather(target, source, stream="stream")
    output, input_, sizes, stream = runtime.gathers[0]
    expected_dtype = "bf16" if backend == "ppu" and dtype == "uint16" else dtype
    assert output.dtype == input_.dtype == expected_dtype
    assert input_.storage is source.storage and output.storage is target.storage
    assert source.dtype == target.dtype == dtype
    assert sizes == ([3, 3] if variable else None)
    assert stream == "stream"
