# SPDX-License-Identifier: Apache-2.0
"""Tests for the packed-uint16 gather relabel (bugfix/pynccl_uint16_gather).

Two-tier design like ``test_qwen4_exp_ple_quant_scheme.py``:

* **Bare-CPU import + metadata shape.** The leaf imports no vLLM at module
  scope; ``METADATA`` and both ``patch`` calls carry the mandatory fields.
* **Behaviour against stubs.** ``install()`` runs against a throwaway
  stand-in of the upstream pynccl module whose dtype table rejects uint16
  exactly like NCCL's, asserting the gathers move double the bytes as uint8
  while every other dtype keeps upstream's count and label. No torch needed:
  the stub module supplies sentinel dtypes under the same global names the
  copied bodies resolve.
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
    / "pynccl_uint16_gather.py"
)
MODULE = "vllm.distributed.device_communicators.pynccl"
GATHER_TARGET = f"{MODULE}.PyNcclCommunicator.all_gather"
GATHERV_TARGET = f"{MODULE}.PyNcclCommunicator.all_gatherv"
REQUIRED_METADATA = ("reason", "affected_versions", "remove_when")

UINT16 = object()
UINT8 = object()
BF16 = object()


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
    name = f"_pynccl_u16_leaf_test_{id(object())}"
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


class _FakeTensor:
    def __init__(self, dtype, numel, ptr=1024, device="cuda:0"):
        self.dtype = dtype
        self.numel_value = numel
        self.ptr = ptr
        self.device = device
        self.shape = (numel,)

    def numel(self):
        return self.numel_value

    def data_ptr(self):
        return self.ptr

    def __getitem__(self, sl):
        start, stop = sl.start or 0, sl.stop
        return _FakeTensor(self.dtype, stop - start, self.ptr + start * 8)


class _RecordingNccl:
    def __init__(self):
        self.calls: list = []

    def ncclAllGather(self, send, recv, count, dtype, comm, stream):
        self.calls.append(("allgather", send, recv, count, dtype))

    def ncclBroadcast(self, send, recv, count, dtype, root, comm, stream):
        self.calls.append(("broadcast", send, recv, count, dtype, root))

    def ncclGroupStart(self):
        self.calls.append(("group_start",))

    def ncclGroupEnd(self):
        self.calls.append(("group_end",))


def _stub_pynccl(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    def from_torch(dtype):
        if dtype is UINT16:
            raise ValueError("Unsupported dtype torch.uint16")
        return dtype

    class PyNcclCommunicator:
        def all_gather(self, output_tensor, input_tensor, stream=None):
            if self.disabled:
                return
            assert input_tensor.device == self.device
            if stream is None:
                stream = current_stream()
            self.nccl.ncclAllGather(
                buffer_type(input_tensor.data_ptr()),
                buffer_type(output_tensor.data_ptr()),
                input_tensor.numel(),
                ncclDataTypeEnum.from_torch(input_tensor.dtype),
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )

        def all_gatherv(self, output_tensor, input_tensor, sizes, stream=None):
            if self.disabled:
                return
            assert input_tensor.device == self.device
            if stream is None:
                stream = current_stream()
            assert output_tensor.shape[0] == sum(sizes)
            split_offset = 0
            self.nccl.ncclGroupStart()
            for root, split_size in enumerate(sizes):
                dst_slice = output_tensor[split_offset : split_offset + split_size]
                self.nccl.ncclBroadcast(
                    buffer_type(input_tensor.data_ptr()),
                    buffer_type(dst_slice.data_ptr()),
                    dst_slice.numel(),
                    ncclDataTypeEnum.from_torch(input_tensor.dtype),
                    root,
                    self.comm,
                    cudaStream_t(stream.cuda_stream),
                )
                split_offset += split_size
            self.nccl.ncclGroupEnd()

    pynccl = types.ModuleType(MODULE)
    pynccl.PyNcclCommunicator = PyNcclCommunicator  # type: ignore[attr-defined]
    # Globals the copied bodies resolve through bind_body.
    pynccl.torch = NS(uint16=UINT16, uint8=UINT8)  # type: ignore[attr-defined]
    pynccl.buffer_type = lambda ptr: ptr  # type: ignore[attr-defined]
    pynccl.cudaStream_t = lambda raw: raw  # type: ignore[attr-defined]
    pynccl.current_stream = lambda: NS(cuda_stream=7)  # type: ignore[attr-defined]
    pynccl.ncclDataTypeEnum = NS(from_torch=staticmethod(from_torch))  # type: ignore[attr-defined]
    _register_chain(monkeypatch, MODULE, pynccl)

    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda name: NS(  # type: ignore[attr-defined]
        warning=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
    )
    _register_chain(monkeypatch, "vllm.logger", logger_mod)
    return pynccl


def _communicator(pynccl: types.ModuleType) -> tuple[object, _RecordingNccl]:
    nccl = _RecordingNccl()
    comm = pynccl.PyNcclCommunicator()
    comm.disabled = False
    comm.device = "cuda:0"
    comm.nccl = nccl
    comm.comm = 42
    return comm, nccl


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
        (GATHER_TARGET, leaf_module.REASON_GATHER, SUPPORTED_RANGE, leaf_module.REMOVE_WHEN),
        (GATHERV_TARGET, leaf_module.REASON_GATHERV, SUPPORTED_RANGE, leaf_module.REMOVE_WHEN),
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
    assert len(calls) == 2
    for call in calls:
        keywords = {kw.arg for kw in call.keywords}
        missing = set(REQUIRED_METADATA) - keywords
        assert not missing, f"patch call at line {call.lineno} lacks {missing}"


def test_static_discovery_resolves_the_targets() -> None:
    hooks, unresolved = scan_patch_file(MODULE_PATH)
    assert not unresolved
    assert [hook.target for hook in hooks] == [GATHER_TARGET, GATHERV_TARGET]
    assert all(hook.node is not None and not hook.additive for hook in hooks)


# ---------------------------------------------------------------------------
# Behaviour against stubs
# ---------------------------------------------------------------------------


def test_install_lands_both_targets_and_guard(
    monkeypatch: pytest.MonkeyPatch, leaf_module
) -> None:
    pynccl = _stub_pynccl(monkeypatch)
    cls = pynccl.PyNcclCommunicator
    original_gather = inspect.getattr_static(cls, "all_gather")
    original_gatherv = inspect.getattr_static(cls, "all_gatherv")

    registry_before = len(PATCH_REGISTRY)
    leaf_module.install()
    records = PATCH_REGISTRY[registry_before:]
    assert [record.target for record in records] == [GATHER_TARGET, GATHERV_TARGET]
    assert all(record.was_missing is False for record in records)
    assert all(record.affected_versions == SUPPORTED_RANGE for record in records)

    assert cls.all_gather is not original_gather
    assert cls.all_gatherv is not original_gatherv
    assert original_of(cls.all_gather, GATHER_TARGET) is original_gather
    assert original_of(cls.all_gatherv, GATHERV_TARGET) is original_gatherv

    leaf_module.install()  # the guard, not the double-patch error
    assert len(PATCH_REGISTRY) == registry_before + 2


def test_uint16_gather_moves_double_bytes_as_uint8(
    monkeypatch: pytest.MonkeyPatch, leaf_module
) -> None:
    pynccl = _stub_pynccl(monkeypatch)
    leaf_module.install()
    comm, nccl = _communicator(pynccl)

    # The stub dtype table rejects uint16 exactly like NCCL's.
    with pytest.raises(ValueError, match="uint16"):
        pynccl.ncclDataTypeEnum.from_torch(UINT16)
    src = _FakeTensor(UINT16, 128, ptr=2048)
    dst = _FakeTensor(UINT16, 1024, ptr=4096)
    comm.all_gather(dst, src)
    assert nccl.calls == [("allgather", 2048, 4096, 256, UINT8)]

    # Other dtypes keep upstream's count and label.
    nccl.calls.clear()
    src = _FakeTensor(BF16, 128, ptr=2048)
    dst = _FakeTensor(BF16, 1024, ptr=4096)
    comm.all_gather(dst, src)
    assert nccl.calls == [("allgather", 2048, 4096, 128, BF16)]


def test_uint16_gatherv_relabels_every_split(
    monkeypatch: pytest.MonkeyPatch, leaf_module
) -> None:
    pynccl = _stub_pynccl(monkeypatch)
    leaf_module.install()
    comm, nccl = _communicator(pynccl)

    src = _FakeTensor(UINT16, 64, ptr=2048)
    dst = _FakeTensor(UINT16, 192, ptr=4096)
    comm.all_gatherv(dst, src, [64, 128])
    assert nccl.calls == [
        ("group_start",),
        ("broadcast", 2048, 4096, 128, UINT8, 0),
        ("broadcast", 2048, 4096 + 64 * 8, 256, UINT8, 1),
        ("group_end",),
    ]

    nccl.calls.clear()
    src = _FakeTensor(BF16, 64, ptr=2048)
    dst = _FakeTensor(BF16, 192, ptr=4096)
    comm.all_gatherv(dst, src, [64, 128])
    assert nccl.calls == [
        ("group_start",),
        ("broadcast", 2048, 4096, 64, BF16, 0),
        ("broadcast", 2048, 4096 + 64 * 8, 128, BF16, 1),
        ("group_end",),
    ]


def test_disabled_communicator_still_noops(
    monkeypatch: pytest.MonkeyPatch, leaf_module
) -> None:
    pynccl = _stub_pynccl(monkeypatch)
    leaf_module.install()
    comm, nccl = _communicator(pynccl)
    comm.disabled = True
    src = _FakeTensor(UINT16, 128)
    dst = _FakeTensor(UINT16, 1024)
    comm.all_gather(dst, src)
    comm.all_gatherv(dst, src, [512, 512])
    assert nccl.calls == []