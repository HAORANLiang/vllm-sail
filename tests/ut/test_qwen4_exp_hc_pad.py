# SPDX-License-Identifier: Apache-2.0
"""Tests for the HC padding-shard gate (bugfix/qwen4_exp_hc_pad).

Two-tier design like ``test_pynccl_uint16_gather.py``:

* **Bare-CPU import + metadata shape.** The leaf imports no vLLM at module
  scope; ``METADATA`` and the ``patch`` call carry the mandatory fields.
* **Behaviour against stubs.** ``install()`` runs against a throwaway
  stand-in of the upstream hyperconnection module whose merged linear records
  the output sizes it is built with, asserting the padding shard exists only
  on capability 90 and that the two checkpoint shards stay untouched. No
  torch needed: the stub module supplies sentinel globals under the same
  names the copied body resolves.
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
from tests.support.upstream import SUPPORTED_RANGE, resolve_symbol
from vllm_sail.patch.utils import PATCH_REGISTRY, original_of

MODULE_PATH = (
    Path(__file__).parents[2] / "vllm_sail" / "patch" / "bugfix" / "qwen4_exp_hc_pad.py"
)
MODULE = "vllm.models.qwen4_exp.nvidia.hyperconnection"
TARGET = f"{MODULE}.GatedResidual.__init__"
REQUIRED_METADATA = ("reason", "affected_versions", "remove_when")

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
    name = f"_qwen4_hc_pad_leaf_test_{id(object())}"
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


class _FakeModule:
    def __init__(self) -> None:
        self.base_initialized = True


class _RecordingLinear:
    def __init__(self, input_size, output_sizes, **kwargs):
        self.input_size = input_size
        self.output_sizes = (
            output_sizes if isinstance(output_sizes, list) else [output_sizes]
        )
        self.kwargs = kwargs
        self.weight = NS(shape=(input_size, sum(self.output_sizes)), dtype=BF16)


def _stub_hyperconnection(
    monkeypatch: pytest.MonkeyPatch, *, capability90: bool
) -> types.ModuleType:
    class GatedResidual(_FakeModule):
        def __init__(self, config, use_combine=True, prefix=""):
            raise AssertionError("the stub upstream init must be replaced")

        @property
        def hyper_hidden_size(self):
            return self.hc_count * self.hidden_size

    hc = types.ModuleType(MODULE)
    hc.GatedResidual = GatedResidual
    # Globals the copied body resolves through bind_body.
    hc.torch = NS(bfloat16=BF16)
    hc.envs = NS(VLLM_BATCH_INVARIANT=False)
    hc.MergedColumnParallelLinear = _RecordingLinear
    hc.ReplicatedLinear = _RecordingLinear
    hc.GroupedGemmaRMSNorm = lambda *args, **kwargs: NS(args=args, kwargs=kwargs)
    hc.HyperConnectionConfig = object
    hc.maybe_prefix = lambda prefix, name: f"{prefix}.{name}" if prefix else name
    hc.current_platform = NS(
        has_device_capability=lambda cap: capability90 and cap == 90
    )
    hc.hc_up_gate_mix_supported = lambda weight, count: False
    _register_chain(monkeypatch, MODULE, hc)

    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda name: NS(
        warning=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
    )
    _register_chain(monkeypatch, "vllm.logger", logger_mod)
    return hc


def _config() -> NS:
    return NS(
        hc_lowrank=320,
        hc_count=4,
        hidden_size=128,
        hc_per_branch_norm=False,
        rms_norm_eps=1e-6,
        params_dtype=BF16,
    )


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


def test_static_discovery_resolves_the_targets() -> None:
    hooks, unresolved = scan_patch_file(MODULE_PATH)
    assert not unresolved
    assert [hook.target for hook in hooks] == [TARGET]
    assert all(hook.node is not None and not hook.additive for hook in hooks)


# ---------------------------------------------------------------------------
# Behaviour against stubs
# ---------------------------------------------------------------------------


def test_install_lands_target_and_guard(
    monkeypatch: pytest.MonkeyPatch, leaf_module
) -> None:
    hc = _stub_hyperconnection(monkeypatch, capability90=False)
    cls = hc.GatedResidual
    original_init = inspect.getattr_static(cls, "__init__")

    registry_before = len(PATCH_REGISTRY)
    leaf_module.install()
    records = PATCH_REGISTRY[registry_before:]
    assert [record.target for record in records] == [TARGET]
    assert all(record.was_missing is False for record in records)
    assert all(record.affected_versions == SUPPORTED_RANGE for record in records)

    assert cls.__init__ is not original_init
    assert original_of(cls.__init__, TARGET) is original_init

    leaf_module.install()  # the guard, not the double-patch error
    assert len(PATCH_REGISTRY) == registry_before + 1


def test_non_cap90_drops_the_padding_shard(
    monkeypatch: pytest.MonkeyPatch, leaf_module
) -> None:
    hc = _stub_hyperconnection(monkeypatch, capability90=False)
    leaf_module.install()

    mixer = hc.GatedResidual(_config(), prefix="model.layers.0.attn_hc")
    assert mixer.base_initialized
    assert mixer.pad_size == 0
    merged = mixer.input_mix_weight_down_block_inject
    # Exactly the two shards the checkpoint stores; no phantom third shard.
    assert merged.output_sizes == [320, 4]
    assert merged.kwargs["disable_tp"] is True
    assert merged.kwargs["prefix"] == (
        "model.layers.0.attn_hc.input_mix_weight_down_block_inject"
    )
    assert mixer._use_hc_down_silu is False
    assert mixer._use_hc_up_gate_mix is False


def test_cap90_keeps_the_padding_shard(
    monkeypatch: pytest.MonkeyPatch, leaf_module
) -> None:
    hc = _stub_hyperconnection(monkeypatch, capability90=True)
    leaf_module.install()

    mixer = hc.GatedResidual(_config())
    assert mixer.pad_size == 12
    merged = mixer.input_mix_weight_down_block_inject
    assert merged.output_sizes == [320, 4, 12]
    # 336 rows stay 8-aligned and bf16, so the fused down+SiLU path is eligible.
    assert mixer._use_hc_down_silu is True


def test_final_mixer_without_combine_never_pads(
    monkeypatch: pytest.MonkeyPatch, leaf_module
) -> None:
    hc = _stub_hyperconnection(monkeypatch, capability90=True)
    leaf_module.install()

    mixer = hc.GatedResidual(_config(), use_combine=False)
    assert mixer.pad_size == 0
    assert not hasattr(mixer, "input_mix_weight_down_block_inject")
    assert mixer.input_mix_weight_down.output_sizes == [320]
    assert mixer.input_mix_weight_up.output_sizes == [512]


def test_install_skips_when_upstream_module_is_missing(
    monkeypatch: pytest.MonkeyPatch, leaf_module
) -> None:
    def missing(name):
        raise ImportError(f"no module named {name}")

    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda name: NS(
        warning=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
    )
    _register_chain(monkeypatch, "vllm.logger", logger_mod)
    monkeypatch.setattr(leaf_module.importlib, "import_module", missing)
    registry_before = len(PATCH_REGISTRY)
    leaf_module.install()
    assert leaf_module._unavailable is True
    assert leaf_module._installed is False
    assert len(PATCH_REGISTRY) == registry_before
    leaf_module.install()  # stays a no-op once unavailable
    assert len(PATCH_REGISTRY) == registry_before


# ---------------------------------------------------------------------------
# Copied-body contract against the pinned upstream source
# ---------------------------------------------------------------------------


@pytest.mark.upstream_source
def test_copied_body_matches_pinned_upstream(upstream_source_root) -> None:
    resolved = resolve_symbol(upstream_source_root, TARGET)
    assert resolved.present, TARGET
    upstream_init = resolved.node

    tree = ast.parse(MODULE_PATH.read_text())
    local_init = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    up_stmts = upstream_init.body
    lo_stmts = local_init.body
    assert len(up_stmts) == len(lo_stmts)
    differing = [
        index
        for index, (up, lo) in enumerate(zip(up_stmts, lo_stmts, strict=True))
        if ast.dump(up) != ast.dump(lo)
    ]
    # Only the explicit super() call and the pad_size assignment may differ.
    assert len(differing) == 2, differing
    super_stmt, pad_stmt = (lo_stmts[index] for index in differing)
    assert isinstance(super_stmt, ast.Expr)
    assert isinstance(pad_stmt, ast.Assign)
    assert any(
        isinstance(target, ast.Attribute) and target.attr == "pad_size"
        for target in pad_stmt.targets
    )
    # The pinned upstream still declares the padding unconditionally: the bug
    # this patch fixes is present in the source it targets.
    up_pad = next(
        stmt
        for stmt in up_stmts
        if isinstance(stmt, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "pad_size"
            for target in stmt.targets
        )
    )
    assert isinstance(up_pad.value, ast.IfExp)
    assert ast.dump(up_pad.value.test) == ast.dump(
        ast.Name(id="use_combine", ctx=ast.Load())
    )
