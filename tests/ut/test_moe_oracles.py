# SPDX-License-Identifier: Apache-2.0
"""CPU-safe contract tests for the four PPU MoE oracle extensions."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.source import ROOT, definition

WNA16_PATCH = "vllm_sail/registry/moe_backends/int_wna16.py"


def test_moe_backend_package_is_importable_without_vllm() -> None:
    package = importlib.import_module("vllm_sail.registry.moe_backends")
    assert callable(package.extend_enum)
    assert callable(package.register)


def test_wna16_kernel_assembly_delegates_to_upstream_make_moe_kernel() -> None:
    """The copied body keeps the PPU allowlist but reuses upstream assembly."""
    node = definition(Path(ROOT) / WNA16_PATCH, "make_wna16_moe_kernel")
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    calls = {
        n.func.id
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "make_moe_kernel" in calls
    assert "maybe_make_prepare_finalize" not in calls
    assert "ZentorchExpertsWNA16" in names
    assert "ZentorchExpertsInt4" not in names
    assert {"PPUDeepGemmExperts", "PPUBatchedDeepGemmExperts"} <= names


@pytest.fixture(scope="module")
def oracles():
    """Load the real oracle stack, or skip when its real deps are unavailable."""
    pytest.importorskip("torch")
    pytest.importorskip("vllm")

    from vllm.platforms import current_platform

    from vllm_sail.registry import moe_backends

    moe_backends.register()
    from vllm_sail.registry.moe_backends import fp8, int8, mxfp4, unquantized

    return SimpleNamespace(
        current_platform=current_platform,
        fp8=fp8,
        int8=int8,
        mxfp4=mxfp4,
        unquantized=unquantized,
    )


def _activation_format(oracles, parallel):
    fmt = oracles.mxfp4.mk.FusedMoEActivationFormat
    return (
        fmt.BatchedExperts if parallel.use_batched_activation_format else fmt.Standard
    )


def _config(oracles):
    parallel = SimpleNamespace(
        use_batched_activation_format=False,
        use_deepep_v2_kernels=False,
        dp_size=1,
        ep_size=1,
    )
    return SimpleNamespace(
        moe_parallel_config=parallel,
        moe_backend="auto",
        activation=oracles.unquantized.oracle.MoEActivation.SILU,
        has_bias=False,
        activation_format=_activation_format(oracles, parallel),
        is_lora_enabled=False,
    )


def _set_platform(monkeypatch: pytest.MonkeyPatch, oracles, *, ppu: bool) -> None:
    platform_type = type(oracles.current_platform)
    monkeypatch.setattr(platform_type, "is_ppu", lambda self: ppu)
    monkeypatch.setattr(platform_type, "is_cuda", lambda self: True)
    monkeypatch.setattr(platform_type, "is_rocm", lambda self: False)
    monkeypatch.setattr(platform_type, "is_xpu", lambda self: False)
    monkeypatch.setattr(platform_type, "is_cpu", lambda self: False)
    monkeypatch.setattr(platform_type, "is_device_capability", lambda *_: False)
    monkeypatch.setattr(platform_type, "is_device_capability_family", lambda *_: False)


@pytest.mark.parametrize(
    ("oracle_name", "members"),
    [
        (
            "unquantized",
            {
                "PPU_DEEPGEMM": "PPU_DEEPGEMM",
                "BATCHED_PPU_DEEPGEMM": "BATCHED_PPU_DEEPGEMM",
                "ACEXT": "ACEXT",
            },
        ),
        (
            "fp8",
            {
                "PPU_DEEPGEMM": "PPU_DEEPGEMM",
                "BATCHED_PPU_DEEPGEMM": "BATCHED_PPU_DEEPGEMM",
            },
        ),
        (
            "int8",
            {
                "PPU_DEEPGEMM": "PPU_DEEPGEMM",
                "BATCHED_PPU_DEEPGEMM": "BATCHED_PPU_DEEPGEMM",
                "ACEXT": "ACEXT",
            },
        ),
        (
            "mxfp4",
            {
                "PPU_DEEPGEMM_MXFP4": "PPU_DEEPGEMM_MXFP4",
                "BATCHED_PPU_DEEPGEMM_MXFP4": ("BATCHED_PPU_DEEPGEMM_MXFP4"),
            },
        ),
    ],
)
def test_enum_members_cover_all_lookup_paths(oracles, oracle_name, members) -> None:
    plugin = getattr(oracles, oracle_name)
    enum_cls = getattr(
        plugin,
        {
            "unquantized": "UnquantizedMoeBackend",
            "fp8": "Fp8MoeBackend",
            "int8": "Int8MoeBackend",
            "mxfp4": "Mxfp4MoeBackend",
        }[oracle_name],
    )
    for name, value in members.items():
        member = getattr(enum_cls, name)
        assert member.name == name
        assert member.value == value
        assert enum_cls[name] is member
        assert enum_cls(value) is member
        assert member in list(enum_cls)


def test_extend_enum_stays_idempotent_after_helper_reimport(oracles) -> None:
    member = oracles.fp8.Fp8MoeBackend.PPU_DEEPGEMM
    extend_module = importlib.reload(
        importlib.import_module("vllm_sail.registry.moe_backends._extend")
    )
    assert (
        extend_module.extend_enum(oracles.fp8.Fp8MoeBackend, member.name, member.value)
        is member
    )


@pytest.mark.parametrize("oracle_name", ["unquantized", "fp8", "int8", "mxfp4"])
def test_backend_to_kernel_cls_delegates_upstream(
    monkeypatch: pytest.MonkeyPatch, oracles, oracle_name
) -> None:
    plugin = getattr(oracles, oracle_name)

    class UpstreamExperts:
        pass

    monkeypatch.setattr(
        plugin, "_upstream_backend_to_kernel_cls", lambda backend: [UpstreamExperts]
    )
    assert plugin.backend_to_kernel_cls(
        plugin.oracle.__dict__[
            {
                "unquantized": "UnquantizedMoeBackend",
                "fp8": "Fp8MoeBackend",
                "int8": "Int8MoeBackend",
                "mxfp4": "Mxfp4MoeBackend",
            }[oracle_name]
        ].TRITON
    ) == [UpstreamExperts]


@pytest.mark.parametrize(
    ("oracle_name", "member_name", "module_name", "class_name"),
    [
        ("unquantized", "PPU_DEEPGEMM", "deep_gemm_moe", "PPUDeepGemmExperts"),
        (
            "unquantized",
            "BATCHED_PPU_DEEPGEMM",
            "batched_deep_gemm_moe",
            "PPUBatchedDeepGemmExperts",
        ),
        ("fp8", "PPU_DEEPGEMM", "deep_gemm_moe", "PPUDeepGemmExperts"),
        (
            "fp8",
            "BATCHED_PPU_DEEPGEMM",
            "batched_deep_gemm_moe",
            "PPUBatchedDeepGemmExperts",
        ),
        ("int8", "PPU_DEEPGEMM", "deep_gemm_moe", "PPUDeepGemmExperts"),
        (
            "int8",
            "BATCHED_PPU_DEEPGEMM",
            "batched_deep_gemm_moe",
            "PPUBatchedDeepGemmExperts",
        ),
        (
            "mxfp4",
            "PPU_DEEPGEMM_MXFP4",
            "deep_gemm_moe",
            "PPUDeepGemmExpertsMXFP4",
        ),
        (
            "mxfp4",
            "BATCHED_PPU_DEEPGEMM_MXFP4",
            "batched_deep_gemm_moe",
            "PPUBatchedDeepGemmExpertsMXFP4",
        ),
    ],
)
def test_backend_to_kernel_cls_returns_ppu_expert(
    oracles, oracle_name, member_name, module_name, class_name
) -> None:
    pytest.importorskip("triton", reason="PPU DeepGEMM expert modules need Triton")
    plugin = getattr(oracles, oracle_name)
    enum_cls = getattr(
        plugin,
        {
            "unquantized": "UnquantizedMoeBackend",
            "fp8": "Fp8MoeBackend",
            "int8": "Int8MoeBackend",
            "mxfp4": "Mxfp4MoeBackend",
        }[oracle_name],
    )
    experts = importlib.import_module(
        "vllm_sail.model_executor.layers.fused_moe.experts." + module_name
    )
    assert plugin.backend_to_kernel_cls(getattr(enum_cls, member_name)) == [
        getattr(experts, class_name)
    ]


@pytest.mark.parametrize("oracle_name", ["unquantized", "int8"])
def test_backend_to_kernel_cls_returns_acext_expert(oracles, oracle_name) -> None:
    from vllm_sail.model_executor.layers.fused_moe.experts.acext import AcextExperts

    plugin = getattr(oracles, oracle_name)
    assert plugin.backend_to_kernel_cls(
        plugin.__dict__[
            "UnquantizedMoeBackend"
            if oracle_name == "unquantized"
            else "Int8MoeBackend"
        ].ACEXT
    ) == [AcextExperts]


def test_map_functions_add_ppu_names_and_delegate_upstream_names(oracles) -> None:
    unquantized, fp8, int8, mxfp4 = (
        oracles.unquantized,
        oracles.fp8,
        oracles.int8,
        oracles.mxfp4,
    )
    assert unquantized.map_unquantized_backend("ppu_deep_gemm") is (
        unquantized.UnquantizedMoeBackend.PPU_DEEPGEMM
    )
    assert unquantized.map_unquantized_backend("ppu_acext") is (
        unquantized.UnquantizedMoeBackend.ACEXT
    )
    assert fp8.map_fp8_backend("ppu_deep_gemm") is fp8.Fp8MoeBackend.PPU_DEEPGEMM
    assert int8.map_int8_backend("ppu_acext") is int8.Int8MoeBackend.ACEXT
    assert mxfp4.map_mxfp4_backend("ppu_deep_gemm") == [
        mxfp4.Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4,
        mxfp4.Mxfp4MoeBackend.BATCHED_PPU_DEEPGEMM_MXFP4,
    ]
    assert unquantized.map_unquantized_backend("triton") is (
        unquantized._upstream_map_unquantized_backend("triton")
    )
    assert fp8.map_fp8_backend("triton") is fp8._upstream_map_fp8_backend("triton")
    assert int8.map_int8_backend("triton") is int8._upstream_map_int8_backend("triton")
    assert mxfp4.map_mxfp4_backend("triton") == (
        mxfp4._upstream_map_mxfp4_backend("triton")
    )


def test_priority_lists_include_ppu_members(
    monkeypatch: pytest.MonkeyPatch, oracles
) -> None:
    _set_platform(monkeypatch, oracles, ppu=True)
    monkeypatch.delenv("VLLM_PPU_MOE_BACKEND", raising=False)
    deep_gemm = importlib.import_module("vllm_sail.utils.deep_gemm")
    acext = importlib.import_module(
        "vllm_sail.model_executor.layers.fused_moe.experts.acext"
    )
    monkeypatch.setattr(deep_gemm, "is_deep_gemm_supported", lambda: True)
    monkeypatch.setattr(acext, "is_acext_supported", lambda: True)
    config = _config(oracles)

    assert oracles.unquantized.UnquantizedMoeBackend.PPU_DEEPGEMM in (
        oracles.unquantized._get_priority_backends(config)
    )
    assert oracles.fp8.Fp8MoeBackend.PPU_DEEPGEMM in (
        oracles.fp8._get_priority_backends(config, None, None)
    )
    assert oracles.int8.Int8MoeBackend.ACEXT in (
        oracles.int8._get_priority_backends(config)
    )
    assert oracles.mxfp4.Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4 in (
        oracles.mxfp4._get_priority_backends()
    )
    assert oracles.mxfp4.Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4 in (
        oracles.mxfp4._get_priority_backends_for_gpt_oss()
    )


def test_priority_lists_are_unchanged_off_ppu(
    monkeypatch: pytest.MonkeyPatch, oracles
) -> None:
    _set_platform(monkeypatch, oracles, ppu=False)
    config = _config(oracles)

    assert oracles.unquantized._get_priority_backends(config) == (
        oracles.unquantized._upstream_get_priority_backends(config)
    )
    assert oracles.fp8._get_priority_backends(config, None, None) == (
        oracles.fp8._upstream_get_priority_backends(config, None, None)
    )
    assert oracles.int8._get_priority_backends(config) == (
        oracles.int8._upstream_get_priority_backends(config)
    )
    assert oracles.mxfp4._get_priority_backends() == (
        oracles.mxfp4._upstream_get_priority_backends()
    )
    assert oracles.mxfp4._get_priority_backends_for_gpt_oss() == (
        oracles.mxfp4._upstream_get_priority_backends_for_gpt_oss()
    )


def test_every_oracle_patch_records_original_and_rejects_reapplication(
    oracles,
) -> None:
    from vllm_sail.patch.utils import PATCH_REGISTRY, original_of, patch

    records = [record for record in PATCH_REGISTRY if ".oracle." in record.target]
    # 18 from the MoE-backend registry; the enhancement chain's mxfp4 and
    # fused_moe_ppu modules add five more when they ran in this process.
    assert len(records) >= 18
    for record in records:
        module_name, attribute = record.target.rsplit(".", 1)
        replacement = getattr(importlib.import_module(module_name), attribute)
        assert original_of(replacement, record.target) is not replacement
        with pytest.raises(RuntimeError, match="already patched"):
            patch(
                module_name,
                attribute,
                reason=record.reason,
                affected_versions=record.affected_versions,
                remove_when=record.remove_when,
            )(replacement)


# ---------------------------------------------------------------------------
# Selection tests with mocked is_ppu() -- the second half of the Phase-3 exit
# criteria. Expert classes are faked so the tests do not need the PPU SDK or
# Triton; the selection *logic* is what is under test here.
# ---------------------------------------------------------------------------


class _FakeExperts:
    @staticmethod
    def is_supported_config(cls, moe_config, weight_key, activation_key, fmt):
        return True, None


def _select_config(oracles, *, backend: str, batched: bool):
    parallel = SimpleNamespace(
        use_batched_activation_format=batched,
        use_deepep_v2_kernels=False,
        dp_size=1,
        ep_size=1,
    )
    return SimpleNamespace(
        moe_parallel_config=parallel,
        moe_backend=backend,
        activation=oracles.unquantized.oracle.MoEActivation.SILU,
        has_bias=False,
        activation_format=_activation_format(oracles, parallel),
        is_lora_enabled=False,
    )


def test_select_unquantized_batched_ppu_deep_gemm(
    monkeypatch: pytest.MonkeyPatch, oracles
) -> None:
    plugin = oracles.unquantized
    monkeypatch.setattr(plugin, "backend_to_kernel_cls", lambda backend: [_FakeExperts])
    backend, kernel_cls = plugin.select_unquantized_moe_backend(
        _select_config(oracles, backend="ppu_deep_gemm", batched=True)
    )
    assert backend is plugin.UnquantizedMoeBackend.BATCHED_PPU_DEEPGEMM
    assert kernel_cls is _FakeExperts


def test_select_unquantized_batched_ppu_deep_gemm_raises_when_unsupported(
    monkeypatch: pytest.MonkeyPatch, oracles
) -> None:
    plugin = oracles.unquantized

    class _Unsupported(_FakeExperts):
        @staticmethod
        def is_supported_config(cls, moe_config, weight_key, activation_key, fmt):
            return False, "testing"

    monkeypatch.setattr(plugin, "backend_to_kernel_cls", lambda backend: [_Unsupported])
    with pytest.raises(ValueError, match="unsupported"):
        plugin.select_unquantized_moe_backend(
            _select_config(oracles, backend="ppu_deep_gemm", batched=True)
        )


@pytest.mark.parametrize("batched", [False, True])
def test_select_unquantized_delegates_for_non_ppu_backends(
    monkeypatch: pytest.MonkeyPatch, oracles, batched
) -> None:
    plugin = oracles.unquantized
    sentinel = object()
    monkeypatch.setattr(
        plugin, "_upstream_select_unquantized_moe_backend", lambda config: sentinel
    )
    assert (
        plugin.select_unquantized_moe_backend(
            _select_config(oracles, backend="auto", batched=batched)
        )
        is sentinel
    )


@pytest.mark.parametrize(
    ("batched", "expected_member"),
    [
        (False, "PPU_DEEPGEMM"),
        (True, "BATCHED_PPU_DEEPGEMM"),
    ],
)
def test_select_fp8_explicit_ppu_deep_gemm(
    monkeypatch: pytest.MonkeyPatch, oracles, batched, expected_member
) -> None:
    plugin = oracles.fp8
    monkeypatch.setattr(plugin, "backend_to_kernel_cls", lambda backend: [_FakeExperts])
    backend, kernel_cls = plugin.select_fp8_moe_backend(
        _select_config(oracles, backend="ppu_deep_gemm", batched=batched), None, None
    )
    assert backend is getattr(plugin.Fp8MoeBackend, expected_member)
    assert kernel_cls is _FakeExperts


@pytest.mark.parametrize("prefix", ["VLLM_SAIL", "VLLM_PPU"])
def test_select_fp8_env_deepgemm_selects_ppu_only_on_ppu(
    monkeypatch: pytest.MonkeyPatch, oracles, prefix: str
) -> None:
    plugin = oracles.fp8
    monkeypatch.setattr(plugin, "backend_to_kernel_cls", lambda backend: [_FakeExperts])
    monkeypatch.delenv("VLLM_SAIL_MOE_BACKEND", raising=False)
    monkeypatch.setenv(f"{prefix}_MOE_BACKEND", "deepgemm")

    _set_platform(monkeypatch, oracles, ppu=True)
    backend, _ = plugin.select_fp8_moe_backend(
        _select_config(oracles, backend="auto", batched=False), None, None
    )
    assert backend is plugin.Fp8MoeBackend.PPU_DEEPGEMM

    _set_platform(monkeypatch, oracles, ppu=False)
    sentinel = object()
    monkeypatch.setattr(
        plugin,
        "_upstream_select_fp8_moe_backend",
        lambda *args, **kwargs: sentinel,
    )
    assert (
        plugin.select_fp8_moe_backend(
            _select_config(oracles, backend="auto", batched=False), None, None
        )
        is sentinel
    )


@pytest.mark.parametrize("prefix", ["VLLM_SAIL", "VLLM_PPU"])
def test_select_fp8_env_acext_delegates_upstream(
    monkeypatch: pytest.MonkeyPatch, oracles, prefix: str
) -> None:
    plugin = oracles.fp8
    monkeypatch.delenv("VLLM_SAIL_MOE_BACKEND", raising=False)
    monkeypatch.setenv(f"{prefix}_MOE_BACKEND", "acext")
    _set_platform(monkeypatch, oracles, ppu=True)
    sentinel = object()
    monkeypatch.setattr(
        plugin,
        "_upstream_select_fp8_moe_backend",
        lambda *args, **kwargs: sentinel,
    )
    assert (
        plugin.select_fp8_moe_backend(
            _select_config(oracles, backend="auto", batched=False), None, None
        )
        is sentinel
    )


def test_select_int8_batched_ppu_deep_gemm(
    monkeypatch: pytest.MonkeyPatch, oracles
) -> None:
    plugin = oracles.int8
    monkeypatch.setattr(plugin, "backend_to_kernel_cls", lambda backend: [_FakeExperts])
    backend, kernel_cls = plugin.select_int8_moe_backend(
        _select_config(oracles, backend="ppu_deep_gemm", batched=True)
    )
    assert backend is plugin.Int8MoeBackend.BATCHED_PPU_DEEPGEMM
    assert kernel_cls is _FakeExperts


@pytest.mark.parametrize(
    "backend_name",
    ["PPU_DEEPGEMM", "BATCHED_PPU_DEEPGEMM", "ACEXT"],
)
def test_convert_int8_ppu_weights_preserves_canonical_layout(
    oracles, backend_name
) -> None:
    """PPU INT8 experts consume the checkpoint's canonical weight layout."""
    import torch

    plugin = oracles.int8
    w13 = torch.empty((2, 4, 8), dtype=torch.int8)
    w2 = torch.empty((2, 8, 2), dtype=torch.int8)

    converted = plugin.oracle.convert_to_int8_moe_kernel_format(
        getattr(plugin.Int8MoeBackend, backend_name),
        w13,
        w2,
    )

    assert converted[0] is w13
    assert converted[1] is w2


def test_convert_int8_upstream_backends_delegate_unchanged(
    monkeypatch: pytest.MonkeyPatch, oracles
) -> None:
    plugin = oracles.int8
    w13 = object()
    w2 = object()
    layer = object()
    w13_scale = object()
    sentinel = (object(), object())
    captured = {}

    def _upstream(backend, arg_w13, arg_w2, *, layer, w13_scale):
        captured.update(
            backend=backend,
            w13=arg_w13,
            w2=arg_w2,
            layer=layer,
            w13_scale=w13_scale,
        )
        return sentinel

    monkeypatch.setattr(
        plugin,
        "_upstream_convert_to_int8_moe_kernel_format",
        _upstream,
    )

    assert (
        plugin.convert_to_int8_moe_kernel_format(
            plugin.Int8MoeBackend.TRITON,
            w13,
            w2,
            layer=layer,
            w13_scale=w13_scale,
        )
        is sentinel
    )
    assert captured == {
        "backend": plugin.Int8MoeBackend.TRITON,
        "w13": w13,
        "w2": w2,
        "layer": layer,
        "w13_scale": w13_scale,
    }


@pytest.mark.parametrize(
    ("backend", "match"),
    [
        ("triton", "Triton MoE backend is disabled"),
        ("ppu_acext", "ACEXT MoE backend is disabled"),
    ],
)
def test_select_int8_batched_rejects_standard_only_backends(
    monkeypatch: pytest.MonkeyPatch, oracles, backend, match
) -> None:
    plugin = oracles.int8
    with pytest.raises(ValueError, match=match):
        plugin.select_int8_moe_backend(
            _select_config(oracles, backend=backend, batched=True)
        )


@pytest.mark.parametrize("batched", [False, True])
def test_select_int8_auto_delegates_upstream(
    monkeypatch: pytest.MonkeyPatch, oracles, batched
) -> None:
    plugin = oracles.int8
    sentinel = object()
    monkeypatch.setattr(
        plugin,
        "_upstream_select_int8_moe_backend",
        lambda *args, **kwargs: sentinel,
    )
    assert (
        plugin.select_int8_moe_backend(
            _select_config(oracles, backend="auto", batched=batched)
        )
        is sentinel
    )


def test_select_mxfp4_batched_ppu_deep_gemm_uses_batched_member(
    monkeypatch: pytest.MonkeyPatch, oracles
) -> None:
    plugin = oracles.mxfp4
    captured = {}

    def _capture(
        self,
        backend,
        available_backends=None,
        kernel_classes=None,
        activation_key=None,
        is_lora=False,
    ):
        # map_mxfp4_backend() lists the standard PPU member followed by its
        # batched sibling and lets the shared is_supported_config() format check
        # drop the one that does not match the MoE layout; this stub mirrors it.
        activation_format = self.moe_config.activation_format
        own_format = (
            plugin.mk.FusedMoEActivationFormat.BatchedExperts
            if backend.name.startswith("BATCHED_")
            else plugin.mk.FusedMoEActivationFormat.Standard
        )
        if activation_format != own_format:
            raise ValueError(f"{backend.value} rejected: activation format")
        captured.update(
            backend=backend,
            weight_key=self.weight_key,
            activation_key=activation_key,
            activation_format=activation_format,
        )
        return backend, _FakeExperts

    monkeypatch.setattr(plugin.oracle.Mxfp4MoEKernelOracle, "return_or_raise", _capture)
    monkeypatch.setattr(plugin.oracle, "_resolve_activation_key", lambda key: key)
    backend, kernel_cls = plugin.select_mxfp4_moe_backend(
        _select_config(oracles, backend="ppu_deep_gemm", batched=True)
    )
    assert backend is plugin.Mxfp4MoeBackend.BATCHED_PPU_DEEPGEMM_MXFP4
    assert kernel_cls is _FakeExperts
    assert captured["backend"] is plugin.Mxfp4MoeBackend.BATCHED_PPU_DEEPGEMM_MXFP4
    assert captured["activation_key"] is plugin.oracle.kMxfp4Dynamic
    assert (
        captured["activation_format"]
        is plugin.mk.FusedMoEActivationFormat.BatchedExperts
    )


def test_select_mxfp4_non_batched_ppu_deep_gemm_uses_standard_member(
    monkeypatch: pytest.MonkeyPatch, oracles
) -> None:
    plugin = oracles.mxfp4
    captured = {}

    def _capture(
        self,
        backend,
        available_backends=None,
        kernel_classes=None,
        activation_key=None,
        is_lora=False,
    ):
        # map_mxfp4_backend() lists the standard PPU member followed by its
        # batched sibling and lets the shared is_supported_config() format check
        # drop the one that does not match the MoE layout; this stub mirrors it.
        activation_format = self.moe_config.activation_format
        own_format = (
            plugin.mk.FusedMoEActivationFormat.BatchedExperts
            if backend.name.startswith("BATCHED_")
            else plugin.mk.FusedMoEActivationFormat.Standard
        )
        if activation_format != own_format:
            raise ValueError(f"{backend.value} rejected: activation format")
        captured.update(
            backend=backend,
            weight_key=self.weight_key,
            activation_key=activation_key,
            activation_format=activation_format,
        )
        return backend, _FakeExperts

    monkeypatch.setattr(plugin.oracle.Mxfp4MoEKernelOracle, "return_or_raise", _capture)
    monkeypatch.setattr(plugin.oracle, "_resolve_activation_key", lambda key: key)
    backend, kernel_cls = plugin.select_mxfp4_moe_backend(
        _select_config(oracles, backend="ppu_deep_gemm", batched=False)
    )
    assert backend is plugin.Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4
    assert kernel_cls is _FakeExperts
    assert captured["backend"] is plugin.Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4
    assert captured["activation_key"] is plugin.oracle.kMxfp4Dynamic
    assert captured["activation_format"] is plugin.mk.FusedMoEActivationFormat.Standard


def test_select_mxfp4_moe_backend_is_upstreams(oracles) -> None:
    """The plugin re-exports upstream's selector instead of copying its body."""
    plugin = oracles.mxfp4
    assert plugin.select_mxfp4_moe_backend is (
        plugin._upstream_select_mxfp4_moe_backend
    )


# ---------------------------------------------------------------------------
# is_supported_config coverage for every registered PPU expert kernel. The
# static predicate matrix below the upstream staticmethod is what the oracles
# consult, so each PPU-overridden gate gets an accept and a reject case.
# ---------------------------------------------------------------------------


@pytest.fixture()
def moe_config_factory():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )

    parallel = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="naive",
        enable_eplb=False,
    )

    def _make(activation=MoEActivation.SILU):
        import torch

        return FusedMoEConfig(
            num_experts=8,
            experts_per_token=2,
            hidden_dim=128,
            intermediate_size=256,
            num_local_experts=8,
            num_logical_experts=8,
            activation=activation,
            device="cpu",
            routing_method=RoutingMethodType.TopK,
            moe_parallel_config=parallel,
            in_dtype=torch.bfloat16,
        )

    return _make


def test_acext_experts_is_supported_config_matrix(
    monkeypatch: pytest.MonkeyPatch, oracles, moe_config_factory
) -> None:
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8StaticTensorSym,
        kInt8DynamicTokenSym,
        kInt8StaticChannelSym,
    )

    from vllm_sail.model_executor.layers.fused_moe.experts.acext import AcextExperts

    _set_platform(monkeypatch, oracles, ppu=True)
    standard = mk.FusedMoEActivationFormat.Standard
    config = moe_config_factory()
    check = AcextExperts.is_supported_config

    assert check(AcextExperts, config, None, None, standard) == (True, None)
    assert check(
        AcextExperts,
        config,
        kInt8StaticChannelSym,
        kInt8DynamicTokenSym,
        standard,
    ) == (True, None)

    supported, reason = check(AcextExperts, config, kFp8StaticTensorSym, None, standard)
    assert supported is False and "quantization scheme" in reason
    supported, reason = check(
        AcextExperts, config, None, None, mk.FusedMoEActivationFormat.BatchedExperts
    )
    assert supported is False and "activation format" in reason
    supported, reason = check(
        AcextExperts, moe_config_factory(MoEActivation.GELU), None, None, standard
    )
    assert supported is False and "activation" in reason

    _set_platform(monkeypatch, oracles, ppu=False)
    supported, reason = check(AcextExperts, config, None, None, standard)
    assert supported is False and "current device" in reason


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("deep_gemm_moe", "PPUDeepGemmExperts"),
        ("deep_gemm_moe", "PPUDeepGemmExpertsMXFP4"),
        ("batched_deep_gemm_moe", "PPUBatchedDeepGemmExperts"),
        ("batched_deep_gemm_moe", "PPUBatchedDeepGemmExpertsMXFP4"),
    ],
)
def test_deep_gemm_experts_support_gates(
    monkeypatch: pytest.MonkeyPatch, oracles, module_name, class_name
) -> None:
    pytest.importorskip("triton", reason="PPU DeepGEMM expert modules need Triton")
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
        kFp8DynamicTokenSym,
        kFp8Static128BlockSym,
        kFp8StaticChannelSym,
        kInt8DynamicTokenSym,
        kInt8StaticChannelSym,
    )

    import vllm_sail.utils.deep_gemm as deep_gemm

    experts = importlib.import_module(
        "vllm_sail.model_executor.layers.fused_moe.experts." + module_name
    )
    kernel_cls = getattr(experts, class_name)

    # Device gate: needs the PPU DeepGEMM wrapper; MXFP4 variants additionally
    # refuse sm80. The experts modules bind the gate at import time, so patch
    # both the utils module and the bound reference.
    def _set_gate(value: bool) -> None:
        monkeypatch.setattr(deep_gemm, "is_deep_gemm_supported", lambda: value)
        monkeypatch.setattr(
            experts, "is_deep_gemm_supported", lambda: value, raising=False
        )

    _set_gate(False)
    assert kernel_cls._supports_current_device() is False
    _set_gate(True)
    _set_platform(monkeypatch, oracles, ppu=True)
    if class_name.endswith("MXFP4"):
        monkeypatch.setattr(
            oracles.current_platform,
            "is_device_capability",
            lambda cap: cap == (8, 0),
        )
        assert kernel_cls._supports_current_device() is False
        monkeypatch.setattr(
            oracles.current_platform, "is_device_capability", lambda cap: False
        )
    assert kernel_cls._supports_current_device() is True

    # Quant-scheme gate: every PPU DeepGEMM expert accepts bf16 and the fp8/int8
    # keys its oracle advertises, and rejects a mismatched pair.
    schemes = [
        (None, None),
        (kFp8Static128BlockSym, kFp8Dynamic128Sym),
        (kFp8StaticChannelSym, kFp8DynamicTokenSym),
        (kInt8StaticChannelSym, kInt8DynamicTokenSym),
    ]
    if class_name.endswith("MXFP4"):
        schemes = [
            (
                oracles.mxfp4.oracle.kMxfp4Static,
                oracles.mxfp4.oracle.kMxfp4Dynamic,
            )
        ]
    for weight_key, activation_key in schemes:
        assert kernel_cls._supports_quant_scheme(weight_key, activation_key) is True
    assert kernel_cls._supports_quant_scheme(kFp8Static128BlockSym, None) is False

    # Activation gate: SILU always; the batched/class variants differ on the
    # SwiGLU variants, so only assert the common floor and one rejection.
    assert kernel_cls._supports_activation(MoEActivation.SILU) is True
    assert kernel_cls._supports_activation(MoEActivation.GELU) is False

    # No act-and_mul support anywhere in the PPU expert family.
    assert kernel_cls._supports_no_act_and_mul() is False


@pytest.mark.parametrize(
    ("module_name", "class_name", "expected_format"),
    [
        ("deep_gemm_moe", "PPUDeepGemmExperts", "Standard"),
        ("batched_deep_gemm_moe", "PPUBatchedDeepGemmExperts", "BatchedExperts"),
        ("batched_deep_gemm_moe", "PPUBatchedDeepGemmExpertsMXFP4", "BatchedExperts"),
    ],
)
def test_deep_gemm_experts_is_supported_config_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    oracles,
    moe_config_factory,
    module_name,
    class_name,
    expected_format,
) -> None:
    pytest.importorskip("triton", reason="PPU DeepGEMM expert modules need Triton")
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kInt8DynamicTokenSym,
        kInt8StaticChannelSym,
    )

    import vllm_sail.utils.deep_gemm as deep_gemm

    experts = importlib.import_module(
        "vllm_sail.model_executor.layers.fused_moe.experts." + module_name
    )
    kernel_cls = getattr(experts, class_name)

    monkeypatch.setattr(deep_gemm, "is_deep_gemm_supported", lambda: True)
    monkeypatch.setattr(experts, "is_deep_gemm_supported", lambda: True, raising=False)
    _set_platform(monkeypatch, oracles, ppu=True)

    fmt = getattr(mk.FusedMoEActivationFormat, expected_format)
    if class_name.endswith("MXFP4"):
        weight_key = oracles.mxfp4.oracle.kMxfp4Static
        activation_key = oracles.mxfp4.oracle.kMxfp4Dynamic
    else:
        weight_key = kInt8StaticChannelSym
        activation_key = kInt8DynamicTokenSym
    supported, reason = kernel_cls.is_supported_config(
        kernel_cls,
        moe_config_factory(),
        weight_key,
        activation_key,
        fmt,
    )
    assert supported is True, reason

    # Wrong activation format for the kernel must be rejected with a reason.
    wrong = mk.FusedMoEActivationFormat.BatchedExperts
    if expected_format == "BatchedExperts":
        wrong = mk.FusedMoEActivationFormat.Standard
    supported, reason = kernel_cls.is_supported_config(
        kernel_cls,
        moe_config_factory(),
        kInt8StaticChannelSym,
        kInt8DynamicTokenSym,
        wrong,
    )
    assert supported is False and reason
