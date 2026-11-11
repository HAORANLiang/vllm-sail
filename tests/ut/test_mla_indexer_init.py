# SPDX-License-Identifier: Apache-2.0
"""Bare-Python regressions for indexer initialization and adaptive dispatch."""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

from tests.support.source import assert_accepts_upstream_keywords, function
from tests.ut import test_ppu_kernel_capabilities as harness

modules = harness.modules

PATCH_PATH = (
    Path(__file__).parents[2] / "vllm_sail/patch/enhancement/attention/mla_indexer.py"
)
FP4_CONSUMERS = (
    "vllm.model_executor.models.deepseek_v2",
    "vllm.models.deepseek_v4.attention",
    "vllm.models.deepseek_v41.attention",
)


@pytest.fixture
def fp4_indexer(modules, patch_utils_module):
    platform = types.SimpleNamespace(
        ppu=True,
        capability=89,
        is_ppu=lambda: platform.ppu,
        is_device_capability=lambda value: platform.capability == value,
    )

    def upstream(config):
        use_fp4 = config.attention_config.resolve_indexer_kv_dtype("fp8") == "mxfp4"
        if use_fp4 and platform.capability != 100:
            raise ValueError("requires Blackwell datacenter GPUs")
        return use_fp4

    modules("vllm.platforms", current_platform=platform)
    modules("torch", float32=object())
    modules("vllm_sail.utils.deep_gemm")
    modules("vllm_sail.patch.utils", patch_value=patch_utils_module.patch_value)
    builder = type(
        "DeepseekV32IndexerMetadataBuilder",
        (),
        {
            "__init__": lambda *a, **kw: None,
            "build": lambda *a, **kw: None,
            "_split_indexer_prefill_chunks": staticmethod(lambda *a, **kw: None),
        },
    )
    target = modules(
        "vllm.v1.attention.backends.mla.indexer",
        DeepseekV32IndexerMetadataBuilder=builder,
        get_paged_mqa_logits_metadata=lambda *a, **kw: None,
        _supports_flattened_device_query_lens=lambda: False,
        DSA_INDEXER_KV_DTYPES=("fp8", "mxfp4"),
        dsa_indexer_uses_fp4=upstream,
    )
    return platform, target, upstream


@pytest.mark.parametrize("consumer_name", FP4_CONSUMERS)
@pytest.mark.parametrize("preloaded", [False, True])
@pytest.mark.parametrize("dtype", ["fp8", "mxfp4"])
def test_fp4_capability_rebinds_preloaded_and_future_consumers(
    modules, fp4_indexer, consumer_name, preloaded, dtype
):
    _, provider, upstream = fp4_indexer
    if preloaded:
        consumer = modules(consumer_name, dsa_indexer_uses_fp4=upstream)
    patched = harness.load_patch("vllm_sail/patch/enhancement/attention/mla_indexer.py")
    if not preloaded:
        consumer = modules(consumer_name)
        exec(
            "from vllm.v1.attention.backends.mla.indexer import dsa_indexer_uses_fp4",
            consumer.__dict__,
        )
    assert consumer.dsa_indexer_uses_fp4 is provider.dsa_indexer_uses_fp4
    assert consumer.dsa_indexer_uses_fp4 is patched.dsa_indexer_uses_fp4
    config = types.SimpleNamespace(
        attention_config=types.SimpleNamespace(resolve_indexer_kv_dtype=lambda _: dtype)
    )
    assert consumer.dsa_indexer_uses_fp4(config) is (dtype == "mxfp4")


@pytest.mark.parametrize("consumer_name", FP4_CONSUMERS)
def test_fp4_capability_preserves_other_consumer_overrides(
    modules, fp4_indexer, consumer_name
):
    def override(config):
        return "other override"

    consumer = modules(consumer_name, dsa_indexer_uses_fp4=override)
    harness.load_patch("vllm_sail/patch/enhancement/attention/mla_indexer.py")
    assert consumer.dsa_indexer_uses_fp4 is override


@pytest.mark.parametrize(
    "ppu,capability,dtype,expected,error",
    [
        (True, 80, "mxfp4", None, "requires capability sm_89"),
        (True, 89, "bf16", None, "Unsupported PPU indexer_kv_dtype"),
        (False, 100, "mxfp4", True, None),
        (False, 90, "mxfp4", None, "requires Blackwell datacenter GPUs"),
        (False, 90, "fp8", False, None),
    ],
)
def test_fp4_capability_keeps_platform_and_dtype_gates(
    fp4_indexer, ppu, capability, dtype, expected, error
):
    platform, provider, _ = fp4_indexer
    platform.ppu, platform.capability = ppu, capability
    harness.load_patch("vllm_sail/patch/enhancement/attention/mla_indexer.py")
    config = types.SimpleNamespace(
        attention_config=types.SimpleNamespace(resolve_indexer_kv_dtype=lambda _: dtype)
    )
    if error:
        with pytest.raises(ValueError, match=error):
            provider.dsa_indexer_uses_fp4(config)
    else:
        assert provider.dsa_indexer_uses_fp4(config) is expected


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "ppu,capability,deep_gemm,expected",
    [
        (True, 89, True, True),
        (True, 89, False, False),
        (False, 90, True, True),
        (False, 80, True, False),
    ],
)
def test_adaptive_indexer_gate_and_flattening(
    modules,
    patch_utils_module,
    upstream_source_root,
    ppu,
    capability,
    deep_gemm,
    expected,
):
    platform = types.SimpleNamespace(
        is_ppu=lambda: ppu,
        is_cuda=lambda: True,
        is_rocm=lambda: False,
        is_device_capability_family=lambda value: capability == value,
    )
    modules("vllm.platforms", current_platform=platform)
    modules("torch", float32=object())
    modules("vllm_sail.utils.deep_gemm", is_deep_gemm_supported=lambda: deep_gemm)
    modules("vllm_sail.patch.utils", patch_value=patch_utils_module.patch_value)
    builder = type(
        "DeepseekV32IndexerMetadataBuilder",
        (),
        {
            "__init__": lambda *a, **kw: None,
            "build": lambda *a, **kw: None,
            "_split_indexer_prefill_chunks": staticmethod(lambda *a, **kw: None),
        },
    )
    target = modules(
        "vllm.v1.attention.backends.mla.indexer",
        DeepseekV32IndexerMetadataBuilder=builder,
        current_platform=platform,
        has_deep_gemm=lambda: deep_gemm,
        get_paged_mqa_logits_metadata=lambda *a, **kw: None,
        dsa_indexer_uses_fp4=lambda *a: False,
        native_next_n_supported=lambda n: n in (1, 2),
    )
    source = upstream_source_root / "vllm/v1/attention/backends/mla/indexer.py"
    for name in (
        "_supports_varlen_paged_mqa_logits",
        "_supports_flattened_device_query_lens",
        "_rocm_supports_flattened_device_query_lens",
        "_supports_native_decode",
        "_use_flattening",
    ):
        function(source, name, target.__dict__)
    capability_fn = function(
        source,
        "DeepseekV32IndexerBackend.supports_device_cpu_query_lens_mismatch",
        target.__dict__,
    )
    harness.load_patch("vllm_sail/patch/enhancement/attention/mla_indexer.py")
    assert capability_fn(None) is expected
    # Even next_n=2, normally native, must flatten with adaptive trimming.
    config = types.SimpleNamespace(
        num_speculative_tokens=1,
        model_config=None,
        speculative_config=types.SimpleNamespace(enable_adaptive_verification=True),
    )
    assert target._use_flattening(config) is expected
    # Ordinary next_n=2 decode must keep its native upstream behavior.
    config.speculative_config.enable_adaptive_verification = False
    assert target._use_flattening(config) is False
    # SAIL does not implement Blackwell's varlen paged-logits interface.
    assert target._supports_varlen_paged_mqa_logits() is False


@pytest.mark.parametrize("subclass", [False, True])
def test_rebased_indexer_init_calls_parent_once(
    monkeypatch, patch_utils_module, subclass
):
    """Execute the real copied body, globals rebasing and patch installation.

    Stop in the parent constructor: reproducing the missing class cell needs
    neither vLLM configuration nor device allocations later in the body.
    The subclass case also catches the recursive super(type(self), self) fix.
    """
    calls = []

    class ParentReached(Exception):
        pass

    class Parent:
        def __init__(self, *args, **kwargs):
            calls.append((self, args, kwargs))
            raise ParentReached

    class DeepseekV32IndexerMetadataBuilder(Parent):
        pass

    class DerivedBuilder(DeepseekV32IndexerMetadataBuilder):
        pass

    target = types.ModuleType("_test_mla_indexer_init_target")
    target.DeepseekV32IndexerMetadataBuilder = DeepseekV32IndexerMetadataBuilder
    monkeypatch.setitem(sys.modules, target.__name__, target)
    tree = ast.parse(PATCH_PATH.read_text())
    selected = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("_with_target_globals", "_builder_init_body")
        ],
        type_ignores=[],
    )
    from vllm_sail.patch.bodies import bind_body

    namespace = {"bind_body": bind_body, "_indexer_module": target}
    exec(compile(selected, str(PATCH_PATH), "exec"), namespace)
    replacement = namespace["_with_target_globals"](namespace["_builder_init_body"])
    patch_utils_module.patch(
        target.__name__,
        "DeepseekV32IndexerMetadataBuilder.__init__",
        reason="exercise the copied indexer constructor's parent dispatch",
        affected_versions=">=0.27.0,<0.28.0",
        remove_when="the indexer constructor no longer needs a copied body",
    )(replacement)

    builder_class = DerivedBuilder if subclass else DeepseekV32IndexerMetadataBuilder
    config = object()
    with pytest.raises(ParentReached):
        builder_class(config, block_table_width=29, device="test-device")
    assert len(calls) == 1
    assert type(calls[0][0]) is builder_class
    assert calls[0][1:] == ((config,), {"device": "test-device"})
    assert replacement.__globals__ is target.__dict__


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "local,local_name,upstream,upstream_name",
    [
        (
            "attention/mla_indexer",
            "_builder_init_body",
            "v1/attention/backends/mla/indexer",
            "DeepseekV32IndexerMetadataBuilder.__init__",
        ),
        (
            "attention/mla_indexer",
            "_split_indexer_prefill_chunks_body",
            "v1/attention/backends/mla/indexer",
            "DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks",
        ),
    ],
)
def test_replacements_accept_upstream_keywords(
    upstream_source_root, local, local_name, upstream, upstream_name
):
    assert_accepts_upstream_keywords(
        local, local_name, upstream_source_root, upstream, upstream_name
    )
