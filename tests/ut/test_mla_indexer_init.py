# SPDX-License-Identifier: Apache-2.0
"""Regression for the delegating indexer metadata-builder constructor patch."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch
from tests.support.source import assert_accepts_upstream_keywords

PATCH = "vllm_sail/patch/enhancement/attention/mla_indexer.py"
TARGET = "vllm.v1.attention.backends.mla.indexer"


class Tensor:
    def __init__(self, shape, **kwargs):
        self.shape = shape
        self.dtype = kwargs.get("dtype")
        self.device = kwargs.get("device")


@pytest.fixture
def builder_factory(modules, patch_utils_module):
    def make(*, ppu, fp4=True, n_head=64, q_head_dim=128, typed=False):
        calls = []

        class Builder:
            def __init__(self, *args, **kwargs):
                calls.append((args, kwargs))
                self.kv_cache_spec = NS(
                    indexer_n_head=n_head,
                    indexer_q_head_dim=q_head_dim,
                    num_states=256,
                )
                self.device = "ppu:0"
                self.indexer_uses_fp4 = fp4
                self.use_flattening = False
                self.vllm_config = NS(
                    model_config=NS(
                        hf_text_config=NS(index_n_heads=32, index_head_dim=128)
                    )
                )
                self.supports_varlen = True
                self.num_sms = 3
                self.scheduler_metadata_buffer = NS(shape=("upstream",))

            @staticmethod
            def _split_indexer_prefill_chunks(*args, **kwargs):
                return []

            def build(self, *args, **kwargs):
                return NS(decode=None)

            def update_draft_decode_metadata(self, metadata):
                return None

        modules(
            "torch",
            empty=lambda shape, **kwargs: Tensor(shape, **kwargs),
            bfloat16="bfloat16",
            float32="float32",
            int32="int32",
            int8="int8",
            float8_e4m3fn="float8_e4m3fn",
        )
        modules(
            "vllm.platforms",
            current_platform=NS(
                is_ppu=lambda: ppu,
                is_device_capability=lambda capability: capability == 89,
            ),
        )
        modules(
            "vllm.v1.attention.backend",
            CommonAttentionMetadata=type("CommonAttentionMetadata", (), {}),
        )
        provider = modules(
            TARGET,
            DeepseekV32IndexerMetadataBuilder=Builder,
            DeepseekV32IndexerMetadata=type("DeepseekV32IndexerMetadata", (), {}),
            DSA_INDEXER_KV_DTYPES=("fp8", "mxfp4"),
            get_paged_mqa_logits_metadata=lambda *args, **kwargs: None,
            dsa_indexer_uses_fp4=lambda config: False,
            _supports_flattened_device_query_lens=lambda: "upstream-query-lengths",
        )
        modules(
            "vllm_sail.utils.deep_gemm",
            get_num_sms=lambda: 7,
            has_typed_mqa_logits_metadata=lambda: typed,
            has_deep_gemm=lambda: True,
            is_deep_gemm_supported=lambda: True,
            get_paged_mqa_logits_metadata=lambda *args: None,
        )
        modules(
            "vllm_sail.patch.utils",
            patch_value=patch_utils_module.patch_value,
        )
        load_patch(PATCH)
        return provider.DeepseekV32IndexerMetadataBuilder, calls

    return make


def test_ppu_fp4_constructor_delegates_then_adds_fork_state(builder_factory):
    builder_cls, calls = builder_factory(ppu=True, fp4=True)
    builder = builder_cls("config", block_table_width=29, device="ppu:0")

    assert calls == [(("config",), {"block_table_width": 29, "device": "ppu:0"})]
    assert builder.supports_varlen is False
    assert builder.indexer_n_head == 64
    assert builder.indexer_q_head_dim == 128
    assert builder.num_sms == 7
    assert builder.scheduler_metadata_buffer.shape == (7 * 8 + 1, 2)
    assert builder.scheduler_metadata_buffer.dtype == "int32"
    assert builder.scheduler_metadata_buffer.device == "ppu:0"


def test_ppu_fp8_constructor_keeps_upstream_buffer(builder_factory):
    builder_cls, _ = builder_factory(ppu=True, fp4=False)
    builder = builder_cls("config")

    assert builder.supports_varlen is False
    assert builder.indexer_n_head == 0
    assert builder.indexer_q_head_dim == 0
    assert builder.num_sms == 7
    assert builder.scheduler_metadata_buffer.shape == ("upstream",)


@pytest.mark.parametrize("fp4,qk_dtype", [(True, "int8"), (False, "float8_e4m3fn")])
def test_typed_metadata_uses_geometry_and_reserves_schedule_buffer(
    builder_factory, fp4, qk_dtype
):
    builder_cls, _ = builder_factory(ppu=True, fp4=fp4, n_head=None, typed=True)
    builder = builder_cls("config")

    assert builder.indexer_n_head == 32
    assert builder.indexer_q_head_dim == 128
    assert builder.indexer_qk_dtype == qk_dtype
    assert builder.use_flattening is fp4
    assert builder.scheduler_metadata_buffer.shape == (7 * 8 + 1, 2)


def test_non_ppu_constructor_is_left_untouched(builder_factory):
    builder_cls, calls = builder_factory(ppu=False)
    builder = builder_cls("config")

    assert len(calls) == 1
    assert builder.supports_varlen is True
    assert builder.num_sms == 3
    assert builder.scheduler_metadata_buffer.shape == ("upstream",)
    assert not hasattr(builder, "indexer_n_head")


def test_subclass_constructs_through_the_parent_once(builder_factory):
    builder_cls, calls = builder_factory(ppu=True)

    class Derived(builder_cls):
        pass

    builder = Derived("config", block_table_width=29)

    assert isinstance(builder, Derived)
    assert len(calls) == 1
    assert calls[0][1] == {"block_table_width": 29}


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "local,local_name,upstream,upstream_name",
    [
        (
            "attention/mla_indexer",
            "_builder_init_wrapper",
            "v1/attention/backends/mla/indexer",
            "DeepseekV32IndexerMetadataBuilder.__init__",
        ),
        (
            "attention/mla_indexer",
            "_build_wrapper",
            "v1/attention/backends/mla/indexer",
            "DeepseekV32IndexerMetadataBuilder.build",
        ),
        (
            "attention/mla_indexer",
            "_update_draft_decode_wrapper",
            "v1/attention/backends/mla/indexer",
            "DeepseekV32IndexerMetadataBuilder.update_draft_decode_metadata",
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
