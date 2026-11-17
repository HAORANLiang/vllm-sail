# SPDX-License-Identifier: Apache-2.0
"""The installed DSA indexer patch resolves PPU auto dtype through its provider."""

from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch


class AttentionConfig:
    def __init__(self, value):
        self.indexer_kv_dtype = value

    def resolve_indexer_kv_dtype(self, default):
        return default if self.indexer_kv_dtype == "auto" else self.indexer_kv_dtype


def config(value, model_type="deepseek_v4"):
    return NS(
        attention_config=AttentionConfig(value),
        model_config=NS(hf_text_config=NS(model_type=model_type)),
    )


@pytest.fixture
def installed_indexer(modules, patch_utils_module):
    def load(
        capability, *, ppu=True, consumer_name="vllm.models.deepseek_v41.attention"
    ):
        calls = []
        platform = NS(
            is_ppu=lambda: ppu, is_device_capability=lambda cap: capability == cap
        )
        modules(
            "torch",
            float32="fp32",
            int8="int8",
            float8_e4m3fn="float8_e4m3fn",
        )
        modules("vllm.platforms", current_platform=platform)
        modules(
            "vllm.v1.attention.backend",
            CommonAttentionMetadata=type("CommonAttentionMetadata", (), {}),
        )
        modules("vllm_sail.patch.utils", patch_value=patch_utils_module.patch_value)
        modules("vllm_sail.utils.deep_gemm")

        class Builder:
            def __init__(self, *args):
                raise AssertionError("not constructing metadata in dtype test")

            @staticmethod
            def _split_indexer_prefill_chunks(*args, **kwargs):
                raise AssertionError("not splitting prefills in dtype test")

            def build(self, *args):
                raise AssertionError("not building a step in dtype test")

            def update_draft_decode_metadata(self, metadata):
                raise AssertionError("not refreshing draft metadata in dtype test")

            def _prepare_decode_tensors(self, *args, **kwargs):
                raise AssertionError("not preparing decode tensors in dtype test")

        def upstream(value):
            calls.append(value)
            return "upstream"

        provider = modules(
            "vllm.v1.attention.backends.mla.indexer",
            DeepseekV32IndexerMetadataBuilder=Builder,
            DeepseekV32IndexerMetadata=type("DeepseekV32IndexerMetadata", (), {}),
            get_paged_mqa_logits_metadata=lambda *args, **kwargs: None,
            dsa_indexer_uses_fp4=upstream,
            _supports_flattened_device_query_lens=lambda: False,
            _supports_varlen_paged_mqa_logits=lambda: False,
            DSA_INDEXER_KV_DTYPES=("fp8", "mxfp4"),
        )
        consumer = modules(consumer_name, dsa_indexer_uses_fp4=upstream)
        patch = load_patch("vllm_sail/patch/enhancement/attention/mla_indexer.py")
        assert provider.dsa_indexer_uses_fp4 is patch.dsa_indexer_uses_fp4
        assert consumer.dsa_indexer_uses_fp4 is patch.dsa_indexer_uses_fp4
        target = f"{consumer_name}.dsa_indexer_uses_fp4"
        assert (
            patch_utils_module.original_of(consumer.dsa_indexer_uses_fp4, target)
            is upstream
        )
        with pytest.raises(RuntimeError, match="already"):
            patch_utils_module.patch(
                consumer_name,
                "dsa_indexer_uses_fp4",
                reason="Reject duplicate capability rebinding.",
                affected_versions=">=0.32.0,<0.33.0",
                remove_when="The consumer resolves the provider dynamically.",
            )(lambda config: False)
        return provider.dsa_indexer_uses_fp4, calls

    return load


def test_preloaded_sparse_layer_uses_ppu_fp4_capability(installed_indexer):
    uses_fp4, calls = installed_indexer(
        89, consumer_name="vllm.model_executor.layers.sparse_mqa_indexer"
    )
    assert uses_fp4(config("mxfp4", "deepseek_v41")) is True
    assert calls == []


@pytest.mark.parametrize(
    "capability,configured,expected",
    [
        (89, "auto", True),
        (80, "auto", False),
        (89, "fp8", False),
        (80, "fp8", False),
        (89, "mxfp4", True),
    ],
)
def test_indexer_cache_dtype_resolution(
    installed_indexer, capability, configured, expected
):
    uses_fp4, calls = installed_indexer(capability)
    assert uses_fp4(config(configured)) is expected
    assert calls == []


def test_mxfp4_stays_rejected_below_sm_89(installed_indexer):
    uses_fp4, _ = installed_indexer(80)
    with pytest.raises(ValueError, match="requires capability sm_89"):
        uses_fp4(config("mxfp4"))


@pytest.mark.parametrize(
    "model_type", ["deepseek_v4", "deepseek_v41", "deepseek_v41_text"]
)
def test_mxfp4_default_is_available_to_both_deepseek_producers(
    installed_indexer, model_type
):
    uses_fp4, _ = installed_indexer(89)
    assert uses_fp4(config("auto", model_type)) is True


@pytest.mark.parametrize("model_type", ["glm5", "glm5_next", "deepseek_v3", "kimi_k3"])
def test_fixed_fp8_models_do_not_inherit_deepseek_mxfp4_metadata(
    installed_indexer, model_type
):
    uses_fp4, _ = installed_indexer(89)
    assert uses_fp4(config("auto", model_type)) is False
    assert uses_fp4(config("fp8", model_type)) is False
    with pytest.raises(ValueError, match="requires DeepSeek V4 or V4.1"):
        uses_fp4(config("mxfp4", model_type))


def test_non_ppu_delegates_to_upstream(installed_indexer):
    uses_fp4, calls = installed_indexer(100, ppu=False)
    value = config("auto")
    assert uses_fp4(value) == "upstream"
    assert calls == [value]
