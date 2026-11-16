# SPDX-License-Identifier: Apache-2.0
"""Exercise QSA construction with differently quantized QKV/indexer shards."""

import re
from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch


@pytest.fixture
def construct_qsa(modules, patch_utils_module):
    state = NS(ppu=True, projections=[])

    class Module:
        def register_buffer(self, name, value, **kwargs):
            setattr(self, name, value)

    class Attention(Module):
        def __init__(self, **kwargs):
            raise AssertionError("the installed constructor must run")

    def ignored(name, ignore):
        return any(
            re.match(entry[3:], name) if entry.startswith("re:") else entry == name
            for entry in ignore
        )

    def should_ignore_layer(layer_name, ignore=(), fused_mapping=None):
        return ignored(layer_name, ignore)

    def linear(*args, quant_config=None, prefix="", **kwargs):
        if prefix.endswith("qkv_proj") and quant_config is not None:
            shards = quant_config.packed_modules_mapping["qkv_proj"]
            if getattr(quant_config, "fp8_channelwise_layers", None):
                selected = [
                    name in quant_config.fp8_channelwise_layers for name in shards
                ]
            else:
                base = prefix[: -len("qkv_proj")]
                selected = [
                    ignored(base + name, quant_config.ignore) for name in shards
                ]
            if any(selected) and not all(selected):
                raise ValueError("some but not all shards are quantized")
        state.projections.append((prefix, quant_config))
        return NS(output_size=16)

    modules(
        "vllm.model_executor.layers.quantization.compressed_tensors.utils",
        should_ignore_layer=should_ignore_layer,
    )

    def fused(qkv, size, quant):
        state.fused = True
        return NS(output_size=32)

    modules("vllm.models.qwen4_exp.nvidia.model")
    qsa = modules(
        "vllm.models.qwen4_exp.nvidia.qsa",
        Qwen4ExpQSAAttention=Attention,
        nn=NS(Module=Module),
        torch=NS(
            bfloat16="bf16",
            float16="fp16",
            uint8="u8",
            int32="i32",
            tensor=lambda *a, **k: [],
            empty=lambda *a, **k: [],
        ),
        current_platform=NS(
            is_ppu=lambda: state.ppu,
            is_cuda=lambda: True,
            has_device_capability=lambda cap: True,
        ),
        get_tensor_model_parallel_world_size=lambda: 4,
        QKVParallelLinear=linear,
        RowParallelLinear=linear,
        model=NS(without_modelopt_fp4=lambda config: config),
        get_rope=lambda **kwargs: NS(),
        mrope_positions_factor=lambda config: 1,
        GemmaRMSNorm=lambda *a, **k: NS(),
        MRotaryEmbedding=type("MRotaryEmbedding", (), {}),
        AttentionType=NS(DECODER="decoder"),
        qsa_kv_cache_dtype=lambda config, prefix: "auto",
        kv_cache_dtype_str_to_dtype=lambda *a: "bf16",
        set_default_quant_scales=lambda *a, **k: None,
        Qwen4ExpQSAFlashAttentionBackend=object,
        Qwen4ExpQSAFlashAttentionImpl=lambda *a: NS(),
        QSAIndexer=lambda **k: NS(
            use_fused_pre_indexer=False,
            index_qk_proj=NS(output_size=8),
            packed_output_width=4,
        ),
        Qwen4ExpQSAQKVIndexerLinear=fused,
    )
    load_patch("vllm_sail/patch/enhancement/models/qwen4_qsa_quant.py")

    def run(quant, *, ppu=True, lora=None):
        state.ppu = ppu
        state.fused = False
        config = NS(
            hidden_size=1024,
            num_attention_heads=8,
            num_key_value_heads=8,
            head_dim=128,
            max_position_embeddings=4096,
            rope_parameters={},
            rms_norm_eps=1e-6,
        )
        vconfig = NS(
            cache_config=NS(cache_dtype="auto"),
            model_config=NS(dtype="bf16", multimodal_config=None),
            parallel_config=NS(
                prefill_context_parallel_size=1, decode_context_parallel_size=1
            ),
            num_speculative_tokens=3,
            lora_config=lora,
            scheduler_config=NS(max_num_batched_tokens=8),
            compilation_config=NS(static_forward_context={}),
        )
        attn = qsa.Qwen4ExpQSAAttention(
            vllm_config=vconfig,
            config=config,
            layer_id=3,
            quant_config=quant,
            prefix="self_attn",
        )
        return attn, state

    return run


def mixed_quant():
    return NS(
        fp8_channelwise_layers=["q_proj", "k_proj", "v_proj"],
        packed_modules_mapping={
            "qkv_proj": ["q_proj", "k_proj", "v_proj", "indexer.index_qk_proj"]
        },
    )


def test_channelwise_qkv_keeps_bf16_indexer_separate(construct_qsa):
    quant = mixed_quant()
    attn, state = construct_qsa(quant)
    assert not attn.fuse_indexer_projection
    assert hasattr(attn.indexer, "index_qk_proj")
    assert not state.fused
    assert quant.packed_modules_mapping["qkv_proj"][-1] == "indexer.index_qk_proj"
    assert state.projections[0][1] is not quant


def test_unquantized_projection_keeps_upstream_fusion(construct_qsa):
    attn, state = construct_qsa(None)
    assert attn.fuse_indexer_projection and state.fused
    assert not hasattr(attn.indexer, "index_qk_proj")


def test_non_ppu_preserves_mixed_precision_rejection(construct_qsa):
    with pytest.raises(ValueError, match="some but not all shards"):
        construct_qsa(mixed_quant(), ppu=False)


def int8_quant(ignore):
    return NS(
        ignore=ignore,
        packed_modules_mapping={
            "qkv_proj": ["q_proj", "k_proj", "v_proj", "indexer.index_qk_proj"]
        },
    )


@pytest.mark.parametrize(
    "ignore",
    [
        ["self_attn.indexer.index_qk_proj"],
        ["re:.*indexer\\.index_qk_proj$"],
    ],
)
def test_int8_qkv_keeps_ignored_indexer_separate(construct_qsa, ignore):
    quant = int8_quant(ignore)
    attn, state = construct_qsa(quant)
    assert not attn.fuse_indexer_projection and not state.fused
    assert hasattr(attn.indexer, "index_qk_proj")
    qkv_quant = state.projections[0][1]
    assert qkv_quant is not quant
    assert qkv_quant.packed_modules_mapping["qkv_proj"] == [
        "q_proj",
        "k_proj",
        "v_proj",
    ]
    assert quant.packed_modules_mapping["qkv_proj"][-1] == "indexer.index_qk_proj"


@pytest.mark.parametrize(
    "ignore",
    [
        ["lm_head"],
        [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.indexer.index_qk_proj",
        ],
    ],
)
def test_homogeneous_ignore_keeps_upstream_fusion(construct_qsa, ignore):
    attn, state = construct_qsa(int8_quant(ignore))
    assert attn.fuse_indexer_projection and state.fused
    assert not hasattr(attn.indexer, "index_qk_proj")


def test_non_ppu_preserves_int8_mixed_rejection(construct_qsa):
    with pytest.raises(ValueError, match="some but not all shards"):
        construct_qsa(int8_quant(["self_attn.indexer.index_qk_proj"]), ppu=False)


def test_int8_ignored_qkv_keeps_quantized_indexer_separate(construct_qsa):
    quant = int8_quant(
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
    )
    attn, state = construct_qsa(quant)
    assert not attn.fuse_indexer_projection and not state.fused
    assert hasattr(attn.indexer, "index_qk_proj")
    qkv_quant = state.projections[0][1]
    assert qkv_quant is not quant
    assert qkv_quant.packed_modules_mapping["qkv_proj"] == [
        "q_proj",
        "k_proj",
        "v_proj",
    ]


@pytest.mark.parametrize(
    "ignore",
    [
        ["self_attn.q_proj"],
        ["self_attn.k_proj", "self_attn.indexer.index_qk_proj"],
        ["re:.*self_attn\\.v_proj$"],
    ],
)
def test_mixed_qkv_shards_raise_clear_error(construct_qsa, ignore):
    with pytest.raises(ValueError, match="ignores only some of q/k/v"):
        construct_qsa(int8_quant(ignore))


def test_lora_keeps_unfused_projection(construct_qsa):
    attn, state = construct_qsa(None, lora=object())
    assert not attn.fuse_indexer_projection and not state.fused
