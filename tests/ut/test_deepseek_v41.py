# SPDX-License-Identifier: Apache-2.0
"""V4.1 integration contracts; CPU tests never import torch or vLLM."""

from types import SimpleNamespace

import pytest

from tests.support.source import function
from tests.ut import test_ppu_kernel_capabilities as harness

modules = harness.modules
load_patch = harness.load_patch


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("record_bytes", [528, 584])
def test_cache_gather_dispatch_after_preload(
    modules, upstream_source_root, ppu, record_bytes
):
    calls = []
    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: ppu))
    modules(
        "vllm.utils.import_utils", has_cutedsl=lambda: True, has_humming=lambda: True
    )
    provider = modules("vllm.utils.import_utils")
    cache = modules(
        "vllm.models.deepseek_v41.common.ops.cache_utils",
        has_cutedsl=provider.has_cutedsl,
        V41_BYTES_PER_TOKEN=528,
        V41_NVFP4_BYTES_PER_TOKEN=288,
        V41_QUANT_BLOCK=32,
        dequantize_and_gather_k_cache_triton=lambda *a, **kw: calls.append("triton"),
    )
    modules(
        "vllm.models.deepseek_v4.nvidia.ops.dequant_gather_k_cutedsl",
        _DEQUANT_GATHER_K_CACHE_CUTEDSL_KERNEL=lambda **kw: calls.append("cute"),
    )
    gather = function(
        upstream_source_root / "vllm/models/deepseek_v41/common/ops/cache_utils.py",
        "dequantize_and_gather_k_cache",
        cache.__dict__,
    )
    load_patch("vllm_sail/patch/enhancement/import_gates.py")
    gather(None, SimpleNamespace(shape=(4, 64, record_bytes)), None, None, None, 64, 3)
    assert calls == ["triton" if ppu else "cute"]


@pytest.fixture
def quant_module(modules):
    calls = []

    class Upstream:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        @classmethod
        def override_quantization_method(cls, *args):
            calls.append(("recognize", args))
            return "upstream-method"

        @classmethod
        def from_config(cls, config):
            calls.append(("config", config))
            return cls()

        def get_quant_method(self, layer, prefix):
            calls.append(("dispatch", layer, prefix))
            return "upstream-kernel"

    modules("vllm.models.deepseek_v41.quant_config", DeepseekV4FP8Config=Upstream)
    modules("vllm.model_executor.layers.fused_moe", RoutedExperts=type("MoE", (), {}))
    modules(
        "vllm.model_executor.layers.linear",
        LinearBase=type("Linear", (), {}),
        UnquantizedLinearMethod=object,
    )
    modules(
        "vllm.model_executor.layers.quantization.utils.quant_utils",
        is_layer_skipped=lambda **kwargs: False,
    )
    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: True))
    return load_patch("vllm_sail/models/deepseek_v41/quant_config.py"), calls


def test_standard_quantization_delegates_without_rewriting_config(quant_module):
    module, calls = quant_module
    cls = module.DeepseekV41FP8Config
    checkpoint = {"quant_method": "fp8", "weight_block_size": [32, 32]}
    hf = SimpleNamespace(model_type="deepseek_v41")
    assert cls.override_quantization_method(checkpoint, None, hf) == "deepseek_v41_fp8"
    config = cls.from_config(checkpoint)
    layer = object()
    assert config.get_quant_method(layer, "model.ffn") == "upstream-kernel"
    assert calls == [
        ("recognize", (checkpoint, None, hf)),
        ("config", checkpoint),
        ("dispatch", layer, "model.ffn"),
    ]
    assert calls[1][1] is checkpoint


@pytest.mark.parametrize(
    "model_type,architectures",
    [
        ("deepseek_v41", ["DeepseekV41ForCausalLM"]),
        ("deepseek_v41_text", []),
        # SpeculativeConfig verifies this temporary config before rewriting
        # it to deepseek_v41 / DSparkV41DraftModel.
        ("deepseek_mtp", ["DeepseekV41ForCausalLM"]),
    ],
)
def test_qlean_metadata_is_translated_without_editing_checkpoint(
    quant_module, model_type, architectures
):
    module, calls = quant_module
    cls = module.DeepseekV41FP8Config
    cls.__bases__[0].override_quantization_method = classmethod(lambda *args: None)
    checkpoint = {
        "quant_method": "mxfp4",
        "expert_dtype": "mxfp4",
        "fp8_channelwise_layers": ["layers.0.attn.wo_a"],
        "ignore": ["layers.0.attn.compressor.wkv"],
    }
    assert (
        cls.override_quantization_method(
            checkpoint,
            None,
            SimpleNamespace(model_type=model_type, architectures=architectures),
        )
        == "deepseek_v41_fp8"
    )
    result = cls.from_config(checkpoint)
    translated = calls[-1][1]
    assert translated["quant_method"] == "fp8"
    assert translated["activation_scheme"] == "dynamic"
    assert translated["ignored_layers"] == checkpoint["ignore"]
    assert result.fp8_channelwise_layers == checkpoint["fp8_channelwise_layers"]
    assert (
        checkpoint["quant_method"] == "mxfp4" and "activation_scheme" not in checkpoint
    )


@pytest.mark.parametrize("quant_method", ["fp8", "mxfp4"])
@pytest.mark.parametrize(
    "ppu,model_type,architectures",
    [
        (False, "deepseek_v41", ["DeepseekV41ForCausalLM"]),
        (True, "deepseek_v4", ["DeepseekV4ForCausalLM"]),
        (True, "deepseek_v4_text", []),
        (True, "deepseek_mtp", ["DeepSeekV4MTPModel"]),
        (True, "deepseek_mtp", ["DeepSeekMTPModel"]),
        (True, "other", []),
    ],
)
def test_v41_quant_override_leaves_other_models_and_platforms_untouched(
    quant_module, quant_method, ppu, model_type, architectures
):
    module, calls = quant_module
    module.current_platform.is_ppu = lambda: ppu
    assert (
        module.DeepseekV41FP8Config.override_quantization_method(
            {
                "quant_method": quant_method,
                "fp8_channelwise_layers": ["layers.0.attn.wq_a"],
            },
            None,
            SimpleNamespace(model_type=model_type, architectures=architectures),
        )
        is None
    )
    assert calls == []


@pytest.mark.parametrize("ppu", [False, True])
def test_quant_registration_is_lazy_idempotent_and_preserves_v4(modules, ppu):
    calls = []
    modules("vllm.logger", init_logger=lambda _: SimpleNamespace(debug=lambda *a: None))
    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: ppu))
    modules(
        "vllm.model_executor.layers.quantization",
        QUANTIZATION_METHODS=[],
        register_quantization_config=lambda name: lambda cls: calls.append((name, cls)),
    )
    mixed = type("Mixed", (), {"get_name": staticmethod(lambda: "mixed_precision_w4")})
    v4 = type("V4", (), {})
    v41 = type("V41", (), {"get_name": staticmethod(lambda: "deepseek_v41_fp8")})
    modules(
        "vllm_sail.model_executor.layers.quantization.mixed_precision_w4",
        MixedPrecisionW4Config=mixed,
    )
    modules("vllm_sail.models.deepseek_v4.quant_config", DeepseekV4FP8Config=v4)
    modules("vllm_sail.models.deepseek_v41.quant_config", DeepseekV41FP8Config=v41)
    registry = load_patch("vllm_sail/registry/quant_config/__init__.py")
    assert calls == []
    registry.register()
    registry.register()
    assert calls == [("mixed_precision_w4", mixed)] + (
        [("deepseek_v4_fp8", v4), ("deepseek_v41_fp8", v41)] if ppu else []
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize("quant_method", ["fp8", "deepseek_v4_fp8", "mxfp4"])
@pytest.mark.parametrize(
    "model_type,architectures,expected",
    [
        ("deepseek_v41", ["DeepseekV41ForCausalLM"], "deepseek_v41_fp8"),
        ("deepseek_v41_text", [], "deepseek_v41_fp8"),
        ("deepseek_mtp", ["DeepseekV41ForCausalLM"], "deepseek_v41_fp8"),
        ("deepseek_v4", ["DeepseekV4ForCausalLM"], "deepseek_v4_fp8"),
    ],
)
def test_upstream_model_config_selects_model_specific_quantization(
    quant_module,
    modules,
    upstream_source_root,
    quant_method,
    model_type,
    architectures,
    expected,
):
    """Exercise upstream recognition and override ordering, not a reimplemented selector."""
    from typing import Literal, cast, get_args

    module, _ = quant_module
    cls = module.DeepseekV41FP8Config
    upstream = modules("vllm.models.deepseek_v41.quant_config").DeepseekV4FP8Config
    assert cls.__bases__ == (upstream,)
    upstream.override_quantization_method = classmethod(
        function(
            upstream_source_root / "vllm/models/deepseek_v41/quant_config.py",
            "DeepseekV4FP8Config.override_quantization_method",
            {},
        )
    )

    class UpstreamV4(upstream):
        pass

    UpstreamV4.override_quantization_method = classmethod(
        function(
            upstream_source_root / "vllm/models/deepseek_v4/quant_config.py",
            "DeepseekV4FP8Config.override_quantization_method",
            {},
        )
    )
    modules("vllm.models.deepseek_v4.quant_config", DeepseekV4FP8Config=UpstreamV4)
    v4 = load_patch("vllm_sail/models/deepseek_v4/quant_config.py").DeepseekV4FP8Config
    assert v4.__bases__ == (UpstreamV4,)
    platform = module.current_platform
    platform.supported_quantization = ["fp8", "mxfp4", "deepseek_v4_fp8"]
    verified = []
    platform.verify_quantization = verified.append
    registry = {}
    names = list(platform.supported_quantization)
    register = function(
        upstream_source_root / "vllm/model_executor/layers/quantization/__init__.py",
        "register_quantization_config",
        {
            "QUANTIZATION_METHODS": names,
            "_CUSTOMIZED_METHOD_TO_QUANT_CONFIG": registry,
            "QuantizationConfig": upstream,
            "current_platform": platform,
        },
    )
    register(cls.get_name())(cls)
    assert cls.get_name() in platform.supported_quantization
    registry["deepseek_v4_fp8"] = v4
    no_override = SimpleNamespace(override_quantization_method=lambda *a, **kw: None)
    quant = SimpleNamespace(
        QUANTIZATION_METHODS=names,
        QuantizationMethods=Literal["fp8", "mxfp4", "deepseek_v4_fp8"],
        DEPRECATED_QUANTIZATION_METHODS=[],
        get_quantization_config=lambda name: registry.get(name, no_override),
    )
    verify = function(
        upstream_source_root / "vllm/config/model.py",
        "ModelConfig._verify_quantization",
        {
            "me_quant": quant,
            "get_args": get_args,
            "cast": cast,
            "current_platform": platform,
        },
    )
    checkpoint = {"quant_method": quant_method}
    if quant_method == "mxfp4":
        checkpoint["fp8_channelwise_layers"] = ["layers.0.attn.wq_a"]
    config = SimpleNamespace(
        quantization=None,
        model_arch_config=SimpleNamespace(quantization_config=checkpoint),
        hf_config=SimpleNamespace(model_type=model_type, architectures=architectures),
    )
    verify(config)
    assert config.quantization == expected
    assert verified == [expected]
    assert registry[config.quantization] is (cls if expected == cls.get_name() else v4)
    assert checkpoint["quant_method"] == quant_method


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "model_type",
    ["deepseek_v4", "deepseek_v4_text", "deepseek_v41", "deepseek_v41_text"],
)
def test_upstream_recognizes_target_and_text_configs(upstream_source_root, model_type):
    recognize = function(
        upstream_source_root / "vllm/models/deepseek_v41/quant_config.py",
        "DeepseekV4FP8Config.override_quantization_method",
        {},
    )
    assert (
        recognize(
            object,
            {"quant_method": "fp8"},
            None,
            SimpleNamespace(model_type=model_type),
        )
        == "deepseek_v4_fp8"
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize("experts,topk", [(384, 6), (128, 3)])
def test_upstream_mxfp4_dispatch_preserves_target_and_draft_moe(
    upstream_source_root, modules, experts, topk
):
    class RoutedExperts:
        moe_config = SimpleNamespace(num_experts=experts, experts_per_token=topk)

    expected = object()
    modules(
        "vllm.model_executor.layers.quantization.mxfp4",
        Mxfp4MoEMethod=lambda cfg: expected,
    )
    dispatch = function(
        upstream_source_root / "vllm/models/deepseek_v41/quant_config.py",
        "DeepseekV4FP8Config.get_quant_method",
        {
            "LinearBase": type("Linear", (), {}),
            "RoutedExperts": RoutedExperts,
            "is_layer_skipped": lambda **kwargs: False,
            "Mxfp4MoEMethod": lambda cfg: (expected, cfg),
        },
    )
    config = SimpleNamespace(
        ignored_layers=[],
        packed_modules_mapping={},
        expert_dtype="fp4",
        moe_quant_algo="",
    )
    layer = RoutedExperts()
    assert dispatch(config, layer, "model.ffn.experts") == (expected, layer.moe_config)


def test_v41_selector_preserves_cuda_and_other_backends(modules):
    upstream, ppu, other = (type(name, (), {}) for name in ("Upstream", "PPU", "Other"))
    platform = SimpleNamespace(ppu=True)
    modules(
        "vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: platform.ppu)
    )
    modules(
        "vllm.models.deepseek_v41.nvidia.flashmla", DeepseekV4FlashMLAAttention=upstream
    )
    modules("vllm_sail.models.deepseek_v41.flashmla", DeepseekV41FlashMLAAttention=ppu)

    def original(config):
        return config

    provider = modules(
        "vllm.models.deepseek_v41.nvidia.model",
        _select_dsv4_attn_cls=original,
        _linear_scale_param_name=lambda *args: "weight_scale_inv",
    )
    original_scale = provider._linear_scale_param_name
    draft = modules(
        "vllm.models.deepseek_v41.nvidia.dspark",
        _linear_scale_param_name=original_scale,
    )
    vl = modules(
        "vllm.models.deepseek_v41.nvidia.vl_model",
        _linear_scale_param_name=original_scale,
    )
    replacement = load_patch("vllm_sail/patch/enhancement/models/deepseek_v41.py")
    assert provider._select_dsv4_attn_cls is replacement._select_dsv4_attn_cls
    marker = getattr(provider._select_dsv4_attn_cls, replacement.PATCH_MARKER)
    assert marker[replacement._TARGET] is original
    assert provider._select_dsv4_attn_cls(upstream) is ppu
    assert provider._select_dsv4_attn_cls(other) is other
    config = SimpleNamespace(
        quant_config=SimpleNamespace(fp8_channelwise_layers=["attn.wo_a"])
    )
    assert (
        draft._linear_scale_param_name
        is vl._linear_scale_param_name
        is provider._linear_scale_param_name
    )
    assert draft._linear_scale_param_name(config, "fp4") == "weight_scale"
    platform.ppu = False
    assert provider._select_dsv4_attn_cls(upstream) is upstream
    assert draft._linear_scale_param_name(config, "fp4") == "weight_scale_inv"
    with pytest.raises(RuntimeError, match="already patched"):
        load_patch("vllm_sail/patch/enhancement/models/deepseek_v41.py")


@pytest.mark.parametrize("dequant_at_load", [True, False])
def test_v41_projection_preserves_weights_and_uses_bf16_helper(
    modules, dequant_at_load
):
    calls, hook_calls = [], []
    projected, output = object(), object()

    def wo_b_proj(value):
        hook_calls.append(value)
        return output

    def helper(*args, **kwargs):
        calls.append((args, kwargs))
        return args[4](projected)

    loaded = SimpleNamespace(element_size=lambda: 2 if dequant_at_load else 1)
    bf16 = loaded if dequant_at_load else object()
    modules(
        "vllm.model_executor.layers.quantization.utils.mxfp8_utils",
        dequant_mxfp8_to_bf16=lambda *args: bf16,
    )
    modules(
        "vllm.models.deepseek_v4.nvidia.ops.o_proj",
        deep_gemm_fp8_o_proj=helper,
    )

    class Upstream:
        forward_mqa = object()
        _forward_prefill = object()

    modules(
        "vllm.models.deepseek_v41.nvidia.flashmla", DeepseekV4FlashMLAAttention=Upstream
    )
    modules("vllm_sail.models.deepseek_v4.flashmla", DeepseekV4FlashMLAAttention=object)
    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: True))
    module = load_patch("vllm_sail/models/deepseek_v41/flashmla.py")
    attn = module.DeepseekV41FlashMLAAttention()
    attn.__dict__.update(
        _o_proj_block_size=32,
        wo_a=SimpleNamespace(weight=loaded, weight_scale=object()),
        wo_b=object(),
        _wo_b_proj=wo_b_proj,
        rotary_emb=SimpleNamespace(cos_sin_cache=object()),
        n_local_groups=2,
        n_local_heads=16,
        nope_head_dim=448,
        rope_head_dim=64,
        o_lora_rank=1024,
    )
    o, pos = object(), object()
    assert attn._o_proj(o, pos) is output
    assert hook_calls == [projected]
    ((args, kwargs),) = calls
    assert args[:2] == (o, pos) and args[3].weight is bf16
    assert args[4] is attn._wo_b_proj
    assert kwargs["heads_per_group"] == 8 and kwargs["einsum_recipe"] == (1, 1, 32)
    assert attn.wo_a.weight is loaded
    assert attn.forward_mqa is Upstream.forward_mqa
    assert attn._forward_prefill is Upstream._forward_prefill


@pytest.mark.parametrize("has_hook", [False, True])
@pytest.mark.parametrize("quantization", ["int8", "channelwise_fp8", "block_fp8"])
def test_v4_projection_uses_hook_or_raw_layer_once(quantization, has_hook):
    helper_calls, projection_calls = [], []
    projected, raw_output, hook_output = object(), object(), object()

    def raw_layer(value):
        projection_calls.append(("raw", value))
        return raw_output

    def hook(value):
        projection_calls.append(("hook", value))
        return hook_output

    def helper(kind):
        def call(*args, **kwargs):
            helper_calls.append((kind, args, kwargs))
            return args[4](projected)

        return call

    project = function(
        "vllm_sail/models/deepseek_v4/flashmla.py",
        "DeepseekV4FlashMLAAttention._o_proj",
        {
            "torch": SimpleNamespace(int8="int8", float8_e4m3fn="fp8"),
            "deep_gemm_int8_o_proj": helper("int8"),
            "deep_gemm_fp8_channel_o_proj": helper("channelwise_fp8"),
            "deep_gemm_fp8_o_proj": helper("block_fp8"),
        },
    )
    wo_a = SimpleNamespace(
        weight=SimpleNamespace(dtype="int8" if quantization == "int8" else "fp8")
    )
    if quantization == "block_fp8":
        wo_a.weight_scale_inv = object()
    else:
        wo_a.weight_scale = object()
    attn = SimpleNamespace(
        wo_a=wo_a,
        wo_b=raw_layer,
        rotary_emb=SimpleNamespace(cos_sin_cache=object()),
        n_local_groups=2,
        n_local_heads=16,
        nope_head_dim=448,
        rope_head_dim=64,
        o_lora_rank=1024,
        _einsum_recipe=(1, 128, 128),
        _tma_aligned_scales=False,
    )
    if has_hook:
        attn._wo_b_proj = hook
    o, positions = object(), object()
    assert project(attn, o, positions) is (hook_output if has_hook else raw_output)
    assert projection_calls == [("hook" if has_hook else "raw", projected)]
    ((kind, args, kwargs),) = helper_calls
    assert kind == quantization
    assert args == (
        o,
        positions,
        attn.rotary_emb.cos_sin_cache,
        wo_a,
        hook if has_hook else raw_layer,
    )
    expected = {
        "n_groups": 2,
        "heads_per_group": 8,
        "nope_dim": 448,
        "rope_dim": 64,
        "o_lora_rank": 1024,
        "einsum_recipe": attn._einsum_recipe,
    }
    if quantization == "block_fp8":
        expected["tma_aligned_scales"] = attn._tma_aligned_scales
    assert kwargs == expected


def test_v41_draft_inherits_its_own_loader_and_execution(modules):
    class Upstream:
        load_weights = object()
        forward = object()

    class SupportsQuant:
        pass

    mapper_args = []
    mapper = object()

    def make_mapper(*args):
        mapper_args.append(args)
        return mapper

    modules("vllm.model_executor.models.interfaces", SupportsQuant=SupportsQuant)
    modules(
        "vllm.models.deepseek_v41.nvidia.dspark", DSparkDeepseekV4ForCausalLM=Upstream
    )
    modules(
        "vllm.models.deepseek_v41.nvidia.model",
        _make_deepseek_v4_weights_mapper=make_mapper,
    )
    draft = load_patch(
        "vllm_sail/models/deepseek_v41/dspark.py"
    ).DSparkDeepseekV41ForCausalLM
    assert issubclass(draft, SupportsQuant)
    assert (
        draft.load_weights is Upstream.load_weights
        and draft.forward is Upstream.forward
    )
    assert draft.hf_to_vllm_mapper is mapper
    assert mapper_args == [("fp4", "weight_scale")]
    assert draft.packed_modules_mapping["fused_wqa_wkv"] == ["wq_a", "wkv"]
