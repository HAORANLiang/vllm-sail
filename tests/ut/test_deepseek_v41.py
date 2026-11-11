# SPDX-License-Identifier: Apache-2.0
"""V4.1 integration contracts; CPU tests never import torch or vLLM."""

import ast
import importlib
import inspect
import sys
from types import SimpleNamespace

import pytest

from tests.support.source import definition, function
from tests.ut import test_ppu_kernel_capabilities as harness

modules = harness.modules
load_patch = harness.load_patch

_MEGAMOE_IMPL = harness.ROOT / "vllm_sail/models/deepseek_v41/mega_moe.py"
_MEGAMOE_PATCH = (
    harness.ROOT / "vllm_sail/patch/enhancement/models/deepseek_v41_megamoe.py"
)
_MEGAMOE_API = harness.ROOT / "vllm_sail/utils/deep_gemm_mega.py"
_MEGAMOE_MODEL = "vllm.models.deepseek_v4.nvidia.model"
_MEGAMOE_CLASS = "PPUDeepGemmMegaMoEExperts"
_MEGAMOE_REQUIRED = (
    "get_symm_buffer_for_mega_moe",
    "mega_moe_pre_dispatch",
    "fp4_fp4_mega_moe",
)


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


@pytest.fixture
def megamoe_patch(modules):
    class NativeExperts:
        def __init__(self, vllm_config, **kwargs):
            self.vllm_config = vllm_config
            self.kwargs = kwargs

    class PPUExperts(NativeExperts):
        pass

    class FlashInferExperts(NativeExperts):
        pass

    class MoE:
        def forward(self, *args, **kwargs):
            raise AssertionError("unpatched forward")

        def _init_mega_moe_experts(self, vllm_config, config, prefix):
            raise AssertionError("unpatched initializer")

    calls = []
    outputs = {name: object() for name in ("native", "ppu")}

    def gate(name):
        def route(*args, **kwargs):
            calls.append((name, args, kwargs))
            return outputs[name]

        return route

    platform = SimpleNamespace(ppu=True, is_cuda=lambda: True)
    platform.is_ppu = lambda: platform.ppu
    modules("vllm.platforms", current_platform=platform)
    envs = modules("vllm.envs", VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION=False)
    modules("vllm.config", VllmConfig=object)
    group = SimpleNamespace(world_size=2, rank_in_group=1)
    modules("vllm.distributed", get_ep_group=lambda: group)
    model = modules(
        _MEGAMOE_MODEL, DeepseekV4MegaMoEExperts=NativeExperts, DeepseekV4MoE=MoE
    )
    fi = modules(
        "vllm.models.deepseek_v4.nvidia.fi_moe",
        DeepseekV4MegaMoEExperts=NativeExperts,
        DeepseekV4MegaMoEExpertsFI=FlashInferExperts,
    )
    modules(
        "vllm_sail.models.deepseek_v41.mega_moe",
        PPUDeepGemmMegaMoEExperts=PPUExperts,
        mega_gate=gate("ppu"),
    )
    original_gate = gate("native")
    provider = modules("vllm.utils.deep_gemm", bf16_mega_gate=original_gate)
    modules(
        "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router",
        fused_topk_bias=lambda **kwargs: (object(), object()),
    )
    original_forward = MoE.forward
    original_init = MoE._init_mega_moe_experts
    patch = load_patch(_MEGAMOE_PATCH)
    return SimpleNamespace(
        platform=platform,
        envs=envs,
        group=group,
        model=model,
        fi=fi,
        native=NativeExperts,
        ppu=PPUExperts,
        flashinfer=FlashInferExperts,
        provider=provider,
        original_gate=original_gate,
        original_init=original_init,
        original_forward=original_forward,
        patch=patch,
        calls=calls,
        outputs=outputs,
    )


@pytest.mark.parametrize("ppu", [False, True], ids=["cuda", "ppu"])
@pytest.mark.parametrize("flashinfer", [False, True])
@pytest.mark.parametrize(
    "has_shared,disable_fusion,tp_size,sequence_parallel,native_fusion",
    [
        (False, False, 1, False, False),
        (True, False, 1, False, True),
        (True, False, 2, True, True),
        (True, False, 2, False, False),
        (True, True, 1, False, False),
    ],
)
def test_megamoe_selects_ppu_only_without_flashinfer_and_preserves_native_fusion(
    megamoe_patch,
    ppu,
    flashinfer,
    has_shared,
    disable_fusion,
    tp_size,
    sequence_parallel,
    native_fusion,
):
    harness = megamoe_patch
    harness.platform.ppu = ppu
    harness.envs.VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION = disable_fusion
    config = SimpleNamespace(
        n_shared_experts=2, hidden_size=512, moe_intermediate_size=128
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=8)
        )
    )
    moe = harness.model.DeepseekV4MoE()
    shared = object() if has_shared else None
    moe.__dict__.update(
        n_routed_experts=4,
        n_activated_experts=2,
        shared_experts=shared,
        use_sequence_parallel=sequence_parallel,
        tp_size=tp_size,
        use_fi_mega_moe=flashinfer,
        swiglu_limit=7,
    )
    moe._init_mega_moe_experts(
        vllm_config=vllm_config, config=config, prefix="model.layers.3.ffn"
    )
    expected = (
        harness.flashinfer if flashinfer else harness.ppu if ppu else harness.native
    )
    assert type(moe.experts) is expected
    assert moe.experts.vllm_config is vllm_config
    kwargs = dict(
        num_experts=12,
        num_local_experts=6,
        experts_start_idx=6,
        num_logical_experts=4,
        top_k=2,
        hidden_size=512,
        intermediate_size=128,
        num_shared_experts=2 if native_fusion and not ppu and not flashinfer else 0,
        prefix="model.layers.3.ffn.experts",
    )
    if flashinfer:
        kwargs["activation_clamp"] = 7.0
    assert moe.experts.kwargs == kwargs
    assert moe.shared_experts is shared
    assert moe.ep_group is harness.group
    assert (moe.n_local_experts, moe.experts_start_idx, moe.experts_end_idx) == (
        6,
        6,
        12,
    )
    assert harness.model.DeepseekV4MegaMoEExperts is harness.native
    assert harness.fi.DeepseekV4MegaMoEExperts is harness.native
    assert harness.fi.DeepseekV4MegaMoEExpertsFI is harness.flashinfer
    assert harness.flashinfer.__bases__ == (harness.native,)
    marker = getattr(moe._init_mega_moe_experts, harness.patch.PATCH_MARKER)
    target = f"{_MEGAMOE_MODEL}.DeepseekV4MoE._init_mega_moe_experts"
    assert marker[target] is harness.original_init


def test_megamoe_gate_routes_only_ppu_and_delegates_cuda_unchanged(megamoe_patch):
    harness = megamoe_patch
    args = (object(), object(), 2)
    kwargs = dict(
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        ep_rank=1,
        bias=object(),
        image_bias=object(),
        image_token_mask=object(),
        fix_routing_mask=object(),
        unmapped_topk_idx=object(),
    )
    assert harness.provider.bf16_mega_gate is harness.patch.route_experts
    marker = getattr(harness.provider.bf16_mega_gate, harness.patch.PATCH_MARKER)
    assert marker["vllm.utils.deep_gemm.bf16_mega_gate"] is harness.original_gate
    for ppu in (True, False, True):
        harness.platform.ppu = ppu
        name = "ppu" if ppu else "native"
        harness.calls.clear()
        assert harness.provider.bf16_mega_gate(*args, **kwargs) is harness.outputs[name]
        assert harness.calls == [(name, args, kwargs)]


def test_megamoe_inherits_native_expert_mapping_and_eplb_interfaces():
    for path in (_MEGAMOE_IMPL, _MEGAMOE_PATCH):
        imports = {
            (node.module, alias.name, alias.asname)
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert (_MEGAMOE_MODEL, "DeepseekV4MegaMoEExperts", None) in imports
    local = definition(_MEGAMOE_IMPL, _MEGAMOE_CLASS)
    assert [ast.unparse(base) for base in local.bases] == ["DeepseekV4MegaMoEExperts"]
    local_methods = {
        node.name for node in local.body if isinstance(node, ast.FunctionDef)
    }
    assert not local_methods.intersection(
        {
            "_map_global_expert_id",
            "set_eplb_state",
            "get_expert_weights",
            "update_expert_map",
            "has_fused_shared_experts",
            "layer_id",
        }
    )


@pytest.mark.upstream_source
def test_megamoe_targets_and_inherited_interfaces_exist_in_v031(upstream_source_root):
    upstream_model = upstream_source_root / "vllm/models/deepseek_v4/nvidia/model.py"
    for name in (
        "_map_global_expert_id",
        "set_eplb_state",
        "get_expert_weights",
        "update_expert_map",
        "has_fused_shared_experts",
        "layer_id",
    ):
        definition(upstream_model, f"DeepseekV4MegaMoEExperts.{name}")
    definition(upstream_model, "DeepseekV4MoE._init_mega_moe_experts")
    definition(
        upstream_source_root / "vllm/models/deepseek_v4/nvidia/fi_moe.py",
        "DeepseekV4MegaMoEExpertsFI",
    )
    gate_path = upstream_source_root / "vllm/utils/deep_gemm.py"
    if not gate_path.is_file():
        gate_path = upstream_source_root / "vllm/utils/deep_gemm/__init__.py"
    definition(gate_path, "bf16_mega_gate")


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "method",
    [
        "__init__",
        "forward",
        "weight_loader",
        "finalize_weights",
        "_init_mega_moe_experts",
    ],
)
def test_megamoe_signatures_accept_upstream_keywords(upstream_source_root, method):
    selector = method == "_init_mega_moe_experts"
    upstream_class = "DeepseekV4MoE" if selector else "DeepseekV4MegaMoEExperts"
    upstream = function(
        upstream_source_root / "vllm/models/deepseek_v4/nvidia/model.py",
        f"{upstream_class}.{method}",
        {},
    )
    local = function(
        _MEGAMOE_PATCH if selector else _MEGAMOE_IMPL,
        method if selector else f"{_MEGAMOE_CLASS}.{method}",
        {},
    )
    offered = inspect.signature(upstream)
    accepted = inspect.signature(local)
    keyword_kinds = (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
    for include_defaults in (False, True):
        kwargs = {
            name: object()
            for name, param in offered.parameters.items()
            if param.kind in keyword_kinds
            and (include_defaults or param.default is inspect.Parameter.empty)
        }
        accepted.bind(**kwargs)
    for kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
        if any(param.kind == kind for param in offered.parameters.values()):
            assert any(param.kind == kind for param in accepted.parameters.values())


@pytest.fixture
def megamoe_loader():
    loader = function(_MEGAMOE_IMPL, f"{_MEGAMOE_CLASS}.weight_loader", {})

    def prepare(local_ids):
        mapped, indexed, loaded = [], [], []
        slots = {slot: object() for slot in local_ids}

        class Data:
            def __getitem__(self, slot):
                indexed.append(slot)
                return slots[slot]

        def map_expert(expert_id):
            mapped.append(expert_id)
            return local_ids

        expert = SimpleNamespace(
            _map_global_expert_id=map_expert,
            _load_weight=lambda *args: loaded.append(args),
        )
        return SimpleNamespace(
            loader=loader,
            expert=expert,
            param=SimpleNamespace(data=Data()),
            mapped=mapped,
            indexed=indexed,
            loaded=loaded,
            slots=slots,
        )

    return prepare


@pytest.mark.parametrize("shard_id", ["w1", "w3", "w2"])
@pytest.mark.parametrize("is_scale", [False, True])
@pytest.mark.parametrize("local_ids", [(0, 4), ()], ids=["replicas", "nonlocal"])
@pytest.mark.parametrize("return_success", [False, None, True])
def test_megamoe_weight_loader_routes_replicas_and_skips_nonlocal(
    megamoe_loader, shard_id, is_scale, local_ids, return_success
):
    harness = megamoe_loader(local_ids)
    bank = "w2" if shard_id == "w2" else "w13"
    weight_name = f"model.layers.3.ffn.experts.{bank}_weight"
    if is_scale:
        weight_name += "_scale"
    source = object()
    result = harness.loader(
        harness.expert,
        param=harness.param,
        loaded_weight=source,
        weight_name=weight_name,
        shard_id=shard_id,
        expert_id=2,
        return_success=return_success,
    )
    assert result is (bool(local_ids) if return_success else None)
    assert harness.mapped == [2]
    assert harness.indexed == list(local_ids)
    assert harness.loaded == [
        (harness.slots[slot], source, shard_id, is_scale) for slot in local_ids
    ]


@pytest.mark.parametrize(
    "shard_id,wrong_bank", [("w1", "w2"), ("w3", "w2"), ("w2", "w13")]
)
@pytest.mark.parametrize("return_success", [False, True])
def test_megamoe_weight_loader_skips_mismatched_parameter_bank(
    megamoe_loader, shard_id, wrong_bank, return_success
):
    harness = megamoe_loader((0, 4))
    result = harness.loader(
        harness.expert,
        harness.param,
        object(),
        f"{wrong_bank}_weight",
        shard_id,
        2,
        return_success=return_success,
    )
    assert result is (False if return_success else None)
    assert harness.indexed == harness.loaded == []


def test_megamoe_weight_loader_defaults_to_none_and_rejects_unknown_shards(
    megamoe_loader,
):
    harness = megamoe_loader((0,))
    result = harness.loader(
        harness.expert, harness.param, object(), "w2_weight", "w2", 2
    )
    assert result is None
    assert harness.indexed == [0]
    harness.indexed.clear()
    harness.loaded.clear()
    with pytest.raises(ValueError, match="Unsupported expert shard id"):
        harness.loader(harness.expert, harness.param, object(), "w13_weight", "w9", 2)
    assert harness.indexed == harness.loaded == []


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "start,end,logical_id,local_ids",
    [(6, 12, 2, (0, 4)), (6, 12, 0, (2,)), (8, 10, 2, ())],
    ids=["redundant-replicas", "offset-slot", "nonlocal"],
)
def test_megamoe_weight_loader_uses_upstream_logical_to_physical_map(
    megamoe_loader, upstream_source_root, start, end, logical_id, local_ids
):
    harness = megamoe_loader(local_ids)
    mapping = function(
        upstream_source_root / "vllm/models/deepseek_v4/nvidia/model.py",
        "DeepseekV4MegaMoEExperts._map_global_expert_id",
        {},
    )
    harness.expert.__dict__.update(
        experts_start_idx=start, experts_end_idx=end, num_logical_experts=4
    )
    harness.expert._map_global_expert_id = lambda expert_id: mapping(
        harness.expert, expert_id
    )
    source = object()
    result = harness.loader(
        harness.expert,
        harness.param,
        source,
        "w13_weight",
        "w1",
        logical_id,
        return_success=True,
    )
    assert result is bool(local_ids)
    assert harness.indexed == list(local_ids)
    assert harness.loaded == [
        (harness.slots[slot], source, "w1", False) for slot in local_ids
    ]


@pytest.mark.parametrize("shared_experts", [None, object()])
def test_megamoe_finalize_reuses_loader_storage_and_checks_runtime_once(shared_experts):
    calls = []
    l1, l2 = (object(), object()), (object(), object())
    expert = SimpleNamespace(
        l1_weights=l1,
        l2_weights=l2,
        num_shared_experts=0,
        _transformed_l1_weights=None,
        _transformed_l2_weights=None,
        _transformed_shared_l1_weights=None,
        _transformed_shared_l2_weights=None,
        _check_runtime_supported=lambda: calls.append("check"),
    )
    finalize = function(_MEGAMOE_IMPL, f"{_MEGAMOE_CLASS}.finalize_weights", {})
    for _ in range(2):
        assert finalize(expert, shared_experts=shared_experts) is None
        assert expert._transformed_l1_weights is l1
        assert expert._transformed_l2_weights is l2
        assert expert._transformed_shared_l1_weights is None
        assert expert._transformed_shared_l2_weights is None
    assert calls == ["check"]


@pytest.fixture
def megamoe_api_module(monkeypatch):
    calls = []
    original_import = importlib.import_module

    def import_module(name, *args, **kwargs):
        if name == "deep_gemm.mega":
            calls.append(name)
            raise ImportError("DeepGEMM is not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", import_module)
    for name in ("deep_gemm", "torch", "vllm"):
        monkeypatch.setitem(sys.modules, name, None)
    module = load_patch(_MEGAMOE_API)
    assert calls == []
    return module, calls


def test_megamoe_api_import_is_lazy_and_reports_missing_package(megamoe_api_module):
    module, calls = megamoe_api_module
    with pytest.raises(
        RuntimeError, match=r"DeepGEMM-for-sail.*deep_gemm\.mega"
    ) as error:
        module.mega_moe_api()
    assert isinstance(error.value.__cause__, ImportError)
    assert calls == ["deep_gemm.mega"]


@pytest.mark.parametrize("missing", _MEGAMOE_REQUIRED)
@pytest.mark.parametrize("noncallable", [False, True], ids=["absent", "noncallable"])
def test_megamoe_api_reports_missing_required_functions_and_allows_retry(
    megamoe_api_module, monkeypatch, missing, noncallable
):
    module, calls = megamoe_api_module
    api = SimpleNamespace(**{name: lambda: None for name in _MEGAMOE_REQUIRED})
    if noncallable:
        setattr(api, missing, object())
    else:
        delattr(api, missing)

    def import_module(name):
        calls.append(name)
        return api

    monkeypatch.setattr(
        module, "importlib", SimpleNamespace(import_module=import_module)
    )
    with pytest.raises(RuntimeError) as error:
        module.mega_moe_api()
    assert str(error.value) == f"PPU MegaMoE API is missing: {missing}"
    setattr(api, missing, lambda: None)
    assert module.mega_moe_api() is api
    assert module.mega_moe_api() is api
    assert calls == ["deep_gemm.mega", "deep_gemm.mega"]


def test_megamoe_api_reports_all_missing_functions(megamoe_api_module, monkeypatch):
    module, _ = megamoe_api_module
    monkeypatch.setattr(
        module, "importlib", SimpleNamespace(import_module=lambda _: SimpleNamespace())
    )
    with pytest.raises(RuntimeError) as error:
        module.mega_moe_api()
    missing = ", ".join(_MEGAMOE_REQUIRED)
    assert str(error.value) == f"PPU MegaMoE API is missing: {missing}"


@pytest.mark.parametrize(
    "hidden,intermediate,supported",
    [
        (2048, 64, True),
        (2560, 128, True),
        (512, 128, False),
        (2048, 0, False),
        (2048, 96, False),
        (2304, 128, False),
    ],
)
def test_megamoe_geometry_matches_wheel_contract(
    modules, monkeypatch, hidden, intermediate, supported
):
    modules(
        "vllm.platforms",
        current_platform=SimpleNamespace(
            is_ppu=lambda: True,
            has_device_capability=lambda capability, device_id: capability == 89,
        ),
    )
    modules("vllm_sail.utils.deep_gemm", is_deep_gemm_supported=lambda: True)
    module = load_patch(_MEGAMOE_API)
    calls = []
    monkeypatch.setattr(module, "mega_moe_api", lambda: calls.append("api"))
    device = SimpleNamespace(index=0)
    if supported:
        module.check_mega_moe_supported(device, hidden, intermediate)
        assert calls == ["api"]
    else:
        with pytest.raises(ValueError, match="H >= 2048.*I divisible by 64"):
            module.check_mega_moe_supported(device, hidden, intermediate)
        assert not calls


@pytest.mark.parametrize("tokens", [0, 1, 16, 17, 104, 256])
@pytest.mark.parametrize("ppu", [False, True])
def test_megamoe_forward_uses_fused_routing_for_all_ppu_batches(
    megamoe_patch, monkeypatch, tokens, ppu
):
    harness = megamoe_patch
    harness.platform.ppu = ppu
    calls = []
    weights, ids = object(), object()
    hidden = SimpleNamespace(shape=(tokens, 5120))
    output = SimpleNamespace(view=lambda shape: ("output", shape))

    class Gate:
        weight = SimpleNamespace(shape=(384, 5120))
        tid2eid = object()
        e_score_correction_bias = SimpleNamespace(data=object())
        bias_vl = SimpleNamespace(data=object())

        def __call__(self, value):
            assert value is hidden
            calls.append("linear")
            return "logits", None

    class Experts:
        has_fused_shared_experts = False

        def __call__(self, value, routed_weights, routed_ids, **kwargs):
            assert value is hidden
            assert (routed_weights, routed_ids) == (weights, ids)
            assert kwargs == {"activation_clamp": 7.0}
            return output

    def route(**kwargs):
        calls.append(kwargs)
        return weights, ids

    def mega_route(*args, **kwargs):
        calls.append("mega")
        return weights, ids

    monkeypatch.setattr(harness.patch, "fused_topk_bias", route)
    monkeypatch.setattr(harness.provider, "bf16_mega_gate", mega_route)
    moe = harness.model.DeepseekV4MoE()
    moe.gate = Gate()
    moe.use_mega_moe = True
    moe.scoring_func = "sqrtsoftplus"
    moe.n_activated_experts = 6
    moe.renormalize = True
    moe.hash_indices_dtype = "int64"
    moe.routed_scaling_factor = 2.5
    moe.image_sentinel_lo = 100
    moe.ep_rank = 0
    moe.swiglu_limit = 7
    moe.experts = Experts()
    moe.shared_experts = None
    inputs = object()
    if ppu or tokens <= 16:
        assert moe.forward(hidden, inputs) == ("output", hidden.shape)
        assert calls[0] == "linear"
        assert calls[1] == dict(
            hidden_states=hidden,
            gating_output="logits",
            scoring_func="sqrtsoftplus",
            e_score_correction_bias=moe.gate.e_score_correction_bias.data,
            topk=6,
            renormalize=True,
            indices_type="int64",
            input_tokens=inputs,
            hash_indices_table=moe.gate.tid2eid,
            routed_scaling_factor=2.5,
            bias_vl=moe.gate.bias_vl.data,
            image_sentinel_lo=100,
        )
    else:
        moe.gate.tid2eid = None
        metadata = SimpleNamespace(image_token_mask=object())
        assert moe.forward(hidden, inputs, metadata) == ("output", hidden.shape)
        assert calls == ["mega"]
    marker = getattr(moe.forward, harness.patch.PATCH_MARKER)
    assert marker[f"{_MEGAMOE_MODEL}.DeepseekV4MoE.forward"] is harness.original_forward


def test_megamoe_forward_retains_non_mega_path(megamoe_patch):
    moe = megamoe_patch.model.DeepseekV4MoE()
    moe.gate = SimpleNamespace(tid2eid=None)
    moe.use_mega_moe = False
    moe._forward_fused_moe = lambda hidden, ids: (hidden, ids)
    hidden, ids = object(), object()
    assert moe.forward(hidden, ids) == (hidden, ids)


def test_megamoe_forward_rejects_duplicate_patch(megamoe_patch):
    from vllm_sail.patch.utils import patch

    with pytest.raises(RuntimeError, match="already patched"):
        patch(
            _MEGAMOE_MODEL,
            "DeepseekV4MoE.forward",
            reason="test",
            affected_versions=">=0.31.0,<0.32.0",
            remove_when="test",
        )(megamoe_patch.patch.forward)
