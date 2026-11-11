# SPDX-License-Identifier: Apache-2.0
"""V4.1 integration contracts; CPU tests never import torch or vLLM."""

import ast
import importlib
import inspect
import sys
from types import SimpleNamespace

import pytest

from tests.support.modules import load_patch
from tests.support.source import ROOT, definition, function

_MEGAMOE_IMPL = ROOT / "vllm_sail/models/deepseek_v41/mega_moe.py"
_MEGAMOE_PATCH = ROOT / "vllm_sail/patch/enhancement/models/deepseek_v41_megamoe.py"
_MEGAMOE_API = ROOT / "vllm_sail/utils/deep_gemm_mega.py"
_MEGAMOE_MODEL = "vllm.models.deepseek_v4.nvidia.model"
_MEGAMOE_EXPERTS = "vllm/model_executor/layers/fused_moe/deep_gemm_mega_moe.py"
_MEGAMOE_CLASS = "PPUDeepGemmMegaMoEExperts"
_MEGAMOE_REQUIRED = (
    "get_symm_buffer_for_mega_moe",
    "mega_moe_pre_dispatch",
    "fp4_fp4_mega_moe",
)


@pytest.fixture
def megamoe_patch(modules):
    class NativeExperts:
        def __init__(self, vllm_config, **kwargs):
            self.vllm_config = vllm_config
            self.kwargs = kwargs
            self.has_fused_shared_experts = kwargs.get("shared_experts") is not None

    class PPUExperts(NativeExperts):
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
    model = modules(_MEGAMOE_MODEL, DeepseekV4MoE=MoE)
    modules(
        "vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe",
        DeepGemmMegaMoEExperts=NativeExperts,
        MEGA_MOE_MXFP4="mxfp4",
        MegaMoEActivation=lambda name, clamp: SimpleNamespace(name=name, clamp=clamp),
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
        native=NativeExperts,
        ppu=PPUExperts,
        provider=provider,
        original_gate=original_gate,
        original_init=original_init,
        original_forward=original_forward,
        patch=patch,
        calls=calls,
        outputs=outputs,
    )


@pytest.mark.parametrize("ppu", [False, True], ids=["cuda", "ppu"])
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
def test_megamoe_selects_ppu_and_preserves_native_fusion(
    megamoe_patch,
    ppu,
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
        swiglu_limit=7,
    )
    moe._init_mega_moe_experts(
        vllm_config=vllm_config, config=config, prefix="model.layers.3.ffn"
    )
    expected = harness.ppu if ppu else harness.native
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
        weight_format="mxfp4",
        activation=SimpleNamespace(name="swiglu", clamp=7.0),
        shared_experts=shared if native_fusion and not ppu else None,
        prefix="model.layers.3.ffn.experts",
    )
    assert moe.experts.kwargs == kwargs
    assert moe.shared_experts is (None if native_fusion and not ppu else shared)
    assert moe.ep_group is harness.group
    assert (moe.n_local_experts, moe.experts_start_idx, moe.experts_end_idx) == (
        6,
        6,
        12,
    )
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
        assert (
            _MEGAMOE_EXPERTS[:-3].replace("/", "."),
            "DeepGemmMegaMoEExperts",
            None,
        ) in imports
    local = definition(_MEGAMOE_IMPL, _MEGAMOE_CLASS)
    assert [ast.unparse(base) for base in local.bases] == ["DeepGemmMegaMoEExperts"]
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
def test_megamoe_targets_and_inherited_interfaces_exist_in_main(upstream_source_root):
    upstream_model = upstream_source_root / "vllm/models/deepseek_v4/nvidia/model.py"
    for name in (
        "_map_global_expert_id",
        "set_eplb_state",
        "get_expert_weights",
        "update_expert_map",
        "has_fused_shared_experts",
        "layer_id",
    ):
        definition(
            upstream_source_root / _MEGAMOE_EXPERTS, f"DeepGemmMegaMoEExperts.{name}"
        )
    definition(upstream_model, "DeepseekV4MoE._init_mega_moe_experts")
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
        "_init_mega_moe_experts",
    ],
)
def test_megamoe_signatures_accept_upstream_keywords(upstream_source_root, method):
    selector = method == "_init_mega_moe_experts"
    upstream_class = "DeepseekV4MoE" if selector else "DeepGemmMegaMoEExperts"
    upstream = function(
        upstream_source_root
        / ("vllm/models/deepseek_v4/nvidia/model.py" if selector else _MEGAMOE_EXPERTS),
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
        upstream_source_root / _MEGAMOE_EXPERTS,
        "DeepGemmMegaMoEExperts._map_global_expert_id",
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

        def __call__(self, value, routed_weights, routed_ids, shared_experts):
            assert value is hidden
            assert (routed_weights, routed_ids) == (weights, ids)
            assert shared_experts is None
            return output

    def route(**kwargs):
        calls.append(kwargs)
        return weights, ids

    def mega_route(*args, **kwargs):
        calls.append("mega")
        return weights, ids

    monkeypatch.setattr(harness.patch, "fused_topk_bias", route)
    monkeypatch.setattr(harness.patch, "bf16_mega_gate", mega_route)
    moe = harness.model.DeepseekV4MoE()
    moe.gate = Gate()
    moe.use_native_mega_moe = True
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
    moe.use_native_mega_moe = False
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
            affected_versions=">=0.32.0,<0.33.0",
            remove_when="test",
        )(megamoe_patch.patch.forward)


@pytest.mark.parametrize("tokens", [0, 3])
@pytest.mark.parametrize("with_shared", [False, True])
def test_expert_forward_uses_sail_dispatch_and_main_shared_contract(
    tokens, with_shared
):
    calls = []

    class Tensor:
        shape = (tokens, 5120)

        def to(self, dtype):
            calls.append(("dtype", dtype))
            return self

        def contiguous(self):
            return self

        def __iadd__(self, other):
            assert other == "shared-output"
            calls.append("add-shared")
            return self

    hidden, ids, weights, output = Tensor(), Tensor(), Tensor(), Tensor()
    buffer = SimpleNamespace(
        x=object(), x_sf=object(), topk_idx=object(), topk_weights=object()
    )
    l1, l2 = object(), object()

    def dispatch(
        x,
        routed_ids,
        routed_weights,
        buf_x,
        buf_x_sf,
        buf_topk_idx,
        buf_topk_weights,
        num_tokens,
        group_size=32,
        mma_type="mxf4xmxf4",
    ):
        assert (x, routed_ids, routed_weights) == (hidden, ids, weights)
        assert buf_x is buffer.x
        assert buf_x_sf is buffer.x_sf
        assert buf_topk_idx is buffer.topk_idx
        assert buf_topk_weights is buffer.topk_weights
        assert num_tokens == tokens
        assert group_size == 32 and mma_type == "mxf4xmxf4"
        calls.append("dispatch")

    def kernel(y, first, second, symm, **kwargs):
        assert (y, first, second, symm) == (output, l1, l2, buffer)
        assert kwargs == dict(
            recipe=(1, 1, 32),
            activation="swiglu",
            activation_clamp=7.0,
            fast_math=False,
        )
        calls.append("kernel")

    def shared(x):
        assert x is hidden
        calls.append("shared")
        return "shared-output"

    ns = {
        "torch": SimpleNamespace(
            int32="i32",
            float32="fp32",
            bfloat16="bf16",
            empty_like=lambda x, dtype: output,
        ),
        "envs": SimpleNamespace(VLLM_MOE_SKIP_PADDING=False),
        "mega_moe_api": lambda: SimpleNamespace(
            mega_moe_pre_dispatch=dispatch, fp4_fp4_mega_moe=kernel
        ),
    }
    forward = function(_MEGAMOE_IMPL, f"{_MEGAMOE_CLASS}.forward", ns)
    expert = SimpleNamespace(
        max_num_tokens=4,
        _synchronize_first_launch=lambda: calls.append("sync"),
        capture_fn=lambda value: calls.append(("capture", value)),
        eplb_state=SimpleNamespace(logical_to_physical_map=None),
        get_symm_buffer=lambda: buffer,
        l1_weights=l1,
        l2_weights=l2,
        activation=SimpleNamespace(clamp=7.0),
        fast_math=False,
    )
    assert (
        forward(expert, hidden, weights, ids, shared if with_shared else None) is output
    )
    assert calls == [
        "sync",
        ("capture", ids),
        ("dtype", "i32"),
        ("dtype", "fp32"),
        "dispatch",
        "kernel",
    ] + (["shared", "add-shared"] if with_shared else [])
    expert.max_num_tokens = -1
    with pytest.raises(ValueError, match="exceeds capacity"):
        forward(expert, hidden, weights, ids)


@pytest.mark.upstream_source
def test_copied_model_bodies_only_change_ppu_selection(upstream_source_root):
    """Keep current upstream routing and initialization outside marked edits."""
    import textwrap

    source = _MEGAMOE_PATCH.read_text()
    for name in ("_init_mega_moe_experts", "forward"):
        local = definition(_MEGAMOE_PATCH, name)
        upstream = definition(
            upstream_source_root / "vllm/models/deepseek_v4/nvidia/model.py",
            f"DeepseekV4MoE.{name}",
        )
        # Substitute the two intentional PPU dispatch edits, then compare ASTs.
        body = "\n".join(source.splitlines()[local.lineno - 1 : local.end_lineno])
        if name == "forward":
            body = body.replace("current_platform.is_ppu() or ", "")
        else:
            start = body.index("    # PPU MODIFICATION: begin")
            end = body.index("    # PPU MODIFICATION: end", start) + len(
                "    # PPU MODIFICATION: end"
            )
            body = (
                body[:start] + "    self.experts = DeepGemmMegaMoEExperts(" + body[end:]
            )
        restored = ast.parse(textwrap.dedent(body)).body[0]
        upstream.decorator_list = []
        assert ast.dump(restored) == ast.dump(upstream)
