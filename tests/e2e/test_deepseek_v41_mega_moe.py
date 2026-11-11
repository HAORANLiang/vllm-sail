# SPDX-License-Identifier: Apache-2.0
"""Device checks for the SAIL MegaMoE loader and routing contracts."""

import pytest

pytestmark = [pytest.mark.ppu, pytest.mark.cap89]


def test_mega_moe_checkpoint_loader_matches_library_transform():
    import torch
    from deep_gemm.mega import transform_weights_for_mega_moe

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm_sail.models.deepseek_v41.mega_moe import PPUDeepGemmMegaMoEExperts

    e, h, i = 2, 2048, 128
    with torch.device("cuda"):
        w13 = torch.randint(0, 256, (e, 2 * i, h // 2), dtype=torch.uint8)
        w2 = torch.randint(0, 256, (e, h, i // 2), dtype=torch.uint8)
        s13 = torch.randint(120, 130, (e, 2 * i, h // 32), dtype=torch.uint8)
        s2 = torch.randint(120, 130, (e, h, i // 32), dtype=torch.uint8)
        expected = transform_weights_for_mega_moe((w13, s13), (w2, s2))
        layer = PPUDeepGemmMegaMoEExperts.__new__(PPUDeepGemmMegaMoEExperts)
        torch.nn.Module.__init__(layer)
        layer.hidden_size, layer.intermediate_size = h, i
        layer._register_weight("w13", (e, 2 * i, h), None)
        layer._register_weight("w2", (e, h, i), None)
        for expert in range(e):
            for shard, start in (("w1", 0), ("w3", i)):
                layer._load_weight(
                    layer.w13_weight[expert],
                    w13[expert, start : start + i],
                    shard,
                    False,
                )
                layer._load_weight(
                    layer.w13_weight_scale[expert],
                    s13[expert, start : start + i],
                    shard,
                    True,
                )
            layer._load_weight(layer.w2_weight[expert], w2[expert], "w2", False)
            layer._load_weight(layer.w2_weight_scale[expert], s2[expert], "w2", True)
        for actual_pair, expected_pair in zip(
            (layer.l1_weights, layer.l2_weights), expected, strict=True
        ):
            for actual, reference in zip(actual_pair, expected_pair, strict=True):
                assert actual.shape == reference.shape
                assert actual.stride() == reference.stride()
                assert torch.equal(
                    actual.contiguous().view(torch.uint8),
                    reference.contiguous().view(torch.uint8),
                )


def test_mega_gate_preserves_hash_image_and_weight_contracts():
    import torch

    from vllm_sail.models.deepseek_v41.mega_moe import mega_gate

    x = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(8, 64, device="cuda", dtype=torch.float32)
    bias = torch.randn(8, device="cuda")
    image_bias = torch.randn(8, device="cuda")
    image = torch.tensor([False, True, False], device="cuda")
    fixed = torch.tensor([True, False, False], device="cuda")
    hash_ids = torch.tensor([[5, 2], [4, 1], [0, 3]], device="cuda")
    weights, ids = mega_gate(
        x,
        w,
        2,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=2.5,
        ep_rank=0,
        bias=bias,
        image_bias=image_bias,
        image_token_mask=image,
        fix_routing_mask=fixed,
        unmapped_topk_idx=hash_ids,
    )
    torch.testing.assert_close(ids[0], hash_ids[0])
    scores = torch.nn.functional.softplus(x.float() @ w.T).sqrt()
    torch.testing.assert_close(
        ids[1], torch.argsort(scores[1] + image_bias, descending=True, stable=True)[:2]
    )
    selected = scores.gather(1, ids)
    torch.testing.assert_close(weights, selected / selected.sum(-1, keepdim=True) * 2.5)


def test_mega_moe_main_loader_and_eplb_views(monkeypatch):
    from types import SimpleNamespace

    import torch

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
        MEGA_MOE_MXFP4,
        MegaMoEActivation,
    )

    from vllm_sail.models.deepseek_v41.mega_moe import PPUDeepGemmMegaMoEExperts

    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    h, i = 2048, 128
    monkeypatch.setattr(
        PPUDeepGemmMegaMoEExperts, "_check_runtime_supported", lambda self: None
    )
    with torch.device("cpu"):
        layer = PPUDeepGemmMegaMoEExperts(
            config,
            num_experts=6,
            num_local_experts=3,
            experts_start_idx=3,
            num_logical_experts=2,
            top_k=2,
            hidden_size=h,
            intermediate_size=i,
            activation=MegaMoEActivation("swiglu", clamp=7.0),
            weight_format=MEGA_MOE_MXFP4,
            prefix="model.layers.0.ffn.experts",
        )
        w1 = torch.randint(0, 256, (i, h // 2), dtype=torch.uint8)
        w3 = torch.randint(0, 256, (i, h // 2), dtype=torch.uint8)
        w2 = torch.randint(0, 256, (h, i // 2), dtype=torch.uint8)
        s1 = torch.randint(120, 130, (i, h // 32), dtype=torch.uint8)
        s3 = torch.randint(120, 130, (i, h // 32), dtype=torch.uint8)
        s2 = torch.randint(120, 130, (h, i // 32), dtype=torch.uint8)
    expected_w13 = torch.stack((w1, w3), dim=1).reshape(2 * i, h // 2)
    expected_s13 = torch.stack((s1, s3), dim=1).reshape(2 * i, h // 32)
    for shard, weight, scale in (("w1", w1, s1), ("w3", w3, s3), ("w2", w2, s2)):
        name = "w2" if shard == "w2" else "w13"
        for suffix, source in (("weight", weight), ("weight_scale", scale)):
            param_name = f"{name}_{suffix}"
            assert layer.weight_loader(
                getattr(layer, param_name), source, param_name, shard, 1, True
            )
    for slot in (0, 2):
        torch.testing.assert_close(layer.w13_weight[slot], expected_w13)
        actual_s13 = layer.w13_weight_scale[slot].contiguous().view(torch.uint8)
        torch.testing.assert_close(actual_s13, expected_s13)
        torch.testing.assert_close(layer.w2_weight[slot], w2)
        torch.testing.assert_close(
            layer.w2_weight_scale[slot].contiguous().view(torch.uint8), s2
        )
    assert not layer.w13_weight[1].any()
    monkeypatch.setattr(layer, "_check_runtime_supported", lambda: None)
    pointers = [param.data_ptr() for param in layer.parameters()]
    views = layer.get_expert_weights()
    assert [param.data_ptr() for param in layer.parameters()] == pointers
    assert {view.data_ptr() for view in views} == set(pointers)
    assert all(view.shape[0] == 3 and view.is_contiguous() for view in views)
    assert not layer.has_fused_shared_experts
    assert layer.num_shared_experts == 0
    assert layer.weight_loader.supports_moe_loading
    assert all(
        param.weight_loader == layer.weight_loader for param in layer.parameters()
    )
    views[1].zero_()
    assert not layer.w13_weight_scale.contiguous().view(torch.uint8).any()


@pytest.mark.parametrize("tokens", [0, 3])
def test_mega_moe_main_forward_contract(monkeypatch, tokens):
    from types import SimpleNamespace

    import torch

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
        MEGA_MOE_MXFP4,
        MegaMoEActivation,
    )

    from vllm_sail.models.deepseek_v41 import mega_moe

    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    monkeypatch.setattr(
        mega_moe.PPUDeepGemmMegaMoEExperts,
        "_check_runtime_supported",
        lambda self: None,
    )
    with torch.device("cpu"):
        layer = mega_moe.PPUDeepGemmMegaMoEExperts(
            config,
            num_experts=2,
            num_local_experts=2,
            experts_start_idx=0,
            top_k=2,
            hidden_size=512,
            intermediate_size=128,
            activation=MegaMoEActivation("swiglu", clamp=7.0),
            weight_format=MEGA_MOE_MXFP4,
            fast_math=False,
        )
        hidden = torch.randn(tokens, 512, dtype=torch.bfloat16)
        ids = torch.zeros(tokens, 2, dtype=torch.int64)
        weights = torch.ones(tokens, 2, dtype=torch.float32)
        padding = torch.tensor([True, False, False])[:tokens]
    calls = []
    buffer = SimpleNamespace(
        x=object(), x_sf=object(), topk_idx=object(), topk_weights=object()
    )

    def dispatch(
        x,
        selected_ids,
        selected_weights,
        buf_x,
        buf_x_sf,
        buf_topk_idx,
        buf_topk_weights,
        num_tokens,
        group_size=32,
        mma_type="mxf4xmxf4",
    ):
        assert x is hidden
        assert buf_x is buffer.x
        assert buf_x_sf is buffer.x_sf
        assert buf_topk_idx is buffer.topk_idx
        assert buf_topk_weights is buffer.topk_weights
        assert num_tokens == tokens
        assert group_size == 32 and mma_type == "mxf4xmxf4"
        assert selected_ids.dtype == torch.int32 and selected_ids.is_contiguous()
        assert selected_weights.dtype == torch.float32
        if tokens:
            assert (selected_ids[0] == -1).all()
        calls.append("dispatch")

    def kernel(output, l1, l2, actual_buffer, **kwargs):
        assert actual_buffer is buffer
        assert l1[0].data_ptr() == layer.w13_weight.data_ptr()
        assert l2[0].data_ptr() == layer.w2_weight.data_ptr()
        assert kwargs == {
            "recipe": (1, 1, 32),
            "activation": "swiglu",
            "activation_clamp": 7.0,
            "fast_math": False,
        }
        output.fill_(2)
        calls.append("kernel")

    monkeypatch.setattr(layer, "get_symm_buffer", lambda: buffer)
    monkeypatch.setattr(mega_moe, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        mega_moe, "get_forward_context", lambda: SimpleNamespace(is_padding=padding)
    )
    monkeypatch.setattr(
        mega_moe,
        "mega_moe_api",
        lambda: SimpleNamespace(
            mega_moe_pre_dispatch=dispatch, fp4_fp4_mega_moe=kernel
        ),
    )
    monkeypatch.setattr(layer, "_synchronize_first_launch", lambda: None)
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    output = layer(hidden, weights, ids)
    assert calls == ["dispatch", "kernel"]
    assert output.shape == hidden.shape and output.dtype == torch.bfloat16
    assert (output == 2).all()


@pytest.mark.parametrize("tokens", [0, 16])
def test_mega_moe_single_rank_kernel_smoke(monkeypatch, tmp_path, tokens):
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
        MEGA_MOE_MXFP4,
        MegaMoEActivation,
    )

    from vllm_sail.models.deepseek_v41 import mega_moe

    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1
    )
    group = dist.group.WORLD
    monkeypatch.setattr(
        mega_moe, "get_ep_group", lambda: SimpleNamespace(device_group=group)
    )
    try:
        with torch.device("cuda"):
            layer = mega_moe.PPUDeepGemmMegaMoEExperts(
                config,
                num_experts=2,
                num_local_experts=2,
                experts_start_idx=0,
                top_k=2,
                hidden_size=2048,
                intermediate_size=128,
                activation=MegaMoEActivation("swiglu", clamp=None),
                weight_format=MEGA_MOE_MXFP4,
            )
            hidden = torch.randn(tokens, 2048, dtype=torch.bfloat16) * 0.01
            ids = torch.tensor([0, 1], dtype=torch.int64).expand(tokens, 2).contiguous()
            weights = torch.full((tokens, 2), 0.5, dtype=torch.float32)
        layer.w13_weight_scale.data.fill_(0x7878)
        layer.w2_weight_scale.data.fill_(0x7878)
        monkeypatch.setattr(
            layer, "_synchronize_first_launch", lambda: dist.barrier(group=group)
        )
        output = layer(hidden, weights, ids)
        torch.cuda.synchronize()
        assert output.shape == hidden.shape and output.dtype == torch.bfloat16
        torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
        if tokens:
            layer.w13_weight.data.fill_(0x22)
            layer.w2_weight.data.fill_(0x22)
            layer.activation = MegaMoEActivation("swiglu", clamp=7.0)
            output = layer(hidden, weights, ids)
            torch.cuda.synchronize()
            assert torch.isfinite(output).all()
            assert output.abs().max() > 0
    finally:
        torch.cuda.synchronize()
        cache = mega_moe.PPUDeepGemmMegaMoEExperts._symm_buffers
        for key, buffer in list(cache.items()):
            if key[0] == id(group):
                buffer.destroy()
                del cache[key]
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def mega_moe_gate_factory(tmp_path_factory):
    import torch

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    rendezvous = tmp_path_factory.mktemp("mega_gate") / "rendezvous"
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{rendezvous}",
            backend="nccl",
        )
        initialize_model_parallel(1, 1)
        try:

            def make_gate(num_experts):
                return GateLinear(
                    5120,
                    num_experts,
                    params_dtype=torch.float32,
                    out_dtype=torch.float32,
                ).cuda()

            yield make_gate
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


@pytest.mark.parametrize("tokens", [1, 17, 104, 256])
@pytest.mark.parametrize("num_experts", [128, 384])
@pytest.mark.parametrize("routing", ["plain", "bias", "hash", "image", "mixed", "ties"])
def test_mega_moe_fused_routing_matches_eager(
    mega_moe_gate_factory, tokens, num_experts, routing
):
    from types import SimpleNamespace

    import torch
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MoE

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm_sail.models.deepseek_v41.mega_moe import mega_gate
    from vllm_sail.patch.enhancement.models.deepseek_v41_megamoe import forward

    assert DeepseekV4MoE.forward is forward
    torch.manual_seed(123)
    x = torch.randn(tokens, 5120, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(num_experts, 5120, device="cuda") * 0.01
    bias = None if routing == "plain" else torch.randn(num_experts, device="cuda")
    if routing == "ties":
        weight.zero_()
        bias.zero_()
    image_bias = (
        torch.randn(num_experts, device="cuda")
        if routing in ("image", "mixed")
        else None
    )
    input_ids = torch.arange(tokens, device="cuda", dtype=torch.int64) % 16
    image_mask = (input_ids >= 10) & (input_ids < 15)
    table = (
        torch.randint(num_experts, (16, 6), device="cuda", dtype=torch.int64)
        if routing in ("hash", "mixed")
        else None
    )
    fixed = ~image_mask if image_bias is not None else torch.ones_like(image_mask)
    expected_weights, expected_ids = mega_gate(
        x,
        weight,
        6,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=2.5,
        ep_rank=0,
        bias=bias,
        image_bias=image_bias,
        image_token_mask=image_mask,
        fix_routing_mask=fixed if table is not None else None,
        unmapped_topk_idx=table[input_ids] if table is not None else None,
    )
    captured = {}

    gate = mega_moe_gate_factory(num_experts)
    gate.weight.data.copy_(weight)
    gate.tid2eid = table
    gate.e_score_correction_bias = bias
    gate.bias_vl = image_bias

    class Experts:
        has_fused_shared_experts = False

        def __call__(self, hidden, weights, ids, shared_experts):
            assert shared_experts is None
            captured.update(weights=weights, ids=ids)
            return hidden

    moe = SimpleNamespace(
        gate=gate,
        use_native_mega_moe=True,
        scoring_func="sqrtsoftplus",
        n_activated_experts=6,
        renormalize=True,
        hash_indices_dtype=torch.int64,
        routed_scaling_factor=2.5,
        image_sentinel_lo=10,
        swiglu_limit=None,
        experts=Experts(),
        shared_experts=None,
    )
    assert forward(moe, x, input_ids).data_ptr() == x.data_ptr()
    torch.testing.assert_close(captured["ids"], expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(
        captured["weights"], expected_weights, atol=1e-6, rtol=1e-5
    )
    assert captured["weights"].is_contiguous()
    assert captured["ids"].is_contiguous()
