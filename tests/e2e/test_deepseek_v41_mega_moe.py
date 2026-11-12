# SPDX-License-Identifier: Apache-2.0
"""Device checks for SAIL MegaMoE weight loading and dispatch."""

import pytest

pytestmark = [pytest.mark.ppu, pytest.mark.cap89]


def test_mega_moe_checkpoint_loader_matches_library_transform(monkeypatch):
    from types import SimpleNamespace

    import torch
    from deep_gemm.mega import transform_weights_for_mega_moe

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
        MEGA_MOE_MXFP4,
        MegaMoEActivation,
    )

    from vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_mega_moe import (
        PPUDeepGemmMegaMoEExperts,
    )

    monkeypatch.setenv("VLLM_SAIL_MEGA_MOE_MMA_TYPE", "mxf4xmxf4")
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    e, h, i = 2, 2048, 128
    with torch.device("cuda"):
        w13 = torch.randint(0, 256, (e, 2 * i, h // 2), dtype=torch.uint8)
        w2 = torch.randint(0, 256, (e, h, i // 2), dtype=torch.uint8)
        s13 = torch.randint(120, 130, (e, 2 * i, h // 32), dtype=torch.uint8)
        s2 = torch.randint(120, 130, (e, h, i // 32), dtype=torch.uint8)
        expected = transform_weights_for_mega_moe((w13, s13), (w2, s2))
        layer = PPUDeepGemmMegaMoEExperts(
            config,
            num_experts=e,
            num_local_experts=e,
            experts_start_idx=0,
            top_k=2,
            hidden_size=h,
            intermediate_size=i,
            activation=MegaMoEActivation("swiglu"),
            weight_format=MEGA_MOE_MXFP4,
        )
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


def test_mega_moe_main_loader_and_eplb_views(monkeypatch):
    from types import SimpleNamespace

    import torch

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
        MEGA_MOE_MXFP4,
        DeepGemmMegaMoEExperts,
        MegaMoEActivation,
    )
    from vllm.models.deepseek_v4.nvidia import model

    from vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_mega_moe import (
        PPUDeepGemmMegaMoEExperts,
    )

    assert model.DeepGemmMegaMoEExperts is DeepGemmMegaMoEExperts
    assert DeepGemmMegaMoEExperts is PPUDeepGemmMegaMoEExperts
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    h, i = 2048, 128
    monkeypatch.setenv("VLLM_SAIL_MEGA_MOE_MMA_TYPE", "mxf4xmxf4")
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

    from vllm_sail.model_executor.layers.fused_moe.experts import (
        deep_gemm_mega_moe as mega_moe,
    )

    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    monkeypatch.setenv("VLLM_SAIL_MEGA_MOE_MMA_TYPE", "mxf4xmxf4")
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
        lambda mode: SimpleNamespace(
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
@pytest.mark.parametrize("mode", ["mxf4xmxf4", "fp8xmxfp4", "bf16xmxfp4"])
def test_mega_moe_single_rank_kernel_smoke(monkeypatch, tmp_path, tokens, mode):
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
        MEGA_MOE_MXFP4,
        MegaMoEActivation,
    )

    from vllm_sail.model_executor.layers.fused_moe.experts import (
        deep_gemm_mega_moe as mega_moe,
    )

    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    monkeypatch.setenv("VLLM_SAIL_MEGA_MOE_MMA_TYPE", mode)
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1
    )
    group = dist.group.WORLD
    monkeypatch.setattr(
        mega_moe, "get_ep_group", lambda: SimpleNamespace(device_group=group)
    )
    try:
        h, i = 2048, 128
        activation = (
            MegaMoEActivation("situ", alpha=4.0, beta=25.0)
            if mode == "bf16xmxfp4"
            else MegaMoEActivation("swiglu", clamp=7.0)
        )
        with torch.device("cuda"):
            layer = mega_moe.PPUDeepGemmMegaMoEExperts(
                config,
                num_experts=2,
                num_local_experts=2,
                experts_start_idx=0,
                top_k=2,
                hidden_size=h,
                intermediate_size=i,
                activation=activation,
                weight_format=MEGA_MOE_MXFP4,
            )
            hidden = torch.full((tokens, h), 0.125, dtype=torch.bfloat16)
            ids = torch.tensor([0, 1], dtype=torch.int64).expand(tokens, 2).contiguous()
            weights = torch.full((tokens, 2), 0.5, dtype=torch.float32)
            w13 = torch.full((i, h // 2), 0x22, dtype=torch.uint8)
            w2 = torch.full((h, i // 2), 0x22, dtype=torch.uint8)
            s13 = torch.full((i, h // 32), 0x78, dtype=torch.uint8)
            s2 = torch.full((h, i // 32), 0x78, dtype=torch.uint8)

        def load(expert, shard, suffix, source):
            bank = "w2" if shard == "w2" else "w13"
            name = f"{bank}_{suffix}"
            assert layer.weight_loader(
                getattr(layer, name), source, name, shard, expert, True
            )

        for expert in range(2):
            for shard, weight, scale in (
                ("w1", w13, s13),
                ("w3", w13, s13),
                ("w2", w2, s2),
            ):
                load(expert, shard, "weight", weight)
                load(expert, shard, "weight_scale", scale)
        monkeypatch.setattr(
            layer, "_synchronize_first_launch", lambda: dist.barrier(group=group)
        )
        output = layer(hidden, weights, ids)
        torch.cuda.synchronize()
        assert layer.mma_type == mode
        assert output.shape == hidden.shape and output.dtype == torch.bfloat16
        assert torch.isfinite(output).all()
        if tokens:
            assert (output > 0).all()
            zero_weight = torch.zeros_like(w2)
            # The experts start identical and each gets half the route weight.
            load(1, "w2", "weight", zero_weight)
            half = layer(hidden, weights, ids)
            torch.cuda.synchronize()
            torch.testing.assert_close(half, output * 0.5, rtol=0.02, atol=0)
            load(0, "w2", "weight", zero_weight)
            zero = layer(hidden, weights, ids)
            torch.cuda.synchronize()
            torch.testing.assert_close(zero, torch.zeros_like(zero), rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        cache = mega_moe.PPUDeepGemmMegaMoEExperts._symm_buffers
        for key, buffer in list(cache.items()):
            if key[0] == id(group):
                buffer.destroy()
                del cache[key]
        dist.destroy_process_group()
