# SPDX-License-Identifier: Apache-2.0
"""Exercise target/draft expert counts against one real DeepEP buffer."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.ppu
PPU_CI_MIN_DEVICES = 8


def _worker(rank, rendezvous):
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    import deep_ep

    from vllm_sail.model_executor.layers.fused_moe.prepare_finalize.deepep_buffer import (
        dspark_buffer_args,
    )

    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=120),
    )
    buffer = None
    try:
        config = SimpleNamespace(
            speculative_config=SimpleNamespace(use_dspark=lambda: True),
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(
                    n_routed_experts=384,
                    dspark_n_routed_experts=128,
                )
            ),
        )
        keys = [
            dspark_buffer_args(
                dict(
                    max_num_tokens_per_dp_rank=64,
                    token_hidden_size=5120,
                    num_ep_ranks=8,
                    num_global_experts=e,
                    num_local_experts=e // 8,
                ),
                config,
            )
            for e in (384, 128)
        ]
        assert keys[0] == keys[1]
        args = keys[0]
        size = deep_ep.Buffer.get_low_latency_rdma_size_hint(64, 5120, 8, 384)
        buffer = deep_ep.Buffer(
            dist.group.WORLD,
            num_nvl_bytes=0,
            num_rdma_bytes=size,
            low_latency_mode=True,
            num_qps_per_rank=args["num_local_experts"],
            allow_nvlink_for_low_latency_mode=True,
            explicitly_destroy=True,
        )
        from vllm.model_executor.layers.fused_moe.config import ocp_mx_moe_quant_config
        from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
            TopKWeightAndReduceDelegate,
        )

        from vllm_sail.model_executor.layers.fused_moe.prepare_finalize.deepep_ll import (
            DeepEPLLPrepareAndFinalize,
        )

        prepare = DeepEPLLPrepareAndFinalize(
            buffer, 64, 8, use_mxfp4_dispatch=True, clean_buffer_before_dispatch=True
        )
        quant = ocp_mx_moe_quant_config(
            "mxfp4",
            torch.ones(1, device="cuda"),
            torch.ones(1, device="cuda"),
            block_shape=[1, 32],
        )
        tokens = 16 + rank % 3
        x = torch.ones((tokens, 5120), dtype=torch.bfloat16, device="cuda")
        weights = torch.full((tokens, 8), 0.125, dtype=torch.float32, device="cuda")
        ids = {
            experts: (
                torch.arange(tokens * 8, device="cuda").view(tokens, 8) + rank * 13
            )
            % experts
            for experts in (384, 128)
        }

        def roundtrip(experts):
            packed, scales, _, _, _ = prepare.prepare(
                x,
                weights,
                ids[experts],
                experts,
                None,
                False,
                quant,
            )
            assert packed.shape[-1] == 2560 and packed.dtype == torch.uint8
            assert scales.dtype == torch.uint16
            expert_output = torch.ones(
                (*packed.shape[:-1], 5120), dtype=torch.bfloat16, device="cuda"
            )
            output = torch.empty_like(x)
            hook, receiver = prepare.finalize_async(
                output,
                expert_output,
                weights,
                ids[experts],
                False,
                TopKWeightAndReduceDelegate(),
            )
            hook()
            receiver()
            return output

        # Dense packed data from the smaller layout overlaps control counters
        # in the larger layout. Sparse BF16/topk=1 traffic missed this failure.
        for experts in (384, 384, 128, 128, 384, 384, 128, 384):
            output = roundtrip(experts)
            torch.cuda.synchronize()
            torch.testing.assert_close(output, x, rtol=0, atol=0)
        graphs, outputs = {}, {}
        for experts in (384, 128):
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs[experts] = roundtrip(experts)
            graphs[experts] = graph
        # Replay without rerunning Python layout bookkeeping.
        for experts in (128, 384, 128, 128, 384):
            graphs[experts].replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(outputs[experts], x, rtol=0, atol=0)
        if rank == 0:
            print(
                "MXFP4 shared buffer eager + alternating graph replay PASS", flush=True
            )
        dist.barrier()
    finally:
        if buffer is not None:
            buffer.destroy()
        dist.destroy_process_group()


def test_dspark_shared_ll_buffer(tmp_path):
    import torch

    assert torch.cuda.device_count() >= 8
    torch.multiprocessing.spawn(
        _worker, args=(str(tmp_path / "deepep-ll"),), nprocs=8, join=True
    )
