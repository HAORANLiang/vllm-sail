# SPDX-License-Identifier: Apache-2.0
"""Two-device regressions for MXFP4 dispatch and BF16 combine width."""

from datetime import timedelta
from pathlib import Path

import pytest

pytestmark = pytest.mark.ppu
PPU_CI_MIN_DEVICES = 2


def _worker(rank, rendezvous):
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    from types import SimpleNamespace

    import deep_ep
    from vllm.model_executor.layers.fused_moe.config import ocp_mx_moe_quant_config
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernelModularImpl,
    )
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceNoOP,
    )

    from vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        PPUDeepGemmExpertsMXFP4,
    )
    from vllm_sail.model_executor.layers.fused_moe.prepare_finalize.deepep_ht import (
        PPUDeepEPHTPrepareAndFinalize,
    )

    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    buffer = None
    try:
        buffer = deep_ep.Buffer(
            dist.group.WORLD,
            num_nvl_bytes=128 * 1024 * 1024,
            num_rdma_bytes=0,
            low_latency_mode=False,
            explicitly_destroy=True,
        )
        prepare = PPUDeepEPHTPrepareAndFinalize(buffer, 2, 2, rank * 8)
        config = ocp_mx_moe_quant_config(
            "mxfp4",
            torch.ones(1, device="cuda"),
            torch.ones(1, device="cuda"),
            block_shape=[1, 32],
        )
        for tokens, destination in ((3, rank), (0 if rank == 0 else 2, 1), (2, 0)):
            x = torch.ones(tokens, 5120, dtype=torch.bfloat16, device="cuda")
            ids = torch.full(
                (tokens, 1), destination * 8, dtype=torch.int64, device="cuda"
            )
            weights = torch.ones(tokens, 1, dtype=torch.float32, device="cuda")
            packed, scales, _, recv_ids, recv_weights = prepare.prepare(
                x,
                weights,
                ids,
                16,
                None,
                False,
                config,
            )
            assert packed.shape[1] == 2560 and packed.dtype == torch.uint8
            assert scales is not None and scales.dtype == torch.uint16
            # Identity experts isolate communication from GEMM. Every input has
            # one destination with weight one, so BF16 combine must return ones.
            expert_output = torch.ones(
                packed.shape[0], 5120, dtype=torch.bfloat16, device="cuda"
            )
            if packed.shape[0] == 0:
                # Run the actual modular-kernel empty-expert shortcut: it
                # returns BF16 with the packed input's half-width.
                experts = PPUDeepGemmExpertsMXFP4.__new__(PPUDeepGemmExpertsMXFP4)
                expert_output = FusedMoEKernelModularImpl._fused_experts(
                    SimpleNamespace(fused_experts=experts),
                    in_dtype=torch.bfloat16,
                    a1q=packed,
                    a1q_scale=scales,
                    w1=torch.empty((8, 128, 2560), dtype=torch.uint8, device="cuda"),
                    w2=torch.empty((8, 5120, 32), dtype=torch.uint8, device="cuda"),
                    topk_weights=recv_weights,
                    topk_ids=recv_ids,
                    activation=None,
                    global_num_experts=16,
                    local_num_experts=8,
                    expert_map=None,
                    apply_router_weight_on_input=False,
                    expert_tokens_meta=None,
                )
                assert expert_output.shape == (0, 2560)
            output = torch.empty_like(x)
            prepare.finalize(
                output,
                expert_output,
                recv_weights,
                recv_ids,
                False,
                TopKWeightAndReduceNoOP(),
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(output, x, rtol=0, atol=0)
        dist.barrier()
    finally:
        if buffer is not None:
            buffer.destroy()
        dist.destroy_process_group()


def test_mxfp4_ht_dispatch_combine(tmp_path):
    import torch

    assert torch.cuda.device_count() >= 2
    torch.multiprocessing.spawn(
        _worker, args=(str(tmp_path / "deepep-ht"),), nprocs=2, join=True
    )
