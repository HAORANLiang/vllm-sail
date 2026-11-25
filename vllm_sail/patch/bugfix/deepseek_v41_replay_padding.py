# SPDX-License-Identifier: Apache-2.0
"""Honor the upstream SWA replay padding mask in the PPU MegaGate router."""

from __future__ import annotations

import functools
import sys

from vllm_sail.patch.utils import patch

_REMOVE_WHEN = (
    "Upstream MegaGate honors ForwardContext.is_padding when routing "
    "graph-capacity rows."
)
_METADATA = dict(
    reason=(
        "DP SWA replay pads idle replicas to the largest rank's graph capacity. "
        "Unused attention rows can contain non-finite values; MegaGate must "
        "exclude them instead of trapping and stranding peers in MegaMoE."
    ),
    affected_versions=">=0.34.0,<0.35.0",
    remove_when=_REMOVE_WHEN,
)


def install() -> None:
    import torch
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )
    from vllm.platforms import current_platform
    from vllm.utils import deep_gemm

    gate_original = deep_gemm.bf16_mega_gate

    @functools.wraps(gate_original)
    def bf16_mega_gate(x, weight, num_topk, **kwargs):
        padding = None
        if current_platform.is_ppu() and is_forward_context_available():
            padding = get_forward_context().is_padding
        if padding is None:
            return gate_original(x, weight, num_topk, **kwargs)
        # Native MegaGate's mask suppresses invalid-score checks and emits
        # expert -1 / weight 0 for inactive rows, preserving real-row checks.
        deep_gemm._lazy_init()
        indices = torch.empty((x.shape[0], num_topk), dtype=torch.int64, device=x.device)
        weights = torch.empty(
            (x.shape[0], num_topk), dtype=torch.float32, device=x.device
        )
        deep_gemm._bf16_mega_gate_impl(
            x,
            weight,
            num_topk,
            use_shared_as_routed=False,
            num_shared_experts=0,
            mask=~padding[: x.shape[0]],
            out=(indices, weights),
            **kwargs,
        )
        return weights, indices

    patch("vllm.utils.deep_gemm", "bf16_mega_gate", **_METADATA)(bf16_mega_gate)
    # The model may already have imported this function while plugins loaded.
    name = "vllm.models.deepseek_v4.nvidia.model"
    consumer = sys.modules.get(name)
    if consumer is not None and getattr(consumer, "bf16_mega_gate", None) is gate_original:
        patch(name, "bf16_mega_gate", **_METADATA)(bf16_mega_gate)
