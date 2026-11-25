# SPDX-License-Identifier: Apache-2.0
"""Keep SWA replay's padded rows out of the PPU MegaGate router."""

from __future__ import annotations

import functools
import sys

from vllm_sail.patch.utils import patch

_MODEL_STATE = "vllm.models.deepseek_v41.nvidia.model_state"
_REMOVE_WHEN = (
    "Upstream SWA replay publishes a persistent padding mask and MegaGate "
    "honors that mask when routing graph-capacity rows."
)
_METADATA = dict(
    reason=(
        "DP SWA replay pads idle replicas to the largest rank's graph capacity. "
        "Unused attention rows can contain non-finite values; MegaGate must "
        "exclude them instead of trapping and stranding peers in MegaMoE."
    ),
    affected_versions=">=0.33.0,<0.34.0",
    remove_when=_REMOVE_WHEN,
)


def install() -> None:
    import torch
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )
    from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState
    from vllm.platforms import current_platform
    from vllm.utils import deep_gemm

    prepare_original = DeepseekV41ModelState._prepare_replay_batch

    @functools.wraps(prepare_original)
    def prepare_replay_batch(self, input_batch, *args, **kwargs):
        result = prepare_original(self, input_batch, *args, **kwargs)
        layers = self.decoder_replay_layers
        batch = None if layers is None else layers.replay_batch
        if not current_platform.is_ppu() or batch is None or self.pcp_size != 1:
            return result
        # The captured graph must read the same allocation on later steps.
        # Only values change when an idle rank later receives real requests.
        padding = getattr(self, "_sail_replay_padding", None)
        if padding is None:
            padding = torch.empty_like(self._kept_rows, dtype=torch.bool)
            self._sail_replay_padding = padding
        num_rows = batch.rows.shape[0]
        num_padded = batch.forward_context.batch_descriptor.num_tokens
        mask = padding[:num_padded]
        torch.index_select(input_batch.is_padding, 0, batch.rows, out=mask[:num_rows])
        mask[num_rows:].fill_(True)
        batch.forward_context.is_padding = mask
        return result

    patch(
        _MODEL_STATE, "DeepseekV41ModelState._prepare_replay_batch", **_METADATA
    )(prepare_replay_batch)

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
