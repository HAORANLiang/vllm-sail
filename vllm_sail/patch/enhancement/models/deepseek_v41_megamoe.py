# SPDX-License-Identifier: Apache-2.0
"""Select SAIL's W4A4 implementation at the native DeepSeek MoE seam."""

from __future__ import annotations

import typing

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MegaMoEExperts
from vllm.platforms import current_platform

from vllm_sail.models.deepseek_v41.mega_moe import PPUDeepGemmMegaMoEExperts, mega_gate
from vllm_sail.patch.utils import PATCH_MARKER, patch

if typing.TYPE_CHECKING:
    import torch
    from vllm.models.deepseek_v4.nvidia.model import MegaGateRoutingMetadata

_MODEL = "vllm.models.deepseek_v4.nvidia.model"
_GATE = "vllm.utils.deep_gemm"


@patch(
    _MODEL,
    "DeepseekV4MoE._init_mega_moe_experts",
    reason="SAIL MegaMoE uses MXFP4 dispatch and PPU-specific expert scale layouts instead of the SM100 FP8 API.",
    affected_versions=">=0.31.0,<0.32.0",
    remove_when="The native DeepSeek MegaMoE path accepts an out-of-tree expert backend.",
)
def _init_mega_moe_experts(
    self,
    vllm_config: VllmConfig,
    config,
    prefix: str,
) -> None:
    self.ep_group = get_ep_group()
    self.ep_size = self.ep_group.world_size
    self.ep_rank = self.ep_group.rank_in_group

    eplb_config = vllm_config.parallel_config.eplb_config
    self.n_redundant_experts = eplb_config.num_redundant_experts
    self.n_shared_experts = config.n_shared_experts or 0
    self.n_logical_experts = self.n_routed_experts
    self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
    assert self.n_physical_experts % self.ep_size == 0, (
        f"n_physical_experts={self.n_physical_experts} must be divisible by "
        f"ep_size={self.ep_size}. Adjust num_redundant_experts."
    )
    self.n_local_physical_experts = self.n_physical_experts // self.ep_size
    self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
    self.physical_expert_end = (
        self.physical_expert_start + self.n_local_physical_experts
    )

    self.n_local_experts = self.n_local_physical_experts
    self.experts_start_idx = self.physical_expert_start
    self.experts_end_idx = self.physical_expert_end

    # Native DeepGEMM fusion requires each EP rank to own the complete
    # shared MLP. Sequence parallel replicates those weights while sharding
    # tokens. TP=1 is also naturally replicated. With PP+TP the shared MLP
    # remains tensor-sharded, so retain the serial path. The FlashInfer
    # megakernel has no shared-expert fusion, so it keeps the serial path.
    fuse_shared_experts = bool(
        self.shared_experts is not None
        and not envs.VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION
        and (self.use_sequence_parallel or self.tp_size == 1)
        and not self.use_fi_mega_moe
    )

    activation_clamp = (
        float(self.swiglu_limit) if self.swiglu_limit is not None else None
    )
    if self.use_fi_mega_moe:
        # Deferred: fi_moe subclasses DeepseekV4MegaMoEExperts, so a
        # module-level import here would be circular.
        from vllm.models.deepseek_v4.nvidia.fi_moe import (
            DeepseekV4MegaMoEExpertsFI,
        )

        experts_cls: type[DeepseekV4MegaMoEExperts] = DeepseekV4MegaMoEExpertsFI
    else:
        experts_cls = DeepseekV4MegaMoEExperts
        # PPU MODIFICATION: begin
        if current_platform.is_ppu():
            experts_cls = PPUDeepGemmMegaMoEExperts
            fuse_shared_experts = False
        # PPU MODIFICATION: end
    expert_kwargs: dict[str, typing.Any] = dict(
        num_experts=self.n_physical_experts,
        num_local_experts=self.n_local_physical_experts,
        experts_start_idx=self.physical_expert_start,
        num_logical_experts=self.n_logical_experts,
        top_k=self.n_activated_experts,
        hidden_size=config.hidden_size,
        intermediate_size=config.moe_intermediate_size,
        num_shared_experts=(self.n_shared_experts if fuse_shared_experts else 0),
        prefix=f"{prefix}.experts",
    )
    if self.use_fi_mega_moe:
        expert_kwargs["activation_clamp"] = activation_clamp
    self.experts = experts_cls(vllm_config, **expert_kwargs)


@patch(
    _GATE,
    "bf16_mega_gate",
    reason="SAIL's W4A4 MegaMoE API does not export the SM100 MegaGate; preserve sqrtsoftplus, hash and image routing on PPU.",
    affected_versions=">=0.31.0,<0.32.0",
    remove_when="SAIL DeepGEMM provides a compatible MegaGate implementation.",
)
def route_experts(*args, **kwargs):
    if current_platform.is_ppu():
        return mega_gate(*args, **kwargs)
    original = getattr(route_experts, PATCH_MARKER)[f"{_GATE}.bf16_mega_gate"]
    return original(*args, **kwargs)


@patch(
    _MODEL,
    "DeepseekV4MoE.forward",
    reason="Use GateLinear and fused top-k for PPU MegaMoE at every batch size to avoid the eager MegaGate sort and elementwise kernels.",
    affected_versions=">=0.31.0,<0.32.0",
    remove_when="SAIL DeepGEMM provides a compatible fused MegaGate implementation.",
)
def forward(
    self,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor | None = None,
    mega_gate_metadata: MegaGateRoutingMetadata | None = None,
) -> torch.Tensor:
    if self.gate.tid2eid is not None and input_ids is None:
        raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")
    # getattr for test doubles that fake the gate module.
    bias_vl = getattr(self.gate, "bias_vl", None)
    if bias_vl is not None and input_ids is None:
        raise ValueError("DeepSeek V4 vision MoE routing requires input_ids.")

    if not self.use_mega_moe:
        return self._forward_fused_moe(hidden_states, input_ids)

    org_shape = hidden_states.shape
    # Small local padded batches favor GateLinear; 128-expert gates cross earlier.
    gate_threshold = 1 if self.gate.weight.shape[0] == 128 else 16
    # PPU MODIFICATION: begin
    # SAIL has fused top-k but no fused MegaGate. Its eager MegaGate fallback
    # launches a full sort and separate scoring/normalization kernels.
    if current_platform.is_ppu() or hidden_states.shape[0] <= gate_threshold:
    # PPU MODIFICATION: end
        router_logits, _ = self.gate(hidden_states)
        topk_weights, topk_ids = fused_topk_bias(
            hidden_states=hidden_states,
            gating_output=router_logits,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias.data
            if self.gate.e_score_correction_bias is not None
            else None,
            topk=self.n_activated_experts,
            renormalize=self.renormalize,
            indices_type=self.hash_indices_dtype,
            input_tokens=input_ids,
            hash_indices_table=self.gate.tid2eid,
            routed_scaling_factor=self.routed_scaling_factor,
            bias_vl=bias_vl.data if bias_vl is not None else None,
            image_sentinel_lo=self.image_sentinel_lo if bias_vl is not None else 0,
        )
    else:
        from vllm.utils.deep_gemm import bf16_mega_gate

        if mega_gate_metadata is None:
            raise RuntimeError(
                "MegaMoE routing metadata must be prepared once by the model."
            )

        unmapped_topk_idx = None
        fix_routing_mask = None
        if self.gate.tid2eid is not None:
            if mega_gate_metadata.safe_hash_input_ids is None:
                raise RuntimeError("Hash routing requires prepared input IDs.")
            if mega_gate_metadata.hash_token_mask is None:
                raise RuntimeError("Hash routing requires a prepared routing mask.")
            unmapped_topk_idx = self.gate.tid2eid[
                mega_gate_metadata.safe_hash_input_ids
            ]
            fix_routing_mask = mega_gate_metadata.hash_token_mask

        topk_weights, topk_ids = bf16_mega_gate(
            hidden_states,
            self.gate.weight,
            self.n_activated_experts,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            ep_rank=self.ep_rank,
            bias=self.gate.e_score_correction_bias.data
            if self.gate.e_score_correction_bias is not None
            else None,
            image_bias=bias_vl.data if bias_vl is not None else None,
            image_token_mask=mega_gate_metadata.image_token_mask,
            fix_routing_mask=fix_routing_mask,
            unmapped_topk_idx=unmapped_topk_idx,
        )
    activation_clamp = (
        float(self.swiglu_limit) if self.swiglu_limit is not None else None
    )
    final_hidden_states = self.experts(
        hidden_states,
        topk_weights,
        topk_ids,
        activation_clamp=activation_clamp,
    )

    if (
        self.shared_experts is not None
        and not self.experts.has_fused_shared_experts
    ):
        shared_output = self.shared_experts(hidden_states)
        final_hidden_states += shared_output

    return final_hidden_states.view(org_shape)
