# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SAIL W4A4 MegaMoE with direct kernel-layout loading and native EPLB."""

from __future__ import annotations

import torch
import vllm.envs as envs
from vllm.distributed import get_ep_group
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_MXFP4,
    DeepGemmMegaMoEExperts,
)
from vllm.model_executor.layers.fused_moe.router.base_router import (
    eplb_map_to_physical_and_record,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

from vllm_sail.utils.deep_gemm_mega import check_mega_moe_supported, mega_moe_api


def mega_gate(
    x,
    weight,
    num_topk,
    *,
    scoring_func,
    routed_scaling_factor,
    ep_rank,
    bias=None,
    image_bias=None,
    image_token_mask=None,
    fix_routing_mask=None,
    unmapped_topk_idx=None,
):
    """Preserve DeepSeek routing when SAIL has no SM100 MegaGate kernel."""
    if scoring_func != "sqrtsoftplus":
        raise ValueError("DeepSeek SAIL MegaMoE requires sqrtsoftplus routing.")
    logits = torch.nn.functional.linear(x.float(), weight.float())
    scores = torch.nn.functional.softplus(logits).sqrt()
    selection = scores if bias is None else scores + bias
    if image_bias is not None:
        if image_token_mask is None:
            raise ValueError("Image routing bias requires the prepared token mask.")
        selection = torch.where(
            image_token_mask[:, None], scores + image_bias, selection
        )
    ids = torch.argsort(selection, dim=-1, descending=True, stable=True)[:, :num_topk]
    if fix_routing_mask is not None:
        if unmapped_topk_idx is None:
            raise ValueError("Hash routing mask requires prepared expert IDs.")
        ids = torch.where(fix_routing_mask[:, None], unmapped_topk_idx, ids)
    weights = scores.gather(1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True) * routed_scaling_factor
    return weights.contiguous(), ids.to(torch.int64).contiguous()


class PPUDeepGemmMegaMoEExperts(DeepGemmMegaMoEExperts):
    """Retain vLLM routing/loading; dispatch BF16 tokens as MXFP4 on PPU."""

    _symm_buffers: dict[tuple, object] = {}

    def __init__(
        self,
        vllm_config,
        *,
        num_experts,
        num_local_experts,
        experts_start_idx,
        top_k,
        hidden_size,
        intermediate_size,
        activation,
        weight_format,
        shared_experts=None,
        num_logical_experts=None,
        fast_math=True,
        max_num_tokens=None,
        prefix="",
    ):
        # The SAIL kernel consumes a different scale layout and dispatch API.
        if weight_format != MEGA_MOE_MXFP4 or activation.name != "swiglu":
            raise ValueError("SAIL MegaMoE requires MXFP4 weights and SwiGLU.")
        self.weight_format = weight_format
        self.activation = activation
        self.fast_math = fast_math
        torch.nn.Module.__init__(self)
        self.prefix = prefix
        self.capture_fn = None
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_start_idx = experts_start_idx
        self.experts_end_idx = experts_start_idx + num_local_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_shared_experts = 0
        self.max_num_tokens = (
            max_num_tokens
            if max_num_tokens is not None
            else vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.num_logical_experts = (
            num_logical_experts if num_logical_experts is not None else num_experts
        )
        self.eplb_state = EplbLayerState()
        self._register_weight(
            "w13", (num_local_experts, 2 * intermediate_size, hidden_size), None
        )
        self._register_weight(
            "w2", (num_local_experts, hidden_size, intermediate_size), None
        )
        for name, param in self.named_parameters():
            set_weight_attrs(param, {"weight_loader": self.weight_loader})
            if name.endswith("_scale"):
                param.quant_method = "block"
        self._check_runtime_supported()
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self

    @property
    def l1_weights(self):
        return self.w13_weight.data, self.w13_weight_scale.data

    @property
    def l2_weights(self):
        return self.w2_weight.data, self.w2_weight_scale.data

    def _check_runtime_supported(self):
        check_mega_moe_supported(
            self.w13_weight.device, self.hidden_size, self.intermediate_size
        )

    def weight_loader(
        self,
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success=False,
    ):
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"Unsupported expert shard id: {shard_id}")
        expected = "w2_" if shard_id == "w2" else "w13_"
        local_ids = (
            self._map_global_expert_id(expert_id) if expected in weight_name else []
        )
        for local_id in local_ids:
            self._load_weight(
                param.data[local_id], loaded_weight, shard_id, "scale" in weight_name
            )
        return bool(local_ids) if return_success else None

    weight_loader.supports_moe_loading = True

    def _register_weight(self, name, shape, spec):
        experts, n, k = shape
        self.register_parameter(
            f"{name}_weight",
            torch.nn.Parameter(
                torch.zeros(experts, n, k // 2, dtype=torch.uint8), requires_grad=False
            ),
        )
        scales = torch.zeros(experts, k // 64, n, dtype=torch.uint16).transpose(1, 2)
        self.register_parameter(
            f"{name}_weight_scale", torch.nn.Parameter(scales, requires_grad=False)
        )

    def _load_weight(self, dst, src, shard_id, is_scale):
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"Unsupported expert shard id: {shard_id}")
        if not is_scale:
            src = src.view(torch.uint8)
            rows = dst.shape[0] // (1 if shard_id == "w2" else 2)
            if src.shape != (rows, dst.shape[1]):
                raise ValueError(f"Invalid {shard_id} packed weight shape: {src.shape}")
        else:
            if src.dtype not in (torch.uint8, torch.float8_e8m0fnu):
                raise ValueError(
                    "MegaMoE requires raw UE8M0 scales, not NVFP4 E4M3 scales."
                )
            rows, k = (
                (self.hidden_size, self.intermediate_size)
                if shard_id == "w2"
                else (self.intermediate_size, self.hidden_size)
            )
            if src.shape != (rows, k // 32):
                raise ValueError(f"Invalid {shard_id} UE8M0 scale shape: {src.shape}")
            src = src.view(torch.uint8).contiguous().view(torch.uint16)
        if shard_id != "w2":
            half = 0 if shard_id == "w1" else 1
            dst = dst[half::2]
        dst.copy_(src)

    def get_symm_buffer(self):
        group = get_ep_group().device_group
        key = (
            id(group),
            torch.cuda.current_device(),
            self.num_experts,
            self.max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
        )
        if key not in self._symm_buffers:
            self._symm_buffers[key] = mega_moe_api().get_symm_buffer_for_mega_moe(
                group,
                self.num_experts,
                self.max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                mma_type="mxf4xmxf4",
                activation="swiglu",
            )
        return self._symm_buffers[key]

    def forward(self, hidden_states, topk_weights, topk_ids, shared_experts=None):
        tokens = hidden_states.shape[0]
        if tokens > self.max_num_tokens:
            raise ValueError(
                f"MegaMoE token count {tokens} exceeds capacity {self.max_num_tokens}."
            )
        self._synchronize_first_launch()
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            padding = get_forward_context().is_padding
            if padding is not None:
                topk_ids = torch.where(padding[:tokens, None], -1, topk_ids)
        state = self.eplb_state
        if state.logical_to_physical_map is not None:
            topk_ids = eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                expert_load_view=state.expert_load_view,
                logical_to_physical_map=state.logical_to_physical_map,
                logical_replica_count=state.logical_replica_count,
                record_enabled=state.should_record_tensor,
                num_unpadded_tokens=(
                    state.num_unpadded_tokens_tensors[dbo_current_ubatch_id()]
                    if state.num_unpadded_tokens_tensors is not None
                    else None
                ),
            )
        api, buffer = mega_moe_api(), self.get_symm_buffer()
        api.mega_moe_pre_dispatch(
            hidden_states,
            topk_ids.to(torch.int32).contiguous(),
            topk_weights.to(torch.float32).contiguous(),
            sym_buffer=buffer,
            num_tokens=tokens,
        )
        output = torch.empty_like(hidden_states, dtype=torch.bfloat16)
        # Every EP rank participates, including ranks with zero local tokens.
        api.fp4_fp4_mega_moe(
            output,
            self.l1_weights,
            self.l2_weights,
            buffer,
            recipe=(1, 1, 32),
            activation="swiglu",
            activation_clamp=self.activation.clamp,
            fast_math=self.fast_math,
        )
        if shared_experts is not None:
            output += shared_experts(hidden_states)
        return output
