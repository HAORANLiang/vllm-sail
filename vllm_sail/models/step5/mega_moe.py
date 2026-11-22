# SPDX-License-Identifier: Apache-2.0
"""SAIL MXFP4 MegaMoE expert adaptation for Step5."""

from contextlib import nullcontext
from copy import copy
from types import SimpleNamespace

import torch
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
    override_forward_context,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.models.common.ops.sequence_parallel import sp_all_gather, sp_shard
from vllm.models.step5.nvidia.mega_moe import _MegaMoEMethod

from vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_mega_moe import (
    PPUDeepGemmMegaMoEExperts,
)
from vllm_sail.utils.deep_gemm_mega import mega_moe_api


class PPUStep5BF16MoERunner(MoERunner):
    """Tail layers stay BF16 and accept in-stage SP padding."""

    def __init__(self, *, sequence_parallel=False, **kwargs):
        super().__init__(**kwargs)
        self.sequence_parallel = sequence_parallel

    def forward(self, hidden_states, router_logits, padding=None):
        context = nullcontext()
        if self.sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)
            router_logits = sp_all_gather(router_logits)
            if is_forward_context_available():
                gathered = copy(get_forward_context())
                if gathered.dp_metadata is not None:
                    tp_size = get_tensor_model_parallel_world_size()
                    metadata = copy(gathered.dp_metadata)
                    # TP gather keeps aligned rows; DP dispatch/combine must
                    # use the same sizes.
                    counts = metadata.num_tokens_across_dp_cpu
                    metadata.num_tokens_across_dp_cpu = (
                        (counts + tp_size - 1) // tp_size
                    ) * tp_size
                    gathered.dp_metadata = metadata
                gathered.is_padding = (
                    sp_all_gather(padding[:, None].to(hidden_states.dtype))
                    .flatten()
                    .bool()
                    if padding is not None
                    else None
                )
                context = override_forward_context(gathered)
        with context:
            output = super().forward(hidden_states, router_logits)
        if self.sequence_parallel:
            output = sp_shard(output)
        if padding is not None:
            output = output.masked_fill(padding[: output.shape[0], None], 0)
        return output


class PPUStep5MegaMoEExperts(PPUDeepGemmMegaMoEExperts):
    """Reuses original MXFP4 weight loading; EP workspaces are isolated per PP stage."""

    def __init__(
        self,
        vllm_config,
        *,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size,
        quant_config,
        activation,
        activation_clamp,
        prefix,
        max_num_tokens=None,
    ):
        from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
            MEGA_MOE_MXFP4,
        )
        from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config

        parallel = vllm_config.parallel_config
        if not parallel.enable_expert_parallel:
            raise ValueError("SAIL Step5 MegaMoE requires expert parallelism")
        if parallel.enable_eplb or parallel.eplb_config.num_redundant_experts:
            raise ValueError("SAIL Step5 MegaMoE does not support EPLB")
        if parallel.enable_dbo or getattr(parallel, "ubatch_size", 1) > 1:
            raise ValueError("SAIL Step5 MegaMoE requires serial microbatches")
        if not isinstance(quant_config, Mxfp4Config):
            raise ValueError("SAIL Step5 MegaMoE requires MXFP4 weights")
        if activation not in ("silu", "swiglustep"):
            raise ValueError(f"Unsupported Step5 MegaMoE activation: {activation}")
        if activation == "swiglustep" and activation_clamp != 7.0:
            raise ValueError("Step5 swiglustep requires activation_clamp=7.0")
        group = get_ep_group().device_group
        if num_experts % group.size():
            raise ValueError("MegaMoE experts must divide evenly across EP ranks")
        local_experts = num_experts // group.size()
        super().__init__(
            vllm_config,
            num_experts=num_experts,
            num_local_experts=local_experts,
            experts_start_idx=group.rank() * local_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=SimpleNamespace(name="swiglu", clamp=activation_clamp),
            weight_format=MEGA_MOE_MXFP4,
            max_num_tokens=max_num_tokens,
            prefix=prefix,
        )
        if self.mma_type == "bf16xfp4":
            raise ValueError("Step5 requires a MegaMoE mode supporting SwiGLU")
        self.activation.name = "swiglu_step" if activation == "swiglustep" else "swiglu"
        self.quant_method = _MegaMoEMethod()
        self.ready = False
        self.pool = None

    def prepare_weights(self):
        if not self.ready:
            self._transform_weights(mega_moe_api(self.mma_type))
            self.ready = True

    def weight_loader(
        self,
        param,
        loaded_weight,
        weight_name=None,
        shard_id=None,
        expert_id=None,
        return_success=False,
    ):
        if self.ready:
            raise RuntimeError(
                "Reloading transformed Step5 MegaMoE weights is unsupported"
            )
        if expert_id is not None:
            return super().weight_loader(
                param, loaded_weight, weight_name, shard_id, expert_id, return_success
            )
        # Fused checkpoints keep the original gate/up layout.
        weight_name = next(
            name for name, value in self.named_parameters() if value is param
        )
        bank = "w2" if weight_name.startswith("w2_") else "w13"
        count = loaded_weight.shape[0]
        if count not in (self.num_experts, self.num_local_experts):
            raise ValueError("Invalid fused Step5 MegaMoE expert count")
        start = 0 if count == self.num_experts else self.experts_start_idx
        for index in range(count):
            expert_id = start + index
            value = loaded_weight[index]
            shards = (
                (("w2", value),)
                if bank == "w2"
                else zip(("w1", "w3"), value.chunk(2, dim=0), strict=True)
            )
            for shard, value in shards:
                super().weight_loader(param, value, weight_name, shard, expert_id)
        return True if return_success else None

    weight_loader.supports_moe_loading = True

    def get_symm_buffer(self):
        if self.pool is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Warm up Step5 MegaMoE before capturing a CUDA graph"
                )
            self._synchronize_first_launch()
            group = get_ep_group().device_group
            key = (
                id(group),
                torch.cuda.current_device(),
                self.num_experts,
                self.max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                self.mma_type,
                self.activation.name,
            )
            if key not in self._symm_buffers:
                self._symm_buffers[key] = mega_moe_api(
                    self.mma_type
                ).get_symm_buffer_for_mega_moe(
                    group,
                    self.num_experts,
                    self.max_num_tokens,
                    self.top_k,
                    self.hidden_size,
                    self.intermediate_size,
                    mma_type=self.mma_type,
                    activation=self.activation.name,
                )
            self.pool = self._symm_buffers[key]
        return self.pool

    def forward(self, hidden_states, topk_weights, topk_ids, padding=None):
        if not self.ready:
            raise RuntimeError(
                "Step5 MegaMoE weights must be prepared before inference"
            )
        tokens = hidden_states.shape[0]
        if tokens > self.max_num_tokens:
            raise ValueError("Step5 MegaMoE input exceeds its allocated token capacity")
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError("Step5 MegaMoE expects BF16 hidden states")
        if padding is not None:
            topk_ids = torch.where(padding[:tokens, None], -1, topk_ids)
            topk_weights = torch.where(padding[:tokens, None], 0, topk_weights)
        api = mega_moe_api(self.mma_type)
        l1, l2 = self.l1_weights, self.l2_weights
        buffer = self.get_symm_buffer()
        x = hidden_states.contiguous()
        api.mega_moe_pre_dispatch(
            x,
            topk_ids.to(torch.int32).contiguous(),
            topk_weights.to(torch.float32).contiguous(),
            buf_x=buffer.x,
            buf_x_sf=buffer.x_sf,
            buf_topk_idx=buffer.topk_idx,
            buf_topk_weights=buffer.topk_weights,
            num_tokens=tokens,
            mma_type=self.mma_type,
        )
        output = torch.empty_like(x)
        getattr(api, self.mma_format.kernel)(
            output,
            l1,
            l2,
            buffer,
            activation=self.activation.name,
            activation_clamp=self.activation.clamp,
            fast_math=self.fast_math,
            **({"recipe": (1, 1, 32)} if self.mma_format.transform is None else {}),
        )
        return output
