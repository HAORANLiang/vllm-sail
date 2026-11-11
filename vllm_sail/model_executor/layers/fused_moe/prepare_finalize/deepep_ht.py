# SPDX-License-Identifier: Apache-2.0
"""Preserve the BF16 dispatch width for PPU MXFP4 expert communication."""

from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht import (
    DeepEPHTPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input


class PPUDeepEPHTPrepareAndFinalize(DeepEPHTPrepareAndFinalize):
    def prepare_async(
        self,
        a1,
        topk_weights,
        topk_ids,
        num_experts,
        expert_map,
        apply_router_weight_on_input,
        quant_config,
        defer_input_quant=False,
    ):
        quantize_after_dispatch = (
            quant_config.quant_dtype == "mxfp4" and not defer_input_quant
        )
        receiver = super().prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant=defer_input_quant or quantize_after_dispatch,
        )
        if not quantize_after_dispatch:
            return receiver

        def receive():
            x, _, tokens_meta, ids, weights = receiver()
            # Quantize the received BF16 activations for the MXFP4 experts.
            x, scales = moe_kernel_quantize_input(
                x,
                quant_config.a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
                is_scale_swizzled=quant_config.is_scale_swizzled,
                mx_alignment=quant_config.mx_alignment,
            )
            return x, scales, tokens_meta, ids, weights

        return receive

    def _finalize(
        self,
        output,
        fused_expert_output,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input,
        weight_and_reduce_impl,
        do_async,
    ):
        # The modular kernel's zero-token shortcut uses empty_like(a1q).
        # Packed FP4 a1q has half the logical hidden width, even though that
        # shortcut returns BF16. DeepEP derives the combine width from x,
        # including on ranks with no local expert tokens.
        if (
            fused_expert_output.shape[0] == 0
            and fused_expert_output.shape[-1] != output.shape[-1]
        ):
            fused_expert_output = fused_expert_output.new_empty((0, output.shape[-1]))
        return super()._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            do_async,
        )
