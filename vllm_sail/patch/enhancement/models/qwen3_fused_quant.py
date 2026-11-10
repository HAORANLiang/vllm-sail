# ruff: noqa: E731, W291, W293, UP037
# ruff: noqa: F821
# Copied bodies resolve globals in their upstream module through bind_body.
# SPDX-License-Identifier: Apache-2.0
"""Fuse Qwen3 RMSNorm with dynamic FP8 group quantization when compatible."""

from __future__ import annotations

from vllm.model_executor.models import qwen3 as _qwen

from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch


def _setup_fused_rmsnorm_on_linear(linear, residual, weight, epsilon):
    from vllm.platforms import current_platform

    if not current_platform.is_ppu() or not current_platform.has_device_capability(89):
        return False
    method = getattr(linear, "quant_method", None)
    kernel = getattr(method, "fp8_linear", None)
    quant = getattr(kernel, "quant_fp8", None)
    if (
        quant is None
        or not getattr(quant, "is_group_quant", False)
        or quant.static
        or quant.use_ue8m0
        or quant.tma_aligned_scales
        or quant.num_token_padding is not None
        or quant._forward_method != quant.forward_cuda
        or residual.ndim != 2
        or not residual.is_contiguous()
        or quant.group_size <= 0
        or quant.group_size & (quant.group_size - 1)
        or residual.shape[-1] % quant.group_size
    ):
        return False
    quant.set_fused_rmsnorm(residual, weight, epsilon)
    return True


def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # PPU MODIFICATION: begin
    from vllm.platforms import current_platform

    from vllm_sail import envs as ppu_envs
    from vllm_sail.patch.enhancement.models.qwen3_fused_quant import (
        _setup_fused_rmsnorm_on_linear,
    )

    # Fused RMSNorm+quant is only beneficial when we have a previous
    # residual (i.e., not the first layer).
    use_fused = (
        ppu_envs.VLLM_SAIL_FUSED_RMSNORM_QUANT and current_platform.is_ppu() and residual is not None
    )

    # PPU MODIFICATION: end
    # Self Attention
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    # PPU MODIFICATION: begin
    elif not use_fused:
        hidden_states, residual = self.input_layernorm(
            hidden_states, residual
        )

    if use_fused:
        setup_ok = _setup_fused_rmsnorm_on_linear(
            self.self_attn.qkv_proj,
            residual,
            self.input_layernorm.weight.data,
            self.input_layernorm.variance_epsilon,
        )
        if not setup_ok:
            # The linear does not support fused quant context
            # (e.g. INT8 quant, unquantized). The skipped
            # input_layernorm must be executed as fallback.
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual
            )

    # PPU MODIFICATION: end
    hidden_states = self.self_attn(
        positions=positions,
        hidden_states=hidden_states,
    )

    # Fully Connected
    # PPU MODIFICATION: begin
    if use_fused:
        setup_ok = _setup_fused_rmsnorm_on_linear(
            self.mlp.gate_up_proj,
            residual,
            self.post_attention_layernorm.weight.data,
            self.post_attention_layernorm.variance_epsilon,
        )
        if not setup_ok:
            # The linear does not support fused quant context
            # (e.g. INT8 quant, unquantized). The skipped
            # post_attention_layernorm must be executed as
            # fallback to update residual and normalize.
            hidden_states, residual = (
                self.post_attention_layernorm(hidden_states, residual)
            )
    else:
        hidden_states, residual = (
            self.post_attention_layernorm(hidden_states, residual)
        )

    # PPU MODIFICATION: end
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual


forward = patch(
    _qwen.__name__,
    "Qwen3DecoderLayer.forward",
    reason="Fuse residual RMSNorm with supported QuantFP8 input quantization, with norm fallback for other linear kernels.",
    affected_versions=">=0.33.0,<0.34.0",
    remove_when="Upstream exposes a fused norm/quant producer-consumer interface.",
)(bind_body(forward, _qwen))
