# SPDX-License-Identifier: Apache-2.0
"""Quantized routed up-projection support for the Kimi K3 latent-MoE tail.

The fused latent-MoE path assumes the replicated routed up-projection is a
plain dense weight: tier 0's CuTeDSL operator and tier 2's ``narrow()+addmm_``
read ``up_proj.weight`` directly, and tier 1 multiplies it with ``torch.mm``.
A PPU checkpoint may quantize that projection, and then only the layer's quant
method can run its GEMM.

These patches record whether the projection is quantized, force every batch
onto tier 1 when it is, and route both the tier-1 GEMM and
``KimiRoutedOutputTransform``'s residual epilogue through the layer. An
unquantized up-projection keeps delegating to upstream unchanged.
"""

from __future__ import annotations

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.kimi_k3.nvidia.latent_moe_runner import (
    LatentMoERunner,
    LatentTailTier,
)
from vllm.models.kimi_k3.nvidia.model import KimiRoutedOutputTransform
from vllm.platforms import current_platform
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.utils.torch_utils import aux_stream

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.33.0,<0.34.0"

# Captured before patching so every replacement can fall back to upstream.
_upstream_runner_init = LatentMoERunner.__init__
_upstream_select_tail_tier = LatentMoERunner._select_tail_tier
_upstream_overlap_allreduce_tail = LatentMoERunner._overlap_allreduce_tail
_upstream_transform_forward = KimiRoutedOutputTransform.forward


@patch(
    "vllm.models.kimi_k3.nvidia.latent_moe_runner",
    "LatentMoERunner.__init__",
    reason=(
        "PPU checkpoints can quantize the replicated routed up-projection, and "
        "the latent-MoE tail can only run its GEMM through the layer's quant "
        "method. Record whether up_proj is quantized on PPU; delegates to "
        "upstream elsewhere."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "LatentMoERunner derives tail eligibility from the up-projection's "
        "quant method instead of assuming a plain weight."
    ),
)
def runner_init(self, *args, **kwargs) -> None:
    _upstream_runner_init(self, *args, **kwargs)
    if not current_platform.is_ppu():
        return
    # PPU MODIFICATION: begin — a quantized up-proj cannot use the raw-weight
    # tiers, so the tail needs to know about it.
    assert self.enable_k3_latent_moe_tail_fusion is False
    transform = self.routed_output_transform
    self._up_proj_is_quantized = transform is not None and not isinstance(
        transform.up_proj.quant_method, UnquantizedLinearMethod
    )
    # PPU MODIFICATION: end


@patch(
    "vllm.models.kimi_k3.nvidia.latent_moe_runner",
    "LatentMoERunner._select_tail_tier",
    reason=(
        "A quantized routed up-projection cannot run tier 0's CuTeDSL operator "
        "or tier 2's narrow()+addmm_ shard, which both read the packed weight "
        "directly; force tier 1, the only tier that goes through the layer's "
        "quant method. Delegates to upstream for an unquantized weight."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "LatentMoERunner selects its tail from the up-projection quant method, "
        "so a quantized weight never reaches the raw-weight tiers."
    ),
)
def select_tail_tier(self, fused_output, shared_output):
    if not getattr(self, "_up_proj_is_quantized", False):
        return _upstream_select_tail_tier(self, fused_output, shared_output)
    # PPU MODIFICATION: begin — only tier 1 runs the up-projection through its
    # quant method. The attribute is PPU-only, so non-PPU always delegates.
    return LatentTailTier.ALLREDUCE_OVERLAP
    # PPU MODIFICATION: end


@patch(
    "vllm.models.kimi_k3.nvidia.latent_moe_runner",
    "LatentMoERunner._overlap_allreduce_tail",
    reason=(
        "Tier 1 normally multiplies the replicated weight with torch.mm, which "
        "cannot run a quantized up-projection; call the layer so its quant "
        "method owns the GEMM. Quantization forces every batch onto this tier, "
        "so the small-batch token guard no longer applies."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream runs the tier-1 up-projection through the linear layer (or "
        "its quant method) instead of torch.mm on the raw weight."
    ),
)
def overlap_allreduce_tail(self, fused_output, shared_output, trunc_size):
    if not getattr(self, "_up_proj_is_quantized", False):
        return _upstream_overlap_allreduce_tail(
            self, fused_output, shared_output, trunc_size
        )
    # PPU MODIFICATION: begin — verbatim upstream body, with the up-projection
    # GEMM routed through the layer and the small-batch guard dropped.
    transform = self.routed_output_transform
    assert transform is not None
    if transform.norm is not None:
        fused_latent = self.allreduce_norm_latent_out(fused_output, transform.norm)
    else:
        fused_latent = tensor_model_parallel_all_reduce(fused_output)

    # Overlap the shared-expert all-reduce with the up-projection GEMM.
    result, shared_output = maybe_execute_in_parallel(
        lambda: transform.up_proj(fused_latent)[0],
        lambda: tensor_model_parallel_all_reduce(shared_output),
        self._shared_ar_events[0],
        self._shared_ar_events[1],
        aux_stream(),
    )
    result.add_(shared_output)

    # Output is already fully reduced; this only strips padding.
    return self._maybe_reduce_final_output(result, trunc_size, output_is_reduced=True)
    # PPU MODIFICATION: end


@patch(
    "vllm.models.kimi_k3.nvidia.model",
    "KimiRoutedOutputTransform.forward",
    reason=(
        "KimiRoutedOutputTransform folds the shared output into the "
        "up-projection GEMM's beta-add epilogue through residual.addmm_ on the "
        "raw weight, which only works for an unquantized up-projection. On "
        "PPU, call the linear layer for a quantized weight and keep the "
        "epilogue otherwise."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream folds the residual through the layer's quant method, or "
        "exposes an equivalent epilogue for a quantized up-projection."
    ),
)
def transform_forward(self, hidden_states, residual=None):
    if (
        residual is None
        or not current_platform.is_ppu()
        or isinstance(self.up_proj.quant_method, UnquantizedLinearMethod)
    ):
        return _upstream_transform_forward(self, hidden_states, residual)
    # PPU MODIFICATION: begin — the beta-add epilogue reads the raw weight; a
    # quantized up-projection must run its own GEMM and be added afterwards.
    if self.norm is not None:
        hidden_states = self.norm(hidden_states)
    return residual.add_(self.up_proj(hidden_states)[0])
    # PPU MODIFICATION: end
