# SPDX-License-Identifier: Apache-2.0
"""Redirect the upstream MegaMoE experts on PPU."""

from __future__ import annotations

import sys

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.34.0,<0.35.0"
_REASON = "SAIL MegaMoE needs MXFP4 dispatch and PPU-specific scale layouts."
_REMOVE_WHEN = "vLLM supports selecting an out-of-tree MegaMoE expert backend."
_EXPERT_CONSUMERS = (
    "vllm.models.deepseek_v4.nvidia.model",
    "vllm.models.kimi_k3.nvidia.model",
    "vllm.models.minimax_m3.nvidia.model",
)
_installed = False


def install() -> None:
    """Install the expert redirect once on PPU."""
    global _installed
    from vllm.platforms import current_platform

    if _installed or not current_platform.is_ppu():
        return

    from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe

    from vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_mega_moe import (
        PPUDeepGemmMegaMoEExperts,
    )

    original = deep_gemm_mega_moe.DeepGemmMegaMoEExperts
    patch(
        deep_gemm_mega_moe.__name__,
        "DeepGemmMegaMoEExperts",
        reason=_REASON,
        affected_versions=_AFFECTED,
        remove_when=_REMOVE_WHEN,
    )(PPUDeepGemmMegaMoEExperts)

    # Consumers loaded after the provider is patched import the new symbol.
    for module_name in _EXPERT_CONSUMERS:
        consumer = sys.modules.get(module_name)
        if (
            consumer is not None
            and getattr(consumer, "DeepGemmMegaMoEExperts", None) is original
        ):
            patch(
                module_name,
                "DeepGemmMegaMoEExperts",
                reason=_REASON,
                affected_versions=_AFFECTED,
                remove_when=_REMOVE_WHEN,
            )(PPUDeepGemmMegaMoEExperts)
    _install_step5()
    _installed = True


def _install_step5():
    from copy import copy

    import torch
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import FusedMoEFactory
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
    from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
    from vllm.models.step5.nvidia import mega_moe, model
    from vllm.platforms import current_platform

    from vllm_sail.models.step5.mega_moe import (
        PPUStep5BF16MoERunner,
        PPUStep5MegaMoEExperts,
    )

    original = mega_moe.Step5MegaMoE
    original_sp = model._step5_use_sequence_parallel
    metadata = dict(
        reason="Step5 needs SAIL MXFP4 MegaMoE and stage-local sequence parallelism for PP.",
        affected_versions=_AFFECTED,
        remove_when="Step5 exposes out-of-tree MegaMoE experts and enables stage-local SP with PP.",
    )
    patch(mega_moe.__name__, "Step5MegaMoEExperts", **metadata)(PPUStep5MegaMoEExperts)

    @patch(model.__name__, "_step5_use_sequence_parallel", **metadata)
    def _step5_use_sequence_parallel(vllm_config):
        if not current_platform.is_ppu():
            return original_sp(vllm_config)
        parallel = vllm_config.parallel_config
        return (
            parallel.enable_expert_parallel
            and parallel.tensor_parallel_size > 1
            and vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
        )

    @patch(mega_moe.__name__, "Step5MegaMoE", **metadata)
    def _step5_mega_moe(vllm_config, **kwargs):
        quant = kwargs["quant_config"]
        prefix = kwargs["prefix"]
        ignored = isinstance(quant, Mxfp4Config) and any(
            is_layer_skipped(
                prefix=candidate,
                ignored_layers=quant.ignored_layers,
                fused_mapping=quant.packed_modules_mapping,
            )
            for candidate in (prefix, f"{prefix}.routed_experts")
        )
        if not ignored:
            return original(vllm_config, **kwargs)
        # BF16 tail layers from the checkpoint are not requantized, and no
        # extra parameter-name hierarchy is added.
        fallback = copy(vllm_config)
        fallback.kernel_config = copy(vllm_config.kernel_config)
        fallback.kernel_config.moe_backend = "auto"
        kwargs.pop("activation_clamp")
        sequence_parallel = kwargs.pop("sequence_parallel", False)
        kwargs["is_sequence_parallel"] = False
        kwargs["quant_config"] = None
        with set_current_vllm_config(fallback):
            return FusedMoEFactory(
                **kwargs,
                reduce_results=True,
                router_logits_dtype=torch.float32,
                runner_cls=PPUStep5BF16MoERunner,
                runner_args={"sequence_parallel": sequence_parallel},
                skip_padding=False,
            )
