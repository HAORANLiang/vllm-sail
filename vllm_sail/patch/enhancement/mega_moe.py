# SPDX-License-Identifier: Apache-2.0
"""Redirect the upstream MegaMoE experts and gate on PPU."""

from __future__ import annotations

import sys

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.32.0,<0.33.0"
_DEEPSEEK = "vllm.models.deepseek_v4.nvidia.model"
_EXPERT_CONSUMERS = (
    _DEEPSEEK,
    "vllm.models.kimi_k3.nvidia.model",
    "vllm.models.minimax_m3.nvidia.model",
)
_installed = False


def install() -> None:
    """Install the redirects once on PPU."""
    global _installed
    from vllm.platforms import current_platform

    if _installed or not current_platform.is_ppu():
        return

    from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe
    from vllm.utils import deep_gemm

    from vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_mega_moe import (
        PPUDeepGemmMegaMoEExperts,
        mega_gate,
    )

    for provider, name, replacement, consumers, reason, remove_when in (
        (
            deep_gemm_mega_moe,
            "DeepGemmMegaMoEExperts",
            PPUDeepGemmMegaMoEExperts,
            _EXPERT_CONSUMERS,
            "SAIL MegaMoE needs MXFP4 dispatch and PPU-specific scale layouts.",
            "vLLM supports selecting an out-of-tree MegaMoE expert backend.",
        ),
        (
            deep_gemm,
            "bf16_mega_gate",
            mega_gate,
            (_DEEPSEEK,),
            "SAIL DeepGEMM needs a fallback for the SM100 MegaGate kernel.",
            "SAIL DeepGEMM provides a compatible MegaGate implementation.",
        ),
    ):
        original = getattr(provider, name)
        patch(
            provider.__name__,
            name,
            reason=reason,
            affected_versions=_AFFECTED,
            remove_when=remove_when,
        )(replacement)

        # Consumers loaded after the provider is patched import the new symbol.
        for module_name in consumers:
            consumer = sys.modules.get(module_name)
            if consumer is not None and getattr(consumer, name, None) is original:
                patch(
                    module_name,
                    name,
                    reason=reason,
                    affected_versions=_AFFECTED,
                    remove_when=remove_when,
                )(replacement)
    _installed = True
