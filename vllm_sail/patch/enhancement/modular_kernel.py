# SPDX-License-Identifier: Apache-2.0
"""Double the K extent for packed-MXFP4 expert problems on PPU.

``FusedMoEExpertsModular.moe_problem_size`` derives the GEMM K extent from
``a1.size(-1)``. PPU's MXFP4 experts pack two e2m1 values per uint8, so the
trailing activation dimension is ``hidden_size // 2`` and the true K is twice
what upstream computes. The in-tree fork patches exactly this method; none of
the plugin's expert classes override it, so without this patch their workspace
and shape derivation runs with a halved K.

Delegating: upstream computes everything, and only the PPU packed-MXFP4 case
fixes up K.
"""

from __future__ import annotations

import inspect

import torch
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEExpertsModular,
)

from vllm_sail.patch.utils import patch

_upstream_moe_problem_size = FusedMoEExpertsModular.moe_problem_size

# The in-tree fork already carries this fix; detect it so applying the patch
# on top of fork source (test mode B) does not double K twice.
try:
    _UPSTREAM_ALREADY_DOUBLES_K = "K * 2" in inspect.getsource(
        _upstream_moe_problem_size
    )
except (OSError, TypeError):
    _UPSTREAM_ALREADY_DOUBLES_K = False


@patch(
    "vllm.model_executor.layers.fused_moe.modular_kernel",
    "FusedMoEExpertsModular.moe_problem_size",
    reason=(
        "PPU's MXFP4 experts pack two e2m1 values per uint8, so the trailing "
        "activation dimension is hidden_size // 2; upstream derives K from it "
        "unchanged and the PPU experts' workspace/shape derivation would run "
        "with a halved K."
    ),
    affected_versions=">=0.33.0,<0.34.0",
    remove_when=(
        "upstream derives K from the weight/quant metadata instead of the "
        "packed activation extent, or the PPU MXFP4 experts override "
        "moe_problem_size themselves."
    ),
)
def moe_problem_size(self, a1, w1, w2, topk_ids):
    E, M, N, K, topk = _upstream_moe_problem_size(self, a1, w1, w2, topk_ids)

    from vllm.platforms import current_platform

    if (
        not _UPSTREAM_ALREADY_DOUBLES_K
        and current_platform.is_ppu()
        and w1.dtype == torch.uint8
        and a1.dtype == torch.uint8
    ):
        K = K * 2
    return E, M, N, K, topk


from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEExperts

_upstream_supported = FusedMoEExperts.is_supported_config


@patch(
    "vllm.model_executor.layers.fused_moe.modular_kernel",
    "FusedMoEExperts._supports_bias",
    allow_missing=True,
    reason="PPU expert classes declare whether their GEMM supports expert bias.",
    affected_versions=">=0.33.0,<0.34.0",
    remove_when="FusedMoEExperts exposes a bias capability method.",
)
@staticmethod
def _supports_bias():
    return True


@patch(
    "vllm.model_executor.layers.fused_moe.modular_kernel",
    "FusedMoEExperts.is_supported_config",
    reason="Explicit and automatic DeepGEMM selection must reject unsupported bias.",
    affected_versions=">=0.33.0,<0.34.0",
    remove_when="FusedMoEExperts checks its bias capability during configuration selection.",
)
def is_supported_config(cls, moe_config, weight_key, activation_key, activation_format):
    supported, reason = _upstream_supported(
        cls, moe_config, weight_key, activation_key, activation_format
    )
    if supported and moe_config.has_bias and not cls._supports_bias():
        return False, "kernel does not support bias"
    return supported, reason
