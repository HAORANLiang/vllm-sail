# SPDX-License-Identifier: Apache-2.0
"""QuantFP8 context and portable PPU group-quantization selection."""

from __future__ import annotations

from vllm.model_executor.layers.quantization import input_quant_fp8 as _quant
from vllm.platforms import current_platform

from vllm_sail import envs as ppu_envs
from vllm_sail.patch.utils import patch

_META = dict(
    reason="PPU QuantFP8 consumes fused residual RMSNorm context and optional optimized group quantization.",
    affected_versions=">=0.33.0,<0.34.0",
    remove_when="Upstream QuantFP8 exposes a fused norm and platform group-quant hook.",
)
_original = _quant.QuantFP8.forward_cuda


@patch(_quant.__name__, "QuantFP8.set_fused_rmsnorm", allow_missing=True, **_META)
def set_fused_rmsnorm(self, residual, weight, epsilon):
    self._fused_rmsnorm = (
        (residual, weight, epsilon)
        if residual is not None and weight is not None
        else None
    )


@patch(_quant.__name__, "QuantFP8.forward_cuda", **_META)
def forward_cuda(self, x, scale=None, scale_ub=None, use_triton=False):
    context = getattr(self, "_fused_rmsnorm", None)
    self._fused_rmsnorm = None
    if context is not None:
        from vllm_sail.model_executor.layers.quantization.utils.fused_add_rmsnorm_quant import (
            fused_add_rmsnorm_group_quant,
        )

        residual, weight, epsilon = context
        q, scales, _ = fused_add_rmsnorm_group_quant(
            x,
            residual,
            weight,
            epsilon,
            group_size=self.group_size,
            column_major_scales=self.column_major_scales,
        )
        return q, scales
    if (
        current_platform.is_ppu()
        and ppu_envs.VLLM_SAIL_USE_OPT_TOKEN_GROUP_QUANT
        and self.is_group_quant
        and not self.static
        and not self.use_ue8m0
        and x.ndim in (2, 3)
        and x.is_contiguous()
    ):
        assert scale is None, "Dynamic group quantization does not use scale"
        from vllm_sail.model_executor.layers.quantization.utils.group_quant import (
            per_token_group_quant_fp8_ppu_opt,
        )

        return per_token_group_quant_fp8_ppu_opt(
            x,
            group_size=self.group_size,
            column_major_scales=self.column_major_scales,
            tma_aligned_scales=self.tma_aligned_scales,
            dtype=_quant._FP8_DTYPE,
            use_ue8m0=self.use_ue8m0,
        )
    return _original(self, x, scale, scale_ub, use_triton)
