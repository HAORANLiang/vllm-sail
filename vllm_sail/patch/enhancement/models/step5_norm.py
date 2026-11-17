# SPDX-License-Identifier: Apache-2.0
"""Keep Step5 normalization off CuTeDSL on PPU.

``apply_step5_rms_norm`` and ``apply_step5_add_rms_norm`` pick the CuTeDSL
kernel in ``ops/rms_norm.py`` from ``Tensor.is_cuda`` alone, bypassing
``has_cutedsl``. PPU tensors report ``is_cuda`` but CuTeDSL is unsupported, so
supported AddRMS inputs on PPU default to the out-of-place fused Triton kernel
and unsupported inputs keep upstream's PyTorch fallback. The layers and
the MTP head look the functions up in the module at call time, so replacing the
module attributes is enough.
"""

from __future__ import annotations

import math

import torch
from vllm.models.step5.nvidia import layernorm as _layernorm
from vllm.platforms import current_platform

from vllm_sail.patch.utils import patch

_TARGET = "vllm.models.step5.nvidia.layernorm"
_REASON = "Step5 checks Tensor.is_cuda directly and selects unsupported CuTeDSL on PPU."
_AFFECTED = ">=0.32.0,<0.33.0"
_REMOVE_WHEN = "Step5 normalization selects kernels through platform capabilities."

_upstream_apply_step5_rms_norm = _layernorm.apply_step5_rms_norm
_upstream_apply_step5_add_rms_norm = _layernorm.apply_step5_add_rms_norm


@patch(
    _TARGET,
    "apply_step5_rms_norm",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def apply_step5_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    out: torch.Tensor | None = None,
    zero_centered: bool = False,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not current_platform.is_ppu():
        return _upstream_apply_step5_rms_norm(
            x, weight, variance_epsilon, out, zero_centered, out_dtype
        )
    # PPU MODIFICATION: begin — upstream's non-CUDA fallback, verbatim.
    out_dtype = out_dtype or x.dtype
    if out is not None and (out.shape != x.shape or out.dtype != out_dtype):
        raise ValueError("Step5 RMSNorm output buffer must match shape and dtype")
    result = _layernorm._step5_rms_norm(
        x, weight, variance_epsilon, zero_centered, out_dtype
    )
    if out is None:
        return result
    out.copy_(result)
    return out
    # PPU MODIFICATION: end


@patch(
    _TARGET,
    "apply_step5_add_rms_norm",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def apply_step5_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    zero_centered: bool = False,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not current_platform.is_ppu():
        return _upstream_apply_step5_add_rms_norm(
            x, residual, weight, variance_epsilon, zero_centered, out_dtype
        )
    out_dtype = out_dtype or x.dtype
    # Fuse supported AddRMS by default; unsupported metadata or missing
    # dependencies keep the original path.
    if math.isfinite(variance_epsilon) and variance_epsilon > 0:
        try:
            from vllm_sail.models.step5.ops import rms_norm
        except ImportError:
            rms_norm = None
        if (rms_norm is not None
                and rms_norm.supports_fused_add_rms_norm(x, residual, weight, out_dtype)):
            return rms_norm.fused_add_rms_norm(
                x, residual, weight, variance_epsilon, zero_centered, out_dtype
            )
    # PPU MODIFICATION: begin — upstream's non-CUDA fallback, verbatim. The sum
    # keeps the residual's dtype, so an FP32 residual is not rounded first.
    residual_out = x + residual
    output = _layernorm._step5_rms_norm(
        residual_out, weight, variance_epsilon, zero_centered, out_dtype
    )
    return output, residual_out
    # PPU MODIFICATION: end
