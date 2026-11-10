# SPDX-License-Identifier: Apache-2.0
"""Preserve PPU MHC prenorm semantics at vLLM's shared GEMM boundary."""

from __future__ import annotations

import torch
from vllm.model_executor.kernels.mhc import tilelang as _tilelang
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_sail.patch.utils import patch

_upstream_prenorm = _tilelang._hc_prenorm_gemm_outputs


@patch(
    "vllm.model_executor.kernels.mhc.tilelang",
    "_hc_prenorm_gemm_outputs",
    reason="PPU MHC uses SAIL DeepGEMM, one split and zeroed accumulators.",
    affected_versions=">=0.33.0,<0.34.0",
    remove_when="The upstream MHC prenorm helper supports platform dispatch.",
)
def _hc_prenorm_gemm_outputs(
    x: torch.Tensor,
    fn: torch.Tensor,
    *,
    hidden_size: int,
    hc_mult: int,
    use_tilelang_fallback: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not current_platform.is_ppu():
        return _upstream_prenorm(
            x,
            fn,
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            use_tilelang_fallback=use_tilelang_fallback,
        )
    from vllm_sail.utils.deep_gemm import (
        is_deep_gemm_supported,
        tf32_hc_prenorm_gemm,
    )

    # SAIL DeepGEMM needs initialized accumulators and a single split.
    out = torch.zeros(1, x.shape[0], fn.shape[0], dtype=torch.float32, device=x.device)
    sqrsum = torch.zeros(1, x.shape[0], dtype=torch.float32, device=x.device)
    if is_deep_gemm_supported() or not use_tilelang_fallback:
        tf32_hc_prenorm_gemm(x, fn, out, sqrsum, 1)
    else:
        from vllm.model_executor.kernels.mhc.tilelang_kernels import (
            _HC_PRENORM_GEMM_TILELANG_KERNEL,
        )

        _HC_PRENORM_GEMM_TILELANG_KERNEL(x, fn, out, sqrsum, hidden_size, hc_mult)
    return out, sqrsum


# Keep the SAIL CustomOp entry points. Upstream launchers (including previously
# captured aliases) now resolve the patched helper through their live globals.
for _name in ("mhc_pre_tilelang", "mhc_fused_post_pre_tilelang"):
    direct_register_custom_op(
        op_name=f"ppu_{_name}",
        op_func=getattr(_tilelang, _name),
        mutates_args=[],
        fake_impl=getattr(_tilelang, f"_{_name}_fake"),
    )
