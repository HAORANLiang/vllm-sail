# ruff: noqa: E731, W291, W293, UP037
# SPDX-License-Identifier: Apache-2.0
"""Add the fork's ``forward_ppu`` overrides to the MHC custom ops.

The fork adds ``forward_ppu`` to four ``CustomOp`` subclasses in
``vllm.model_executor/layers/mhc.py`` (``MHCPreOp``, ``MHCPostOp``, ``HCHeadOp``,
``MHCFusedPostPreOp``). The fork methods are a straight
``return self.forward_cuda(...)`` — identical to the plugin's default
``CustomOp.forward_ppu`` installed by ``custom_op_dispatch`` — so they are
effective no-ops. The pre and fused pre methods instead call plugin-owned torch.ops so the
patched tilelang launchers survive upstream import-time registration. The
fork carries them and the plugin mirrors the fork's CustomOp surface; they also
make PPU explicit in torch.compile custom-op dispatch lists.

Installed additively (``allow_missing=True``): upstream has no ``forward_ppu``.
"""

from __future__ import annotations

import torch

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.33.0,<0.34.0"
_MODULE = "vllm.model_executor.layers.mhc"
_REASON = (
    "Fork's PPU CustomOp surface: MHC ops get an explicit forward_ppu that "
    "delegates to forward_cuda, matching the plugin's custom-op dispatch "
    "contract (forward_ppu selected whenever forward_cuda would be)."
)
_REMOVE_WHEN = (
    "upstream adds forward_ppu to these ops, or the MHC ops stop being used on PPU."
)


@patch(
    _MODULE,
    "MHCPreOp.forward_ppu",
    allow_missing=True,
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def mhc_pre_op_forward_ppu(
    self,
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # PPU MODIFICATION: begin
    return torch.ops.vllm.ppu_mhc_pre_tilelang(
        # PPU MODIFICATION: end
        residual, fn, hc_scale, hc_base, rms_eps,
        hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
        sinkhorn_repeat, n_splits, norm_weight, norm_eps,
    )


@patch(
    _MODULE,
    "MHCPostOp.forward_ppu",
    allow_missing=True,
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def mhc_post_op_forward_ppu(
    self,
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return self.forward_cuda(x, residual, post_layer_mix, comb_res_mix)


@patch(
    _MODULE,
    "HCHeadOp.forward_ppu",
    allow_missing=True,
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def hc_head_op_forward_ppu(
    self,
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    return self.forward_cuda(
        hidden_states,
        hc_fn,
        hc_scale,
        hc_base,
        rms_norm_eps,
        hc_eps,
    )


@patch(
    _MODULE,
    "MHCFusedPostPreOp.forward_ppu",
    allow_missing=True,
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def mhc_fused_post_pre_op_forward_ppu(
    self,
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # PPU MODIFICATION: begin
    return torch.ops.vllm.ppu_mhc_fused_post_pre_tilelang(
        # PPU MODIFICATION: end
        x, residual, post_layer_mix, comb_res_mix,
        fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
        n_splits, tile_n, norm_weight, norm_eps,
    )
