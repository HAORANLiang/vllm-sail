# SPDX-License-Identifier: Apache-2.0
"""The Marlin MoE gate — phase-1 decision D2, option (c).

The fork's Marlin support needs in-place edits to upstream CUDA sources
(``csrc/libtorch_stable/quantization/marlin/*``,
``csrc/libtorch_stable/moe/marlin_moe_wna16/*``) that a plugin cannot make.
Until those changes are upstreamed, PPU stays on non-Marlin MoE paths:
``check_moe_marlin_supports_config`` returns False on PPU unless
``VLLM_PPU_ENABLE_MOE_MARLIN`` is explicitly set (default off; declared in
``vllm_sail/envs.py``).

Delegating: upstream decides in every case except PPU-with-the-gate-off.
"""

from __future__ import annotations

from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_moe_marlin_supports_config,
)
from vllm.platforms import current_platform

import vllm_sail.envs as ppu_envs
from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.33.0,<0.34.0"

# Captured before patching so the replacement can delegate to upstream.
_upstream_check = check_moe_marlin_supports_config
# The layout check is also used by PPU DeepGEMM weight preparation.
supports_marlin_layout = _upstream_check


@patch(
    "vllm.model_executor.layers.quantization.utils.marlin_utils",
    "check_moe_marlin_supports_config",
    reason=(
        "The fork's Marlin kernels are in-place edits of upstream .cu/.h "
        "sources that a plugin cannot ship; v1 restricts PPU to non-Marlin "
        "MoE paths (phase-1 spec 1.8, decision (c)). Delegates to upstream "
        "unless PPU has the opt-out env var left at its default (off)."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "the Marlin PPU kernel edits are upstreamed (or rebuilt inside "
        "vllm_sail._moe_C) and VLLM_PPU_ENABLE_MOE_MARLIN defaults on."
    ),
)
def gated_check_moe_marlin_supports_config(*args, **kwargs):
    if current_platform.is_ppu() and not ppu_envs.VLLM_SAIL_ENABLE_MOE_MARLIN:
        return False
    return _upstream_check(*args, **kwargs)


import sys

for _consumer in (
    "vllm.model_executor.layers.fused_moe.oracle.int_wna16",
    "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16",
):
    _module = sys.modules.get(_consumer)
    if (
        _module is not None
        and getattr(_module, "check_moe_marlin_supports_config", None)
        is _upstream_check
    ):
        patch(
            _consumer,
            "check_moe_marlin_supports_config",
            reason="Preloaded WNA16 consumers must observe the PPU Marlin capability gate.",
            affected_versions=_AFFECTED,
            remove_when="WNA16 consumers resolve Marlin capabilities through the provider module.",
        )(gated_check_moe_marlin_supports_config)
