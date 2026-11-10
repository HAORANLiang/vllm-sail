# SPDX-License-Identifier: Apache-2.0
"""Add the int8 ``QuantKey`` constants PPU's Acext MoE experts need.

Upstream ``quantization/utils/quant_utils.py`` defines a family of module-level
``QuantKey`` constants (``kFp8StaticTensorSym``, ``kNvfp4Quant``, ...) but not the
two int8 variants PPU's Acext path matches on. The in-tree fork adds them to that
module; here we add them from the plugin, additively.

These are data constants rather than behaviour, so they go through
:func:`~vllm_sail.patch.utils.patch_value`. If upstream later defines them
identically, installation becomes a no-op and this file can simply be deleted; if
upstream defines them *differently*, installation raises so the divergence cannot
pass unnoticed.
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
)

from vllm_sail.patch.utils import patch_value

_AFFECTED = ">=0.33.0,<0.34.0"
_REASON = (
    "PPU's Acext fused-MoE experts declare their supported quantization schemes "
    "with these int8 QuantKey constants. Upstream defines the fp8/nvfp4/mxfp4 "
    "constants but not the int8 per-channel-static and per-token-dynamic ones."
)
_REMOVE_WHEN = (
    "upstream defines these constants itself, at which point patch_value becomes "
    "a no-op and this module can be deleted."
)

_MODULE = "vllm.model_executor.layers.quantization.utils.quant_utils"

_static_channel_scale = ScaleDesc(torch.float32, True, GroupShape.PER_CHANNEL)
_dynamic_token_scale = ScaleDesc(torch.float32, False, GroupShape.PER_TOKEN)

_CONSTANTS = {
    "kStaticChannelScale": _static_channel_scale,
    "kDynamicTokenScale": _dynamic_token_scale,
    "kInt8StaticChannelSym": QuantKey(
        torch.int8, _static_channel_scale, symmetric=True
    ),
    "kInt8DynamicTokenSym": QuantKey(
        torch.int8, _dynamic_token_scale, symmetric=True
    ),
}

for _name, _value in _CONSTANTS.items():
    patch_value(
        _MODULE,
        _name,
        _value,
        allow_missing=True,
        reason=_REASON,
        affected_versions=_AFFECTED,
        remove_when=_REMOVE_WHEN,
    )

del _name, _value
