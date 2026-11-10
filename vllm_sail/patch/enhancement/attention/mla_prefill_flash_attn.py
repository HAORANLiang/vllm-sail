# SPDX-License-Identifier: Apache-2.0
"""PPU patch for ``vllm.v1.attention.backends.mla.prefill.flash_attn``.

The fork extends the "no V padding needed" condition in
``FlashAttnPrefillBackend.__init__`` with ``or (self.vllm_flash_attn_version == 3
and current_platform.is_ppu())`` — FA3 on PPU natively handles differing QK/V
head dims, like FA3 on Hopper and FA4. The attribute is computed inside
``__init__`` and never recomputed, so this delegates to the upstream ``__init__``
and then fixes ``requires_v_padding`` up on PPU.

The target module resolves the FlashAttention version at import time. On hosts
where that chain cannot run (e.g. a bare test environment whose platform
selection refuses to pick an FA version) there is also nothing for the patch to
act on, so this module skips itself with a warning instead of taking down the
rest of the plugin import. On a real PPU host the import always succeeds.
"""

from __future__ import annotations

import importlib

from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_sail.patch.utils import patch

logger = init_logger(__name__)

_AFFECTED = ">=0.33.0,<0.34.0"
_MODULE = "vllm.v1.attention.backends.mla.prefill.flash_attn"

try:
    _prefill_flash_attn = importlib.import_module(_MODULE)
except ImportError as exc:
    logger.warning(
        "vllm-sail: skipping the %s patch because the target module did not "
        "import (%s). This is expected in test environments without a "
        "GPU platform; on a PPU host the import succeeds.",
        _MODULE,
        exc,
    )
else:
    _upstream_init = _prefill_flash_attn.FlashAttnPrefillBackend.__init__

    def _init(self, *args, **kwargs) -> None:
        _upstream_init(self, *args, **kwargs)
        if current_platform.is_ppu() and self.vllm_flash_attn_version == 3:
            self.requires_v_padding = False

    patch(
        _MODULE,
        "FlashAttnPrefillBackend.__init__",
        reason=(
            "FA3 on PPU natively handles differing QK/V head dims, like FA3 on "
            "Hopper and FA4. The fork adds `(vllm_flash_attn_version == 3 and "
            "is_ppu())` to the requires_v_padding condition; this delegates to "
            "the upstream __init__ and clears requires_v_padding for that case."
        ),
        affected_versions=_AFFECTED,
        remove_when=(
            "upstream's requires_v_padding condition covers PPU's FA3, or PPU "
            "stops using FA3 for MLA prefill."
        ),
    )(_init)
