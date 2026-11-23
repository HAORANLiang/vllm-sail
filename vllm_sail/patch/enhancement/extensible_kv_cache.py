# SPDX-License-Identifier: Apache-2.0
"""Keep the extensible KV cache opt-in on PPU.

Upstream settles an unset ``enable_extensible_kv_cache`` to True on CUDA-alike
platforms (VMM-reserved address range, physical pages committed after warmup).
PPU keeps the estimate-based sizing by default; the extensible path is only
used when the user explicitly passes ``--enable-extensible-kv-cache``.
"""

from vllm.config import vllm as vllm_config_module
from vllm.platforms import current_platform

from vllm_sail.patch.utils import patch

_upstream_resolve = vllm_config_module.VllmConfig._resolve_extensible_kv_cache


@patch(
    "vllm.config.vllm",
    "VllmConfig._resolve_extensible_kv_cache",
    reason=(
        "Upstream auto-enables the VMM-based extensible KV cache on CUDA-alike "
        "platforms when the flag is unset; PPU is CUDA-alike but keeps "
        "estimate-based KV cache sizing by default, so an unset flag must "
        "resolve to False before upstream's auto-enable runs. Explicit user "
        "choices still go through upstream validation unchanged."
    ),
    affected_versions=">=0.34.0,<0.35.0",
    remove_when=(
        "upstream settles the extensible KV cache default from a platform "
        "capability query that PPU answers, or PPU's driver VMM path is "
        "validated and the estimate-based default is no longer needed."
    ),
)
def _resolve_extensible_kv_cache(self):
    # PPU MODIFICATION: begin
    if (
        current_platform.is_ppu()
        and self.cache_config.enable_extensible_kv_cache is None
    ):
        self.cache_config.enable_extensible_kv_cache = False
    # PPU MODIFICATION: end
    return _upstream_resolve(self)
