# SPDX-License-Identifier: Apache-2.0
"""PPU patch: enable the DeepSeek V4.1 528B MXFP8 KV record on PPU.

Upstream gates the V4.1 record (512B fp8_e4m3 data + 16B UE8M0 per-32-dim
scales = 528 B/token) to SM100 (``is_device_capability_family(100)``); every
other arch keeps the V4 record (584 B/token). The PPU FlashMLA fork's ordinary
sparse kernels support *both* records (flash_mla/flash_mla_interface.py,
"V4.1 FP8+sparse mode ... 528 bytes per token"), and the PR#54 fused
norm-rope decode kernel *only* accepts the 528 B record. So on PPU the record
choice becomes a runtime switch instead of an SM100-only constant:

* ``VLLM_SAIL_PPU_KV_MXFP8`` truthy (explicit opt-in) -> True -> 528 B V4.1
  record, required by the fused norm-rope decode kernel; enable together
  with ``VLLM_SAIL_FUSED_NORM_ROPE_ATTN=1``;
* unset / falsy (default) -> False -> 584 B V4 record, identical to the
  upstream non-SM100 default, so an unconfigured process keeps baseline
  behaviour (also the clean fallback if the PPU-generated
  ``kv_mxfp8=True`` write path fails UT-C0).

The env is read once per process (first layer construction) and cached; the
first read prints the forensic line consumed by the run ledger:

    [VLLM_SAIL] kv_mxfp8=True kv_bytes_per_token=528

All upstream consumers call the function through the module global
(v0.32.0 attention.py:512 layer init, :1217 kv-cache spec alignment), so
patching the module attribute covers every call site; no by-value alias
rebind is needed.
"""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.models.deepseek_v41 import attention as _dsv41_attention
from vllm.platforms import current_platform

import vllm_sail.envs as ppu_envs
from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.32.0,<0.33.0"
_MODULE = "vllm.models.deepseek_v41.attention"

logger = init_logger(__name__)

_upstream_use_v41_mxfp8_kv_record = _dsv41_attention._use_v41_mxfp8_kv_record

_cached: bool | None = None
_logged = False


@patch(
    _MODULE,
    "_use_v41_mxfp8_kv_record",
    reason=(
        "PPU needs the 528B V4.1 MXFP8 KV record for the FlashMLA PR#54 fused "
        "norm-rope decode kernel (which rejects the 584B V4 record), and the "
        "PPU flash_mla fork's ordinary sparse kernels already support 528B. "
        "Upstream restricts the record to SM100, so without this patch the "
        "fused decode path can never run on PPU and the A/B record format "
        "cannot be held equal on both sides."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream selects the V4.1 KV record from the attention backend's "
        "capability instead of hard-gating it to family(100)."
    ),
)
def _use_v41_mxfp8_kv_record() -> bool:
    global _cached, _logged
    if not current_platform.is_ppu():
        return _upstream_use_v41_mxfp8_kv_record()
    if _cached is None:
        # Read the env exactly once per process (envs resolves
        # VLLM_SAIL_PPU_KV_MXFP8 before its VLLM_PPU_* alias; unset -> False).
        _cached = bool(ppu_envs.VLLM_SAIL_PPU_KV_MXFP8)
    if not _logged:
        logger.info(
            "[VLLM_SAIL] kv_mxfp8=%s kv_bytes_per_token=%d",
            _cached,
            528 if _cached else 584,
        )
        _logged = True
    return _cached
