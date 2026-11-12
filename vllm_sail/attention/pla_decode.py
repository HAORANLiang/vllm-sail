# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU SAIL CUDA PLA kernel integration for GDN decode.

Mirrors SGLang's ``SGLANG_SAIL_PLA_CUDA`` mechanism (sglang commit
b7d85d39bd9e3a4165c068a15314629647e8b393, "feat(pla): support ppu cuda pla
kernel"): when enabled, the GDN decode PLA computations are routed from the
community Triton kernels to the PPU SAIL CUDA kernels provided by the
external ``pla`` package (Fused Sigmoid Gating Delta Rule CUDA kernels):

* ``fused_sigmoid_gating_delta_rule_forward_k_last``
* ``fused_sigmoid_gating_delta_rule_forward_k_last_packed``

The routing is controlled by ``VLLM_GDN_DECODE_KERNEL``. Explicit ``pla``
selects these kernels, ``cuda``/``triton`` bypass them, and ``auto`` retains
``VLLM_SAIL_USE_PLA`` as its compatibility preference. It is effective when:

* running on the PPU platform (``current_platform.is_ppu()``);
* the ``pla`` package is importable;
* the per-call tensor constraints of the CUDA kernels are satisfied
  (checked at the call sites): float32 ssm state pool, int32 state
  indices, head dims K == V == 128, and -- because vLLM reserves slot 0
  as NULL_BLOCK_ID (unlike sglang's PAD_SLOT_ID = -1) -- every call passes
  ``is_sglang=False``.

An explicit ``pla`` request raises on a missing dependency; ``auto`` permits
the GDN resolver to select another backend. Prefill and KDA have separate
resolvers and retain their existing legacy switch behavior.
"""

import threading
from collections.abc import Callable
from typing import Any

from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_sail import envs
from vllm_sail.attention.gdn_decode import pla_decode_enabled

logger = init_logger(__name__)

_lock = threading.Lock()
_resolved: bool = False
_k_last_fn: Callable[..., Any] | None = None
_k_last_packed_fn: Callable[..., Any] | None = None


def _resolve_sail_cuda_pla() -> None:
    """Resolve the PPU SAIL CUDA PLA kernels from the ``pla`` package once.

    Thread-safe: guarded by a module-level lock so concurrent calls from
    multiple threads resolve exactly once.

    Raises ``ImportError`` for an explicit PLA request with a missing package.
    """
    global _resolved, _k_last_fn, _k_last_packed_fn
    if not current_platform.is_ppu() or not pla_decode_enabled():
        return
    # Fast path: after the first resolution, skip the lock entirely.
    if _resolved:
        return
    with _lock:
        if _resolved:
            return

        if not pla_decode_enabled():
            return
        if not current_platform.is_ppu():
            return

        try:
            from pla.decode import (
                fused_sigmoid_gating_delta_rule_forward_k_last,
                fused_sigmoid_gating_delta_rule_forward_k_last_packed,
            )
        except (ImportError, OSError) as exc:
            if envs.VLLM_GDN_DECODE_KERNEL == "auto":
                # Pass a string, not the exception: warning_once is lru_cache'd,
                # so an exception key would pin its traceback (and every caller
                # frame's locals) for the life of the cache. Resolution is
                # retried per layer/step, so each attempt would add an entry
                # and leak that step's tensors until the device runs out.
                logger.warning_once(
                    "PLA decode is unavailable; GDN auto will use another backend: %s",
                    str(exc),
                )
                return
            raise ImportError(
                "VLLM_GDN_DECODE_KERNEL=pla but the PPU `pla` decode package "
                f"cannot load: {exc}. Install a compatible PLA library or select "
                "VLLM_GDN_DECODE_KERNEL=triton."
            ) from exc

        _k_last_fn = fused_sigmoid_gating_delta_rule_forward_k_last
        _k_last_packed_fn = fused_sigmoid_gating_delta_rule_forward_k_last_packed
        _resolved = True
        logger.info_once(
            "GDN PLA decode kernels are available (k_last / k_last_packed)."
        )


def get_sail_cuda_pla_k_last() -> Callable[..., Any] | None:
    """Return ``fused_sigmoid_gating_delta_rule_forward_k_last`` if usable."""
    if not current_platform.is_ppu() or not pla_decode_enabled():
        return None
    _resolve_sail_cuda_pla()
    return _k_last_fn


def get_sail_cuda_pla_k_last_packed() -> Callable[..., Any] | None:
    """Return ``fused_sigmoid_gating_delta_rule_forward_k_last_packed``."""
    if not current_platform.is_ppu() or not pla_decode_enabled():
        return None
    _resolve_sail_cuda_pla()
    return _k_last_packed_fn
