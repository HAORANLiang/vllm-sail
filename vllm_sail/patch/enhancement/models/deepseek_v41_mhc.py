# SPDX-License-Identifier: Apache-2.0
"""Reuse main's shifted MegaMHC framework with SAIL's optional kernel."""

import sys

from vllm.platforms import current_platform

from vllm_sail.models.deepseek_v41 import mega_mhc as sail
from vllm_sail.patch.utils import PATCH_MARKER, patch

_MODULE = "vllm.models.deepseek_v41.nvidia.ops.mega_mhc"
_REASON = "SAIL provides a BF16 shifted MegaMHC kernel on Zhenwu M890; the upstream SM100 gate and vendored loader exclude it."
_AFFECTED = ">=0.32.0,<0.33.0"
_REMOVE_WHEN = "The V4.1 MegaMHC boundary supports platform-specific capability and kernel registration."


@patch(
    _MODULE,
    "is_mega_mhc_supported",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def is_mega_mhc_supported(hidden_size, hc_mult):
    if current_platform.is_ppu():
        return sail.is_supported(hidden_size, hc_mult)
    return getattr(is_mega_mhc_supported, PATCH_MARKER)[
        f"{_MODULE}.is_mega_mhc_supported"
    ](hidden_size, hc_mult)


@patch(
    _MODULE,
    "can_use_mega_mhc",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def can_use_mega_mhc(x, residual, pre_mix, norm_weight, capture_aux):
    if current_platform.is_ppu():
        return sail.can_use(x, residual, pre_mix, norm_weight, capture_aux)
    return getattr(can_use_mega_mhc, PATCH_MARKER)[f"{_MODULE}.can_use_mega_mhc"](
        x, residual, pre_mix, norm_weight, capture_aux
    )


@patch(
    _MODULE,
    "mega_mhc",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def mega_mhc(*args, **kwargs):
    if current_platform.is_ppu():
        kernel = sail.mega_mhc_kernel()
        if kernel is None:
            raise RuntimeError(
                "SAIL MegaMHC is unavailable after capability selection."
            )
        # Let launch errors propagate: never retry after a kernel may have
        # modified outputs. Eager warmup initializes per-stream graph workspace.
        return kernel(*args, **kwargs)
    return getattr(mega_mhc, PATCH_MARKER)[f"{_MODULE}.mega_mhc"](*args, **kwargs)


_consumer_name = "vllm.models.deepseek_v41.nvidia.ops.mhc"
_consumer = sys.modules.get(_consumer_name)
_original = getattr(can_use_mega_mhc, PATCH_MARKER)[f"{_MODULE}.can_use_mega_mhc"]
if _consumer is not None and getattr(_consumer, "can_use_mega_mhc", None) is _original:
    patch(
        _consumer_name,
        "can_use_mega_mhc",
        reason=_REASON,
        affected_versions=_AFFECTED,
        remove_when=_REMOVE_WHEN,
    )(can_use_mega_mhc)
