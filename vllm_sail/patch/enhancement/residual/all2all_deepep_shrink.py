# SPDX-License-Identifier: Apache-2.0
"""Drop ``enable_shrink`` from the DeepEP low-latency all2all buffer kwargs.

The fork comments out ``enable_shrink=self.support_fault_tolerance`` in
``DeepEPLLAll2AllManager._make_all2all_kwargs``
(``vllm/distributed/device_communicators/all2all.py``) with a FIXME: the PPU
``deep_ep`` build does not support ``enable_shrink`` yet, so passing it breaks
low-latency buffer creation on PPU.

Delegating patch: upstream still builds the whole kwargs dict; we only remove
the one key, and only on PPU.
"""

from __future__ import annotations

from typing import Any

from vllm_sail.patch.utils import patch

_MODULE = "vllm.distributed.device_communicators.all2all"
_TARGET = "DeepEPLLAll2AllManager._make_all2all_kwargs"
REASON = (
    "PPU's deep_ep build does not support enable_shrink yet (fork FIXME), so "
    "the DeepEP low-latency buffer kwargs must not carry it. Upstream passes "
    "enable_shrink=self.support_fault_tolerance unconditionally."
)
AFFECTED_VERSIONS = ">=0.33.0,<0.34.0"
REMOVE_WHEN = (
    "the PPU deep_ep build implements enable_shrink (shrinkable buffers for "
    "fault tolerance), at which point the upstream kwarg can flow through "
    "unchanged."
)

#: (target, reason, affected_versions, remove_when) for every patch installed
#: by this module. Read by the metadata-shape unit tests.
METADATA = ((f"{_MODULE}.{_TARGET}", REASON, AFFECTED_VERSIONS, REMOVE_WHEN),)

_upstream_make_all2all_kwargs = None
_installed = False


def install() -> None:
    """Apply the patch. Requires vLLM to be importable; idempotent."""
    global _upstream_make_all2all_kwargs, _installed
    if _installed:
        return

    from vllm.distributed.device_communicators import all2all as _all2all

    if _upstream_make_all2all_kwargs is None:
        _upstream_make_all2all_kwargs = (
            _all2all.DeepEPLLAll2AllManager._make_all2all_kwargs
        )

    @patch(
        _MODULE,
        _TARGET,
        reason=REASON,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN,
    )
    def _make_all2all_kwargs(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> dict[Any, Any]:
        buffer_kwargs = _upstream_make_all2all_kwargs(self, *args, **kwargs)
        from vllm.platforms import current_platform

        if current_platform.is_ppu():
            buffer_kwargs.pop("enable_shrink", None)
        return buffer_kwargs

    _installed = True
