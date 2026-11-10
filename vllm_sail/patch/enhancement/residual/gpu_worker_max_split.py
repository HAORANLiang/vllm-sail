# SPDX-License-Identifier: Apache-2.0
"""Raise the allocator ``max_split_size_mb`` floor for PPU during model load.

Upstream ``Worker.load_model`` scopes
``self._scoped_allocator_max_split(max_split_size_mb=20)`` — 20 MiB is the
minimum stock PyTorch allows. The fork computes ``32`` instead when
``current_platform.is_ppu()`` because PPU's torch raises the CachingAllocator
minimum for ``max_split_size_mb`` from 20 to 32; passing 20 errors out during
weight loading.

Delegating patch on the context-manager helper rather than a copy of
``load_model``: bumping the value at this single seam covers every caller and
survives upstream edits to ``load_model``'s surrounding context stack.
"""

from __future__ import annotations

import contextlib

from vllm_sail.patch.utils import patch

_MODULE = "vllm.v1.worker.gpu_worker"
_TARGET = "Worker._scoped_allocator_max_split"
#: PPU torch's CachingAllocator minimum for max_split_size_mb.
PPU_MIN_MAX_SPLIT_MB = 32
REASON = (
    "PPU torch raises the CachingAllocator minimum for max_split_size_mb from "
    "20 to 32 MiB; upstream's Worker.load_model requests 20, which PPU's "
    "allocator rejects during weight loading."
)
AFFECTED_VERSIONS = ">=0.33.0,<0.34.0"
REMOVE_WHEN = (
    "PPU torch accepts max_split_size_mb=20 again (its CachingAllocator floor "
    "matches stock PyTorch), or upstream makes the floor platform-queryable."
)

#: (target, reason, affected versions, remove_when) for every patch installed
#: by this module. Read by the metadata-shape unit tests.
METADATA = ((f"{_MODULE}.{_TARGET}", REASON, AFFECTED_VERSIONS, REMOVE_WHEN),)

_upstream_scoped_max_split = None
_installed = False


def install() -> None:
    """Apply the patch. Requires vLLM to be importable; idempotent."""
    global _upstream_scoped_max_split, _installed
    if _installed:
        return

    from vllm.v1.worker import gpu_worker as _gpu_worker

    if _upstream_scoped_max_split is None:
        _upstream_scoped_max_split = _gpu_worker.Worker._scoped_allocator_max_split

    @patch(
        _MODULE,
        _TARGET,
        reason=REASON,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN,
    )
    @contextlib.contextmanager
    def _scoped_allocator_max_split(self, max_split_size_mb: int):
        from vllm.platforms import current_platform

        if current_platform.is_ppu() and max_split_size_mb < PPU_MIN_MAX_SPLIT_MB:
            max_split_size_mb = PPU_MIN_MAX_SPLIT_MB
        with _upstream_scoped_max_split(self, max_split_size_mb):
            yield

    _installed = True
