# SPDX-License-Identifier: Apache-2.0
"""PPU change to ``vllm.model_executor.models.qwen3_next``.

Ports the fork's +19 hunk: module-level NVTX range push/pop helpers used for
model profiling. The fork resolves them at import time from
``envs.VLLM_PPU_NVTX_PROFILE``; here they resolve lazily on first use (patch
modules must not do work at import time) with identical observable
behaviour: no-op unless ``VLLM_PPU_NVTX_PROFILE`` is set and
``torch.cuda.nvtx`` imports.

Both helpers are installed additively (``allow_missing``): upstream has no
such attributes. They are defined but currently uncalled in the fork file
itself — the call sites live behind the Phase-6 profiling work — so the
remove condition names that hand-off explicitly.
"""

from __future__ import annotations

import vllm_sail.envs as ppu_envs
from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.33.0,<0.34.0"

_nvtx = None  # (range_push, range_pop), resolved lazily on first use


def _noop_push(label) -> None:
    pass


def _noop_pop() -> None:
    pass


def _resolve_nvtx():
    global _nvtx
    if _nvtx is None:
        if ppu_envs.VLLM_SAIL_NVTX_PROFILE:
            try:
                from torch.cuda.nvtx import range_pop as _pop
                from torch.cuda.nvtx import range_push as _push

                _nvtx = (_push, _pop)
            except ImportError:
                _nvtx = (_noop_push, _noop_pop)
        else:
            _nvtx = (_noop_push, _noop_pop)
    return _nvtx


@patch(
    "vllm.model_executor.models.qwen3_next",
    "th_nvtx_range_push",
    allow_missing=True,
    reason=(
        "Ports the fork's NVTX profiling scaffold for qwen3_next; no-op "
        "unless VLLM_PPU_NVTX_PROFILE is set and torch.cuda.nvtx imports."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "the vllm_sail.profiling subpackage (Phase 6) takes over model-level "
        "NVTX instrumentation, or upstream adds NVTX hooks to qwen3_next."
    ),
)
def th_nvtx_range_push(label) -> None:
    _resolve_nvtx()[0](label)


@patch(
    "vllm.model_executor.models.qwen3_next",
    "th_nvtx_range_pop",
    allow_missing=True,
    reason=(
        "Ports the fork's NVTX profiling scaffold for qwen3_next; no-op "
        "unless VLLM_PPU_NVTX_PROFILE is set and torch.cuda.nvtx imports."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "the vllm_sail.profiling subpackage (Phase 6) takes over model-level "
        "NVTX instrumentation, or upstream adds NVTX hooks to qwen3_next."
    ),
)
def th_nvtx_range_pop() -> None:
    _resolve_nvtx()[1]()
