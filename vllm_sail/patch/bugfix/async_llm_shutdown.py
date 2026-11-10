# SPDX-License-Identifier: Apache-2.0
"""Keep intentional PPU abort shutdown from reporting an engine failure."""

from __future__ import annotations

import functools
from collections.abc import Callable

from vllm_sail.patch.utils import patch

_MODULE = "vllm.v1.engine.async_llm"
_TARGET = "AsyncLLM.shutdown"
_installed = False


def _wrap_shutdown(original: Callable[..., None]) -> Callable[..., None]:
    @functools.wraps(original)
    def shutdown(self, timeout: float | None = None) -> None:
        from vllm.platforms import current_platform

        # A positive timeout drains requests and still needs the output loop.
        if current_platform.is_ppu() and timeout == 0:
            from vllm.utils.async_utils import cancel_task_threadsafe

            handler = getattr(self, "output_handler", None)
            if handler is not None:
                cancel_task_threadsafe(handler)
        return original(self, timeout=timeout)

    return shutdown


def install() -> None:
    """Register the delegating shutdown fix once, without import-time effects."""
    global _installed
    if _installed:
        return

    from vllm.v1.engine.async_llm import AsyncLLM

    patch(
        _MODULE,
        _TARGET,
        reason=(
            "PPU abort shutdown closes EngineCore while its async output task "
            "can still consume the teardown error, logging EngineDeadError "
            "after otherwise successful requests."
        ),
        affected_versions=">=0.33.0,<0.34.0",
        remove_when=(
            "AsyncLLM cancels its output task before closing EngineCore during "
            "abort shutdown, while preserving drain-mode output processing."
        ),
    )(_wrap_shutdown(AsyncLLM.shutdown))
    _installed = True
