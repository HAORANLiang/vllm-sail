# SPDX-License-Identifier: Apache-2.0
"""Filter unsupported warmup signatures before shared async compilation."""

from __future__ import annotations

from vllm_sail.patch.utils import patch


def install() -> None:
    from vllm.logger import init_logger
    from vllm.model_executor.warmup.jit_warmup_triton_helper import (
        VllmTritonJitKernel,
    )

    from vllm_sail.patch.utils import PATCH_MARKER

    original = VllmTritonJitKernel.compile_many
    target = (
        "vllm.model_executor.warmup.jit_warmup_triton_helper."
        "VllmTritonJitKernel.compile_many"
    )
    if target in getattr(original, PATCH_MARKER, {}):
        return
    logger = init_logger(__name__)

    @patch(
        "vllm.model_executor.warmup.jit_warmup_triton_helper",
        "VllmTritonJitKernel.compile_many",
        reason=(
            "On PPU 810E, E4M3 input signatures cannot compile even when kernel "
            "source never names float8e4nv. Filter those keys before submission "
            "so the registry retains its shared asynchronous compiler pool."
        ),
        affected_versions=">=0.32.0,<0.33.0",
        remove_when=(
            "The upstream Triton warmup interface filters compile keys by "
            "input dtype support before submitting asynchronous compilation."
        ),
    )
    def compile_many(self, compile_keys) -> None:
        from vllm.platforms import current_platform

        if not current_platform.is_ppu() or current_platform.has_device_capability(89):
            return original(self, compile_keys)

        import torch

        supported = []
        skipped = 0
        for key in compile_keys:
            inputs = self.warmup_inputs(key)
            if any(
                value is torch.float8_e4m3fn
                or getattr(value, "dtype", None) is torch.float8_e4m3fn
                for value in inputs.values()
            ):
                skipped += 1
            else:
                supported.append(key)
        if skipped:
            logger.warning(
                "Skipping %d JIT warmup keys for %s: this PPU architecture "
                "cannot compile fp8e4nv input signatures.",
                skipped,
                type(self).__name__,
            )
        if supported:
            return original(self, supported)
