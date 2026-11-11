# SPDX-License-Identifier: Apache-2.0
"""Lazy access to the SAIL DeepGEMM MegaMoE API."""

from __future__ import annotations

import functools
import importlib


@functools.cache
def mega_moe_api():
    """Resolve the W4A4 API only when MegaMoE is explicitly selected."""
    try:
        module = importlib.import_module("deep_gemm.mega")
    except ImportError as exc:
        raise RuntimeError(
            "PPU MegaMoE requires DeepGEMM-for-sail with deep_gemm.mega support."
        ) from exc
    required = (
        "get_symm_buffer_for_mega_moe",
        "mega_moe_pre_dispatch",
        "fp4_fp4_mega_moe",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(f"PPU MegaMoE API is missing: {', '.join(missing)}")
    return module


def check_mega_moe_supported(device, hidden_size: int, intermediate_size: int):
    from vllm.platforms import current_platform

    from vllm_sail.utils.deep_gemm import is_deep_gemm_supported

    if not (
        current_platform.is_ppu()
        and current_platform.has_device_capability(89, device_id=device.index or 0)
        and is_deep_gemm_supported()
    ):
        raise NotImplementedError("SAIL MegaMoE requires Zhenwu M890 and DeepGEMM.")
    if (
        hidden_size < 2048
        or hidden_size % 512
        or intermediate_size <= 0
        or intermediate_size % 64
    ):
        raise ValueError(
            "SAIL MegaMoE requires H >= 2048 divisible by 512 and I divisible by 64."
        )
    mega_moe_api()
