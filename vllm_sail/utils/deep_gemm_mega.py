# SPDX-License-Identifier: Apache-2.0
"""Supported PPU MegaMoE formats and lazy access to DeepGEMM."""

from __future__ import annotations

import functools
import importlib
from typing import NamedTuple


class MegaMoEFormat(NamedTuple):
    kernel: str
    transform: str | None
    activations: tuple[str, ...]
    min_hidden: int
    hidden_multiple: int
    intermediate_multiple: int


MEGA_MOE_FORMATS = {
    "mxf4xmxf4": MegaMoEFormat(
        "fp4_fp4_mega_moe", None, ("swiglu", "situ"), 2048, 512, 64
    ),
    "fp8xmxfp4": MegaMoEFormat(
        "fp8_fp4_mega_moe",
        "transform_weights_for_w4a8_mega_moe",
        ("swiglu", "situ"),
        256,
        256,
        128,
    ),
    "bf16xmxfp4": MegaMoEFormat(
        "bf16_fp4_mega_moe",
        "transform_weights_for_w4a16_mega_moe",
        ("situ",),
        256,
        256,
        128,
    ),
}


@functools.cache
def mega_moe_api(mma_type: str):
    """Resolve only the DeepGEMM API required by the selected mode."""
    fmt = MEGA_MOE_FORMATS[mma_type]
    try:
        module = importlib.import_module("deep_gemm.mega")
    except ImportError as exc:
        raise RuntimeError(
            "PPU MegaMoE requires DeepGEMM-for-sail with deep_gemm.mega support."
        ) from exc
    required = ["get_symm_buffer_for_mega_moe", "mega_moe_pre_dispatch", fmt.kernel]
    if fmt.transform is not None:
        required.append(fmt.transform)
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(
            f"PPU MegaMoE API is missing for {mma_type}: {', '.join(missing)}"
        )
    return module


def check_mega_moe_supported(
    device, hidden_size: int, intermediate_size: int, mma_type: str
):
    from vllm.platforms import current_platform

    from vllm_sail.utils.deep_gemm import is_deep_gemm_supported

    if not (
        current_platform.is_ppu()
        and current_platform.has_device_capability(89, device_id=device.index or 0)
        and is_deep_gemm_supported()
    ):
        raise NotImplementedError("SAIL MegaMoE requires Zhenwu M890 and DeepGEMM.")
    fmt = MEGA_MOE_FORMATS[mma_type]
    if (
        hidden_size < fmt.min_hidden
        or hidden_size % fmt.hidden_multiple
        or intermediate_size <= 0
        or intermediate_size % fmt.intermediate_multiple
    ):
        raise ValueError(
            f"SAIL MegaMoE {mma_type} requires H >= {fmt.min_hidden} divisible by "
            f"{fmt.hidden_multiple} and positive I divisible by {fmt.intermediate_multiple}."
        )
    mega_moe_api(mma_type)
