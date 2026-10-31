# SPDX-License-Identifier: Apache-2.0
"""SAIL MegaMHC capability checks for the V4.1 shifted boundary.

Adapted from the SGLang-for-sail MegaMHC integration. The PPU library currently
supports the V4.1 Flash BF16 boundary (H=5120, HC=4), including decode. Preserve
the preceding sublayer's pre-mix; computing a fresh pre-mix before collapsing
the stream would change V4.1 semantics.
"""

from __future__ import annotations

import functools
import importlib

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

import vllm_sail.envs as envs
from vllm_sail.utils.deep_gemm import is_deep_gemm_supported

logger = init_logger(__name__)


@functools.cache
def mega_mhc_kernel():
    try:
        module = importlib.import_module("deep_gemm")
    except ImportError:
        kernel = None
    else:
        kernel = getattr(module, "mega_mhc", None)
    if not callable(kernel):
        logger.warning_once(
            "SAIL MegaMHC requested but unavailable; using delayed mHC."
        )
        return None
    return kernel


def is_supported(hidden_size: int, hc_mult: int) -> bool:
    return (
        envs.VLLM_SAIL_DEEPSEEK_V41_MEGA_MHC
        and current_platform.is_ppu()
        and current_platform.has_device_capability(89)
        and hidden_size == 5120
        and hc_mult == 4
        and is_deep_gemm_supported()
        and mega_mhc_kernel() is not None
    )


def can_use(x, residual, pre_mix, norm_weight, capture_aux: bool) -> bool:
    if (
        capture_aux
        or pre_mix is None
        or norm_weight is None
        or x.ndim != 2
        or residual.ndim != 3
        or not 0 < x.shape[0] <= 1 << 20
        or not is_supported(x.shape[1], residual.shape[1])
    ):
        return False
    rows, hidden = x.shape
    tensors = (
        (x, (rows, hidden), torch.bfloat16),
        (residual, (rows, 4, hidden), torch.bfloat16),
        (pre_mix, (rows, 4), torch.float32),
        (norm_weight, (hidden,), torch.bfloat16),
    )
    return x.is_cuda and all(
        tensor.shape == shape
        and tensor.dtype == dtype
        and tensor.device == x.device
        and tensor.is_contiguous()
        and tensor.data_ptr() % 16 == 0
        for tensor, shape, dtype in tensors
    )
