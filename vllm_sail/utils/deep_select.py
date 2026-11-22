# SPDX-License-Identifier: Apache-2.0
"""Precompiled DeepSelect backend for the SAIL indexer; keeps BF16/FP32 precision.

Follows the wheel API of sglang-for-sail b76ad8286bde; no CUDA JIT backend is
imported. FP32 execution requires external library support; dtype errors from
older libraries propagate unchanged.
"""

import functools

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_sail import envs

logger = init_logger(__name__)


@functools.cache
def _wheel():
    try:
        from deep_select import interface
    except ModuleNotFoundError as exc:
        if exc.name in ("deep_select", "deep_select.interface"):
            return None
        raise
    if not callable(getattr(interface, "topk", None)) or not callable(
        getattr(interface, "get_stride_requirement", None)
    ):
        return None
    return interface


def is_deep_select_available() -> bool:
    return (
        current_platform.is_ppu()
        and current_platform.has_device_capability(89)
        and envs.VLLM_SAIL_USE_DEEPSELECT
        and _wheel() is not None
    )


def can_use_deep_select(logits: torch.Tensor, topk: int) -> bool:
    return (
        logits.dtype in (torch.bfloat16, torch.float32)
        and logits.ndim == 2
        and 0 < topk <= 4096
        and 0 < logits.shape[1] < 2**23
        and is_deep_select_available()
    )


def deep_select_topk(
    logits: torch.Tensor,
    topk: int,
    end: torch.Tensor | None = None,
    output_idx: torch.Tensor | None = None,
    indices_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Write local, -1-padded indices; accept arbitrary caller row strides.

    Rows already start at their query's first K column. Page-table conversion
    remains with vLLM's attention backend, after this local-index selection.
    """
    if logits.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("SAIL DeepSelect requires BF16 or FP32 logits")
    if not can_use_deep_select(logits, topk):
        raise RuntimeError(
            "SAIL DeepSelect wheel, device or top-k shape is unsupported"
        )
    rows, width = logits.shape
    if indices_dtype is None:
        indices_dtype = output_idx.dtype if output_idx is not None else torch.int32
    if indices_dtype not in (torch.int32, torch.int64):
        raise ValueError("DeepSelect indices must have dtype int32 or int64")
    if output_idx is not None and (
        output_idx.dtype != indices_dtype or output_idx.shape != (rows, topk)
    ):
        raise ValueError("DeepSelect output must match [rows, topk] and indices_dtype")
    if end is not None and end.numel() != rows:
        raise ValueError("DeepSelect needs one exclusive end per query row")
    interface = _wheel()
    run_bytes, _ = interface.get_stride_requirement()
    # The wheel requires run-aligned row strides, including narrow views.
    if (
        logits.stride(1) != 1
        or logits.stride(0) * logits.element_size() % run_bytes
        or logits.data_ptr() % 16
    ):
        step = run_bytes // logits.element_size()
        padded = (width + step - 1) // step * step
        scores = torch.empty((rows, padded), dtype=logits.dtype, device=logits.device)
        scores[:, :width].copy_(logits)
        logits = scores[:, :width]
    ends = (
        end.reshape(-1).to(torch.int32).clamp(0, width).contiguous()
        if end is not None
        else None
    )
    # Allocate through the wheel: a caller's k/stride need not meet its 32B
    # output alignment. In particular, preserve k instead of rounding it up.
    _, indices = interface.topk(
        logits,
        topk,
        sorted=False,
        end=ends,
        indices_type=indices_dtype,
        sorted_index=False,
        output_idx=None,
        output_idx_offset=None,
        idx_oob_fill_value=-1,
        return_value=False,
        abort_when_nan_found=False,
    )
    if output_idx is not None:
        output_idx.copy_(indices)
        return output_idx
    return indices


def warn_deep_select_fallback() -> None:
    logger.info_once(
        "DeepSelect fallback: using the existing top-k without score rounding; "
        "check the wheel, dtype, device, "
        "shape and VLLM_SAIL_USE_DEEPSELECT."
    )
