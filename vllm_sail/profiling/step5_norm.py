# SPDX-License-Identifier: Apache-2.0
"""Standalone opt-in instrumentation of Step5 RMSNorm, including MTP's entries."""

from __future__ import annotations

import functools
from contextlib import contextmanager, nullcontext

from vllm.logger import init_logger

from vllm_sail import envs
from vllm_sail.patch.utils import PATCH_MARKER, patch
from vllm_sail.profiling.nvtx import range_pop, range_push
from vllm_sail.step5_norm_cases import make_label

logger = init_logger(__name__)
_TARGET = "vllm.models.step5.nvidia.layernorm"


def _capture(x) -> bool:
    if not x.is_cuda:
        return False
    try:
        import torch

        return bool(torch.cuda.is_current_stream_capturing())
    except (AttributeError, RuntimeError):
        return False


@contextmanager
def case_range(label):
    """Also usable explicitly around graph replay; push/pop stay paired on errors."""
    try:
        from torch.profiler import record_function
    except ImportError:
        record_function = None
    range_push(label)
    try:
        with record_function(label) if record_function else nullcontext():
            yield
    finally:
        range_pop()


def install() -> bool:
    """Install Norm only; never imports PPU native or other profiling modules."""
    if not envs.VLLM_SAIL_NVTX_PROFILE:
        return False
    try:
        from vllm.models.step5.nvidia.layernorm import Step5RMSNorm

        original = Step5RMSNorm.forward
        if getattr(original, "_sail_step5_norm_profile", False):
            return True
        if getattr(original, PATCH_MARKER, None) is not None:
            logger.warning(
                "Step5RMSNorm.forward is already taken over by another patch; skipping Norm instrumentation"
            )
            return False

        @patch(
            _TARGET,
            "Step5RMSNorm.forward",
            reason="Opt-in Step5 Norm metadata ranges for reproducible cross-platform cases.",
            affected_versions=">=0.32.0,<0.33.0",
            remove_when="Upstream emits versioned Step5 Norm input metadata ranges.",
        )
        @functools.wraps(original)
        def forward(
            self, x, residual=None, output=None, fp16_out=False, *, out_dtype=None
        ):
            # Turning the flag off after install stops the markers; processes that start
            # with it off never install the wrapper.
            if not envs.VLLM_SAIL_NVTX_PROFILE:
                return original(
                    self, x, residual, output, fp16_out, out_dtype=out_dtype
                )
            try:
                label = make_label(
                    x,
                    self.weight,
                    self.variance_epsilon,
                    self.zero_centered,
                    residual,
                    output,
                    out_dtype,
                    _capture(x),
                )
            except (AttributeError, IndexError, TypeError, ValueError):
                # If metadata cannot be queried, keep the original checks and errors.
                return original(
                    self, x, residual, output, fp16_out, out_dtype=out_dtype
                )
            with case_range(label):
                return original(
                    self, x, residual, output, fp16_out, out_dtype=out_dtype
                )

        forward._sail_step5_norm_profile = True
        return True
    except (ImportError, AttributeError) as exc:
        logger.warning("Step5 Norm instrumentation unavailable: %s", exc)
        return False
