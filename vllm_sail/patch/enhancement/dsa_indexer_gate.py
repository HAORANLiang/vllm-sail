# SPDX-License-Identifier: Apache-2.0
"""Single platform gate for the two patches that own DSA indexer logits.

``attention.sparse_attn_indexer`` pins the PPU logits dtype on the
``fp8_fp4_*`` scoring helpers unless ``models.kimi_k3_dsa_indexer`` owns those
names with its int8 scoring path. Both patches must reach the *same* verdict:
the int8 path is downgraded when the installed DeepGEMM exports no
``int8_mqa_logits``, and the FP8 path -- the one the pin serves -- becomes live
again. Duplicating the platform test instead skipped both, which left
upstream's pybind entry points installed on a target that cannot execute them.

``enhancement/__init__`` imports ``attention`` before ``models``, so this leaf
module must not import either of them.
"""

from __future__ import annotations

import functools

from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_sail.utils import deep_gemm as _sail_deep_gemm

logger = init_logger(__name__)


def uses_software_e4m3() -> bool:
    """Whether this target needs the software E4M3FN encoder.

    Triton's CUDA backend rejects ``tl.float8e4nv`` below capability 8.9 and the
    target has no f32->e4m3 cvt instruction.
    """
    return not current_platform.has_device_capability((8, 9))


def wants_int8_indexer() -> bool:
    """Whether the platform alone selects the int8 indexer cache and scoring."""
    return current_platform.is_ppu() and not current_platform.supports_fp8()


@functools.cache
def int8_indexer_active() -> bool:
    """Whether the int8 indexer actually owns the cache and the scoring names.

    The int8 logits live behind ``vllm_sail.utils.deep_gemm``, which imports
    DeepGEMM lazily and only forwards what the installed build exports: an image
    whose DeepGEMM predates the 810E kernels must not select a quantizer whose
    consumers are missing, so that case falls back to the FP8 path.
    """
    if not wants_int8_indexer():
        return False
    if not _sail_deep_gemm.has_int8_mqa_logits():
        logger.warning(
            "Skipping the Kimi K3.1 int8 DSA indexer path: the installed "
            "DeepGEMM does not export int8_mqa_logits/int8_paged_mqa_logits. "
            "The FP8 indexer path and its PPU logits-dtype pin stay active."
        )
        return False
    return True
