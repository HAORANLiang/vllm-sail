# SPDX-License-Identifier: Apache-2.0
"""Route packed MLA cache reads to software FP8 decoding on PPU 1.0."""

from __future__ import annotations

import torch
from vllm.models.deepseek_v4.common.ops import cache_utils
from vllm.platforms import current_platform

from vllm_sail.models.deepseek_v4.ops.cache import register_ops
from vllm_sail.patch.utils import patch

register_ops()
_upstream = cache_utils.dequantize_and_gather_k_cache


@patch(
    "vllm.models.deepseek_v4.common.ops.cache_utils",
    "dequantize_and_gather_k_cache",
    reason="PPU 1.0 Triton cannot compile the packed cache's hardware FP8 bitcast; use software E4M3 decoding.",
    affected_versions=">=0.33.0,<0.34.0",
    remove_when="vLLM exposes packed MLA cache kernel registration or supports software FP8 decoding on PPU 1.0.",
)
def dequantize_and_gather_k_cache(
    out,
    k_cache,
    seq_lens,
    gather_lens,
    block_table,
    block_size,
    offset,
    use_fnuz=False,
):
    if current_platform.is_ppu() and current_platform.is_device_capability((8, 0)):
        assert not use_fnuz, "PPU packed MLA cache uses E4M3FN, not FNUZ"
        return torch.ops.vllm.ppu_deepseek_v4_dequant_gather(
            out,
            k_cache,
            seq_lens,
            gather_lens,
            block_table,
            block_size,
            offset,
        )
    if current_platform.is_ppu():
        return cache_utils._DEQUANTIZE_AND_GATHER_K_CACHE_KERNEL(
            out,
            k_cache,
            seq_lens,
            gather_lens,
            block_table,
            block_size,
            offset,
            use_fnuz=use_fnuz,
        )
    return _upstream(
        out, k_cache, seq_lens, gather_lens, block_table, block_size, offset, use_fnuz
    )


import sys

for _consumer_name in (
    "vllm.models.deepseek_v4.common.ops",
    "vllm.models.deepseek_v4.nvidia.flashmla",
    "vllm.models.deepseek_v4.amd.rocm",
    "vllm.models.deepseek_v4.xpu.xpu_sparse",
):
    _consumer = sys.modules.get(_consumer_name)
    if (
        _consumer is not None
        and getattr(_consumer, "dequantize_and_gather_k_cache", None) is _upstream
    ):
        patch(
            _consumer_name,
            "dequantize_and_gather_k_cache",
            reason="Preloaded V4 cache consumers must use the PPU gather implementation.",
            affected_versions=">=0.33.0,<0.34.0",
            remove_when="Consumers resolve cache helpers through their provider module.",
        )(dequantize_and_gather_k_cache)
