# SPDX-License-Identifier: Apache-2.0
"""Per-token init/local block forcing for the SAIL top-k.

``sparse_topk_select`` can force blocks from the start/end of the sequence, but
its ``num_valid_pages`` is a batch-wide scalar while vLLM's rows have different
causal page counts. Writing the forced tiles' scores here, per token, keeps the
force-inclusion correct for every row (SAIL's OnlyScore pass never writes the
init blocks, and the local window would otherwise depend on the batch-wide
padding width).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_INIT_SCORE = 1.0e30
_LOCAL_SCORE = 1.0e29


@triton.jit
def _force_init_local_scores_kernel(
    score_ptr,  # [heads, tiles, tokens] contiguous
    pages_ptr,  # int32 [tokens] causal page count
    stride_h,
    stride_k,
    stride_t,
    init_blocks,
    local_blocks,
    INIT_SCORE: tl.constexpr,
    LOCAL_SCORE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pages = tl.load(pages_ptr + pid_t)
    base = score_ptr + pid_h * stride_h + pid_t * stride_t
    k = tl.arange(0, BLOCK)
    n_init = tl.minimum(init_blocks, pages)
    tl.store(base + k * stride_k, INIT_SCORE, mask=k < n_init)
    lo = tl.maximum(pages - local_blocks, 0)
    local = lo + k
    tl.store(base + local * stride_k, LOCAL_SCORE, mask=local < pages)


def force_init_local_scores(
    max_score: torch.Tensor,
    num_valid_pages: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
) -> None:
    if init_blocks <= 0 and local_blocks <= 0:
        return
    assert max_score.is_contiguous(), "max_score must be contiguous"
    assert num_valid_pages.numel() >= max_score.shape[2], (
        f"num_valid_pages ({num_valid_pages.numel()}) < tokens ({max_score.shape[2]})"
    )
    assert num_valid_pages.device == max_score.device, (
        f"pages on {num_valid_pages.device}, scores on {max_score.device}"
    )
    stride_h, stride_k, stride_t = max_score.stride()
    heads, _, tokens = max_score.shape
    block = triton.next_power_of_2(max(init_blocks, local_blocks, 1))
    _force_init_local_scores_kernel[(tokens, heads)](
        max_score,
        num_valid_pages,
        stride_h,
        stride_k,
        stride_t,
        init_blocks,
        local_blocks,
        INIT_SCORE=_INIT_SCORE,
        LOCAL_SCORE=_LOCAL_SCORE,
        BLOCK=block,
    )
