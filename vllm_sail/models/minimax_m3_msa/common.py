# SPDX-License-Identifier: Apache-2.0
"""Layout helpers shared by the SAIL MSA indexer and sparse attend.

The SAIL kernels consume a *compacted* page table (``sum_pages`` entries, one
row per request laid out back to back), while vLLM's block table is a dense
``[requests, max_blocks]`` matrix. They also require per-row ``-1`` padding in
the selected block ids; ``sparse_topk_select`` only takes a scalar
``num_valid_pages``, so the per-token causal clamp is applied here.
"""

from __future__ import annotations

import torch

PAGE_SIZE = 128

_INT32_MAX = torch.iinfo(torch.int32).max


def plan_device(t: torch.Tensor) -> int:
    """Device index to pass to ``fmha_sail_plan``.

    The planner moves host lens to ``torch.device("cuda", device)`` only when
    ``device`` is an int; handed a ``torch.device`` it falls back to ``cuda:0``
    and the kernel then dereferences another GPU's pointers (illegal access as
    soon as the worker is not on GPU 0).
    """
    index = t.device.index
    return index if index is not None else torch.cuda.current_device()


class PlanWorkspacePool:
    """Private ``fmha_sail_plan`` workspace for one metadata builder.

    ``fmha_sail_plan`` slices all of a plan's metadata (per-request lens and
    offsets, page indptr, tile worklists) out of a single process-wide buffer
    per device and only ever grows it. This plugin keeps two plans alive at
    once -- the indexer's OnlyScore plan and the sparse attend plan -- and vLLM
    builds every layer's metadata before the model runs, so the second plan
    overwrites the first one's lengths in place; the indexer then walks its K
    pages with the attend plan's segment lens and leaves its score buffer
    (intermittent illegal memory access). Each builder hands the planner its
    own buffer to keep the plans independent.

    Sizes come from ``allocate_graph_workspace``, the library's own bound for
    ``batch`` rows of ``max_qo_len`` tokens. Fed the step's real batch it
    assumes every row is ``max_qo_len`` long, so one long prefill chunk in a
    decode batch inflates it by up to ``num_reqs`` times, and this memory is
    outside vLLM's profiling. The bound is taken for the fewest
    ``max_qo_len`` rows covering ``num_tokens`` instead: the same tile config
    as the plan, at most twice the tokens it needs. ``_bounds`` memoises it
    so the (throwaway) sizing allocation happens once per key.
    """

    def __init__(self) -> None:
        self._workspace: torch.Tensor | None = None
        self._bounds: dict[tuple[int, int], int] = {}

    def get(
        self,
        *,
        num_tokens: int,
        max_qo_len: int,
        num_qo_heads: int,
        num_kv_heads: int,
        kv_block_num: int,
        output_maxscore: bool,
        use_fp8_kvcache: bool,
        device: int,
    ) -> torch.Tensor:
        rows = -(-max(num_tokens, 1) // max_qo_len)
        key = (rows, max_qo_len)
        need = self._bounds.get(key)
        if need is None:
            from msa_for_sail.api import allocate_graph_workspace

            need = allocate_graph_workspace(
                rows,
                num_qo_heads,
                num_kv_heads=num_kv_heads,
                max_qo_len=max_qo_len,
                page_size=PAGE_SIZE,
                kv_block_num=kv_block_num,
                num_kv_splits=1,
                device=device,
                output_maxscore=output_maxscore,
                use_fp8_kvcache=use_fp8_kvcache,
            ).numel()
            self._bounds[key] = need
        if self._workspace is None or self._workspace.numel() < need:
            self._workspace = torch.empty(
                need, dtype=torch.uint8, device=torch.device("cuda", device)
            )
        return self._workspace


def query_lens(query_start_loc: torch.Tensor, num_reqs: int) -> torch.Tensor:
    """Per-request query lengths ``[num_reqs]`` int32 from the cumulative map."""
    qsl = query_start_loc[: num_reqs + 1].to(torch.int32)
    return qsl[1:] - qsl[:-1]


def flat_page_table(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_reqs: int,
) -> torch.Tensor:
    """Compact ``[requests, max_blocks]`` into the ``sum_pages`` SAIL form.

    ``block_table`` rows are cut to each request's causal page count and
    concatenated in request order, which is exactly the layout the planner
    derives from ``kv_segment_lens``.
    """
    table = block_table[:num_reqs]
    pages = (seq_lens[:num_reqs].to(torch.int64) + PAGE_SIZE - 1) // PAGE_SIZE
    cols = torch.arange(table.shape[1], device=table.device)
    valid = cols.unsqueeze(0) < pages.unsqueeze(1)
    return table[valid].to(torch.int32)


def token_causal_pages(positions: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """``ceil((position + 1) / PAGE_SIZE)`` per query token, int32."""
    return (positions[:num_tokens] // PAGE_SIZE + 1).to(torch.int32)


def clamp_topk_to_causal(topk: torch.Tensor, num_valid_pages: torch.Tensor) -> None:
    """Rewrite out-of-range block ids to ``-1`` and move them to the tail.

    ``topk`` is ``[tokens, kv_heads, topk_blocks]`` int32, written in place.
    Entries at or beyond the token's causal page count were selected from
    all-``-inf`` score padding; SAIL's sparse attend contract wants them
    dropped (``-1``) and the remaining ids strictly ascending. The ``-1``s must
    land at the *tail*: the attend kernel stops at the first ``-1``, so a
    leading one makes the token attend nothing at all.
    """
    keep = (topk >= 0) & (topk < num_valid_pages.unsqueeze(-1).unsqueeze(-1))
    key = topk.masked_fill(~keep, _INT32_MAX).sort(dim=-1).values
    topk.copy_(key.masked_fill_(key == _INT32_MAX, -1))
