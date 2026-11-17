# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU-optimized Step5 DSA sparse attention kernels.

Further Step5 DSA kernel optimizations belong in this module; they are wired to
upstream call sites by ``vllm_sail/patch/performance/step5_dsa.py``.

Current contents: the indexer scoring kernel, a drop-in replacement for
``paged_mqa_logits_sm90``.

Upstream's ``_paged_scores_sm90`` launches one block per ``(row, tile)`` and
scores a single region tile per block. The query tile and the gate depend only
on the row, so every row reloads them ``cdiv(max_seq_len, TILE)`` times -- 258
times at the production shape. Here each row keeps ``WORKERS`` blocks resident
and strides over the tiles with ``tl.range``, loading query and gate once.

Measured on ZW-M890P (39 SMs): 3.188 ms -> 2.070 ms (-35.1%). The gain is not on
the bandwidth side: DRAM reads drop only 4.17%, while issued memory instructions
drop 97.97% and useful bytes per sector rise from 54 to 63.2 (of 64). The kernel
is bound by memory-transaction throughput, not by bandwidth.

The tile loop also stops at the sequence length: only the tiles ``seq_lens``
covers are scored, and the rest are written with the ``-inf`` sentinel directly.

The head reduction uses ``tl.sum`` rather than upstream's four independent FP32
FMA chains, so scores are not bit-identical to upstream's; this is the variant
validated end to end, and it stays behind the opt-in switch.

Modeled on the ``MAX_TILE_WORKERS`` loop in vllm 0.23's ``step3p5_dsa_triton.py``.
That file's other idea, addressing by page instead of by region, measured as a
regression at this shape (3.19 -> 4.46~6.34 ms; page-table reads are already
absorbed by L1/L2), so it was not ported.
"""

from __future__ import annotations

import functools

import torch
from vllm.triton_utils import tl, triton

_DEFAULT_TILE = 128
_DEFAULT_WARPS = 4


@functools.lru_cache(maxsize=1)
def _resident_blocks() -> int:
    """Target block count of the resident grid.

    Every extra worker per row reloads that row's query tile, which is pure
    overhead; it only pays off when the grid is too small to cover the device.
    Measured: ``workers=1`` is best from rows >= 128 (about 3 waves), and adding
    workers beyond that is monotonically slower, hence a few waves as target.
    """
    return torch.cuda.get_device_properties(0).multi_processor_count * 3


def _resident_workers(rows: int, tiles: int) -> int:
    """Blocks per row, so that ``rows * workers`` approaches the resident target.

    Paired prefill shapes have thousands of rows, so this is 1 there; small
    decode batches pinned to 1 would leave single-digit block counts (measured
    3-7x slower), so it grows as rows shrink.
    """
    return max(1, min(tiles, -(-_resident_blocks() // max(rows, 1))))


@triton.jit(do_not_specialize=["TABLE_ROW", "REGIONS"])
def _resident_scores_sm90(
    Q,
    KV,
    W,
    Lengths,
    Table,
    Output,
    Q_BATCH: tl.constexpr,
    Q_NEXT: tl.constexpr,
    Q_HEAD: tl.constexpr,
    W_ROW: tl.constexpr,
    W_HEAD: tl.constexpr,
    KV_PAGE: tl.constexpr,
    TABLE_ROW,
    TABLE_COL: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    REGIONS,
    NEXT_N: tl.constexpr,
    HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TILE: tl.constexpr,
    WORKERS: tl.constexpr,
):
    """Score by region, with blocks resident per row.

    Each block owns a subset of one row's tiles: ``worker, worker + WORKERS,
    ...``. Query and gate load once outside the loop; the page table and keys
    are still addressed by region, which beats page addressing (see the module
    docstring).
    """
    pid = tl.program_id(0)
    row = pid // WORKERS
    worker = pid % WORKERS
    batch = row // NEXT_N

    d = tl.arange(0, 256)
    h = tl.arange(0, 16)
    query = tl.load(
        Q.to(tl.pointer_type(tl.uint8))
        + batch * Q_BATCH
        + (row % NEXT_N) * Q_NEXT
        + h[None, :] * Q_HEAD
        + d[:, None],
        h[None, :] < HEADS,
        other=0,
    ).to(tl.float8e4nv, bitcast=True)
    weights = tl.load(W + row * W_ROW + h * W_HEAD, h < HEADS, other=0)
    out_base = Output + row * REGIONS

    flat = tl.arange(0, TILE)
    limit = tl.minimum(tl.load(Lengths + batch * LENGTH_STRIDE), REGIONS)
    scored = tl.cdiv(limit, TILE)

    for tile in tl.range(worker, scored, WORKERS):
        region = tile * TILE + flat
        valid = region < limit
        physical = tl.load(
            Table + batch * TABLE_ROW + (region // PAGE_SIZE) * TABLE_COL,
            valid,
            other=0,
        ).to(tl.int64)
        key = tl.load(
            KV
            + physical[:, None] * KV_PAGE
            + (region % PAGE_SIZE)[:, None] * 256
            + d[None, :],
            valid[:, None],
            other=0,
        ).to(tl.float8e4nv, bitcast=True)
        dot = tl.dot(key, query, out_dtype=tl.float32, max_num_imprecise_acc=0)
        dot = tl.maximum(dot, 0.0)
        # dot and weights are zero beyond HEADS, so reducing all 16 lanes equals
        # the masked reduction and avoids the shared-memory syncs of a gather.
        total = tl.sum(dot * weights[None, :], 1)
        scales = tl.load(
            (KV + physical * KV_PAGE + PAGE_SIZE * 256 + (region % PAGE_SIZE) * 4).to(
                tl.pointer_type(tl.float32)
            ),
            valid,
            other=0,
        )
        # The last scored tile straddles limit and its tail must still be
        # written: the output buffer is uninitialized, not zeroed. Mask by
        # REGIONS, not by valid.
        tl.store(
            out_base + region,
            tl.where(valid, total * scales, float("-inf")),
            region < REGIONS,
        )

    # Regions past the sequence length are not scored above, but the consumer's
    # top-k relies on the sentinel, so they must still be written.
    for tile in tl.range(worker + scored, tl.cdiv(REGIONS, TILE), WORKERS):
        region = tile * TILE + flat
        tl.store(out_base + region, float("-inf"), region < REGIONS)


def paged_mqa_logits_sm90(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    *,
    schedule_meta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score Step5's packed FP8 summary pages; same contract as upstream's.

    The packed page holds all FP8 keys followed by one FP32 scale per key.
    Query pairs share a page table and length, as in FlashInfer's interface.
    """
    batch, next_n, heads, dim = q.shape
    assert q.dtype == torch.float8_e4m3fn and dim == 256
    assert heads in (4, 8, 16) and q.stride(-1) == 1
    assert kv_fused.dtype == torch.uint8 and kv_fused.is_contiguous()
    assert kv_fused.shape[2:] == (1, 260)
    assert weights.shape == (batch * next_n, heads)
    output = torch.empty(
        (batch * next_n, max_seq_len), dtype=torch.float32, device=q.device
    )
    if not (batch and max_seq_len):
        return output

    tile, warps = _DEFAULT_TILE, _DEFAULT_WARPS
    rows = batch * next_n
    tiles = -(-max_seq_len // tile)
    workers = _resident_workers(rows, tiles)
    _resident_scores_sm90[(rows * workers,)](
        q,
        kv_fused,
        weights,
        seq_lens,
        block_tables,
        output,
        *q.stride()[:3],
        *weights.stride(),
        kv_fused.stride(0),
        *block_tables.stride(),
        seq_lens.stride(0),
        max_seq_len,
        next_n,
        heads,
        kv_fused.shape[1],
        tile,
        workers,
        num_warps=warps,
    )
    return output
