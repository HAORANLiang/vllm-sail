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

from vllm_sail.models.step5.ops.soft_fp8 import _direct_scores, e4m3_dot
from vllm_sail.step5_sparse_cases import profiled

_DEFAULT_TILE = 128
_DEFAULT_WARPS = 4


@triton.jit
def _direct_four(values, ROWS: tl.constexpr):
    """Extract four lanes without gather or a shared-memory reduction."""
    even, odd = tl.split(values.reshape((ROWS, 2, 2)))
    a, c = tl.split(even)
    b, d = tl.split(odd)
    return a, b, c, d


@triton.jit
def _direct_chain(values, weights, TILE: tl.constexpr, HEADS: tl.constexpr):
    a, b, c, d = _direct_four(values, TILE)
    wa, wb, wc, wd = _direct_four(weights, 1)
    total = tl.fma(a, wa, tl.full((TILE,), 0, tl.float32))
    if HEADS >= 8:
        total = tl.fma(b, wb, total)
    if HEADS >= 16:
        total = tl.fma(c, wc, total)
        total = tl.fma(d, wd, total)
    return total


@triton.jit
def _direct_split_epilogue(dot, weights, TILE: tl.constexpr, HEADS: tl.constexpr):
    a, b, c, d = _direct_four(dot, TILE * 4)
    wa, wb, wc, wd = _direct_four(weights, 4)
    a = _direct_chain(a, wa, TILE, HEADS)
    b = _direct_chain(b, wb, TILE, HEADS)
    c = _direct_chain(c, wc, TILE, HEADS)
    d = _direct_chain(d, wd, TILE, HEADS)
    return ((a + b) + c) + d


@triton.jit
def _direct_packed_epilogue(dot, weights, TILE: tl.constexpr, QUERIES: tl.constexpr):
    """Retain four independent H8 FMA chains per query without assuming padded heads are zero."""
    even, odd = tl.split(dot.reshape((TILE * QUERIES, 4, 2)))
    a, c, e, g = _direct_four(even, TILE * QUERIES)
    b, d, f, h = _direct_four(odd, TILE * QUERIES)
    even_w, odd_w = tl.split(weights.reshape((QUERIES, 4, 2)))
    wa, wc, we, wg = _direct_four(even_w, QUERIES)
    wb, wd, wf, wh = _direct_four(odd_w, QUERIES)
    zero = tl.full((TILE, QUERIES), 0, tl.float32)
    sa = tl.fma(a.reshape((TILE, QUERIES)), wa[None, :], zero)
    sb = tl.fma(b.reshape((TILE, QUERIES)), wb[None, :], zero)
    sc = tl.fma(c.reshape((TILE, QUERIES)), wc[None, :], zero)
    sd = tl.fma(d.reshape((TILE, QUERIES)), wd[None, :], zero)
    sa = tl.fma(e.reshape((TILE, QUERIES)), we[None, :], sa)
    sb = tl.fma(f.reshape((TILE, QUERIES)), wf[None, :], sb)
    sc = tl.fma(g.reshape((TILE, QUERIES)), wg[None, :], sc)
    sd = tl.fma(h.reshape((TILE, QUERIES)), wh[None, :], sd)
    return ((sa + sb) + sc) + sd


@triton.jit
def _direct_scores_mtp(
    Q,
    W,
    Mean,
    Table,
    Slots,
    SeqLens,
    Scores,
    Active,
    Positions,
    Current,
    Lengths,
    TopKLengths,
    Q_TOKEN: tl.constexpr,
    Q_GROUP: tl.constexpr,
    W_TOKEN: tl.constexpr,
    W_GROUP: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    SEQ_STRIDE: tl.constexpr,
    CACHE_PAGES: tl.constexpr,
    REGIONS_PER_BLOCK: tl.constexpr,
    MEAN_BLOCK: tl.constexpr,
    CAPACITY: tl.constexpr,
    SCORE_ROW: tl.constexpr,
    TILE: tl.constexpr,
    WORKERS: tl.constexpr,
    ROW_GROUPS: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    ORIGINAL_GRID: tl.constexpr = False,
):
    """MTP queries from the same request share K reads while preserving the original per-row write boundary."""
    if ORIGINAL_GRID:
        row = tl.program_id(1).to(tl.int64)
        token, kv_group = row // ROW_GROUPS, row % ROW_GROUPS
        request = token // QUERY_LEN
        worker = (token % QUERY_LEN) * WORKERS + tl.program_id(0)
    else:
        pid = tl.program_id(0).to(tl.int64)
        owner, worker = pid // (WORKERS * QUERY_LEN), pid % (WORKERS * QUERY_LEN)
        request, kv_group = owner // ROW_GROUPS, owner % ROW_GROUPS
    qid = tl.arange(0, QUERY_LEN)
    token = request * QUERY_LEN + qid
    row = token * ROW_GROUPS + kv_group
    active = tl.load(Slots + token * SLOT_STRIDE) >= 0
    seq = tl.load(SeqLens + request * SEQ_STRIDE).to(tl.int64)
    position = tl.where(active, seq - QUERY_LEN + qid, 0)
    current = position >> 3
    length = current + 1
    if worker == 0:
        tl.store(Active + row, active)
        tl.store(Positions + row, position)
        tl.store(Current + row, current)
        tl.store(Lengths + row, length)
        tl.store(TopKLengths + row, current)
    live_tiles = tl.cdiv(tl.minimum(tl.maximum(length, 0), CAPACITY), TILE)
    d, h = tl.arange(0, 256), tl.arange(0, QUERY_LEN * 8)
    query_ptr = Q.to(tl.pointer_type(tl.uint8))
    query = tl.load(
        query_ptr
        + (request * QUERY_LEN + h[None, :] // 8) * Q_TOKEN
        + kv_group * Q_GROUP
        + (h[None, :] % 8) * 256
        + d[:, None]
    )
    weights = tl.load(
        W + (request * QUERY_LEN + h // 8) * W_TOKEN + kv_group * W_GROUP + h % 8
    )
    for tile in tl.range(worker, tl.max(live_tiles, 0), WORKERS * QUERY_LEN):
        region = tile * TILE + tl.arange(0, TILE)
        physical = tl.load(
            Table + request * TABLE_ROW + region // REGIONS_PER_BLOCK,
            region // REGIONS_PER_BLOCK < TABLE_COLS,
            other=-1,
        ).to(tl.int64)
        valid = (region < CAPACITY) & (physical >= 0) & (physical < CACHE_PAGES)
        raw = tl.load(
            Mean
            + physical[:, None] * MEAN_BLOCK
            + (region % REGIONS_PER_BLOCK)[:, None] * 256
            + d[None, :],
            valid[:, None],
            other=0,
        )
        dot = tl.maximum(e4m3_dot(raw, query), 0.0)
        result = _direct_packed_epilogue(dot, weights, TILE, QUERY_LEN)
        tl.store(
            Scores + row[None, :] * SCORE_ROW + region[:, None],
            result,
            (region[:, None] < CAPACITY) & (tile < live_tiles[None, :]),
        )


@triton.jit
def _direct_scores_resident(
    Q,
    W,
    Mean,
    Table,
    Slots,
    SeqLens,
    Scores,
    Active,
    Positions,
    Current,
    Lengths,
    TopKLengths,
    Q_TOKEN: tl.constexpr,
    Q_GROUP: tl.constexpr,
    W_TOKEN: tl.constexpr,
    W_GROUP: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    SEQ_STRIDE: tl.constexpr,
    CACHE_PAGES: tl.constexpr,
    REGIONS_PER_BLOCK: tl.constexpr,
    MEAN_BLOCK: tl.constexpr,
    CAPACITY: tl.constexpr,
    SCORE_ROW: tl.constexpr,
    HEADS: tl.constexpr,
    TILE: tl.constexpr,
    WORKERS: tl.constexpr,
    ROW_GROUPS: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    HOIST: tl.constexpr = True,
    SPLIT: tl.constexpr = True,
    PACK_QUERIES: tl.constexpr = False,
    GUARD_MTP: tl.constexpr = False,
):
    """Decode scorer; original dot/FMA order and written prefix.

    Offline ablations are explicit constexpr knobs, not runtime mode switches.
    The original soft_fp8._direct_scores remains the acceptance baseline.
    """
    if GUARD_MTP:
        tl.static_assert(HEADS == 8 and TILE == 128 and QUERY_LEN == 4)
        tl.static_assert(Q.dtype.element_ty != tl.float16)
        request = tl.program_id(1).to(tl.int64) // (ROW_GROUPS * QUERY_LEN)
        seq = tl.load(SeqLens + request * SEQ_STRIDE).to(tl.int64)
        use_mtp = (seq >= 65536) & (seq <= 69632)
        if use_mtp:
            for offset in tl.static_range(QUERY_LEN):
                use_mtp &= (
                    tl.load(Slots + (request * QUERY_LEN + offset) * SLOT_STRIDE) >= 0
                )
        # Cover only measured long-context windows; all CTAs for a request must select the same path.
        if use_mtp:
            _direct_scores_mtp(
                Q,
                W,
                Mean,
                Table,
                Slots,
                SeqLens,
                Scores,
                Active,
                Positions,
                Current,
                Lengths,
                TopKLengths,
                Q_TOKEN,
                Q_GROUP,
                W_TOKEN,
                W_GROUP,
                TABLE_COLS,
                TABLE_ROW,
                SLOT_STRIDE,
                SEQ_STRIDE,
                CACHE_PAGES,
                REGIONS_PER_BLOCK,
                MEAN_BLOCK,
                CAPACITY,
                SCORE_ROW,
                TILE,
                WORKERS,
                ROW_GROUPS,
                QUERY_LEN,
                ORIGINAL_GRID=True,
            )
        else:
            _direct_scores(
                Q,
                W,
                Mean,
                Table,
                Slots,
                SeqLens,
                Scores,
                Active,
                Positions,
                Current,
                Lengths,
                TopKLengths,
                Q_TOKEN,
                Q_GROUP,
                W_TOKEN,
                W_GROUP,
                TABLE_COLS,
                TABLE_ROW,
                SLOT_STRIDE,
                SEQ_STRIDE,
                CACHE_PAGES,
                REGIONS_PER_BLOCK,
                MEAN_BLOCK,
                CAPACITY,
                SCORE_ROW,
                HEADS,
                TILE,
                WORKERS,
                ROW_GROUPS,
                QUERY_LEN,
            )
        return
    if PACK_QUERIES:
        tl.static_assert(HEADS == 8 and TILE == 128)
        tl.static_assert(QUERY_LEN == 2 or QUERY_LEN == 4)
        tl.static_assert(Q.dtype.element_ty != tl.float16)
        _direct_scores_mtp(
            Q,
            W,
            Mean,
            Table,
            Slots,
            SeqLens,
            Scores,
            Active,
            Positions,
            Current,
            Lengths,
            TopKLengths,
            Q_TOKEN,
            Q_GROUP,
            W_TOKEN,
            W_GROUP,
            TABLE_COLS,
            TABLE_ROW,
            SLOT_STRIDE,
            SEQ_STRIDE,
            CACHE_PAGES,
            REGIONS_PER_BLOCK,
            MEAN_BLOCK,
            CAPACITY,
            SCORE_ROW,
            TILE,
            WORKERS,
            ROW_GROUPS,
            QUERY_LEN,
        )
        return
    pid = tl.program_id(0).to(tl.int64)
    row, worker = pid // WORKERS, pid % WORKERS
    token, kv_group = row // ROW_GROUPS, row % ROW_GROUPS
    request = token // QUERY_LEN
    active = tl.load(Slots + token * SLOT_STRIDE) >= 0
    seq = tl.load(SeqLens + request * SEQ_STRIDE).to(tl.int64) - (
        QUERY_LEN - 1 - token % QUERY_LEN
    )
    position = tl.where(active, seq - 1, 0)
    current = position >> 3
    length = current + 1
    if worker == 0:
        tl.store(Active + row, active)
        tl.store(Positions + row, position)
        tl.store(Current + row, current)
        tl.store(Lengths + row, length)
        tl.store(TopKLengths + row, current)
    live_tiles = tl.cdiv(tl.minimum(tl.maximum(length, 0), CAPACITY), TILE)
    d, h = tl.arange(0, 256), tl.arange(0, 16)
    if Q.dtype.element_ty == tl.float16:
        query_ptr = Q
    else:
        query_ptr = Q.to(tl.pointer_type(tl.uint8))
    query_ptr += token * Q_TOKEN + kv_group * Q_GROUP
    weight_ptr = W + token * W_TOKEN + kv_group * W_GROUP
    if HOIST:
        query = tl.load(
            query_ptr + h[None, :] * 256 + d[:, None], h[None, :] < HEADS, other=0
        )
        weights = tl.load(weight_ptr + h, h < HEADS, other=0)
    for tile in tl.range(worker, live_tiles, WORKERS):
        region = tile * TILE + tl.arange(0, TILE)
        col = region // REGIONS_PER_BLOCK
        physical = tl.load(
            Table + request * TABLE_ROW + col, col < TABLE_COLS, other=-1
        ).to(tl.int64)
        valid = (region < CAPACITY) & (physical >= 0) & (physical < CACHE_PAGES)
        raw = tl.load(
            Mean
            + physical[:, None] * MEAN_BLOCK
            + (region % REGIONS_PER_BLOCK)[:, None] * 256
            + d[None, :],
            valid[:, None],
            other=0,
        )
        if not HOIST:
            query = tl.load(
                query_ptr + h[None, :] * 256 + d[:, None], h[None, :] < HEADS, other=0
            )
            weights = tl.load(weight_ptr + h, h < HEADS, other=0)
        dot = tl.maximum(e4m3_dot(raw, query), 0.0)
        if SPLIT:
            result = _direct_split_epilogue(dot, weights, TILE, HEADS)
        else:
            sums = tl.full((TILE, 4), 0, tl.float32)
            lane = tl.arange(0, 4)
            for group in tl.static_range(HEADS // 4):
                index = group * 4 + lane
                a = tl.gather(dot, tl.broadcast_to(index[None, :], (TILE, 4)), 1)
                w = tl.gather(weights, index, 0)
                sums = tl.fma(a, w[None, :], sums)
            a = tl.gather(sums, tl.full((TILE, 1), 0, tl.int32), 1).reshape((TILE,))
            b = tl.gather(sums, tl.full((TILE, 1), 1, tl.int32), 1).reshape((TILE,))
            c = tl.gather(sums, tl.full((TILE, 1), 2, tl.int32), 1).reshape((TILE,))
            e = tl.gather(sums, tl.full((TILE, 1), 3, tl.int32), 1).reshape((TILE,))
            result = ((a + b) + c) + e
        tl.store(Scores + row * SCORE_ROW + region, result, region < CAPACITY)


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


@triton.jit(do_not_specialize=["REGIONS", "PAGES"])
def _grouped_scores_sm89(
    Q,
    KV,
    W,
    Lengths,
    Spans,
    Output,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    W_ROW: tl.constexpr,
    W_HEAD: tl.constexpr,
    KV_PAGE: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    REGIONS,
    PAGES,
    HEADS: tl.constexpr,
    NEXT_N: tl.constexpr,
    GROUP_ROWS: tl.constexpr,
    TILE: tl.constexpr,
    GRID_TILES: tl.constexpr,
    SPAN_COUNT: tl.constexpr,
    Q_GROUP: tl.constexpr,
    W_GROUP: tl.constexpr,
    OUTPUT_GROUP: tl.constexpr,
):
    """All local groups reuse the same packed K; spans never cross a request."""
    program = tl.program_id(0).to(tl.int64)
    group = program // SPAN_COUNT
    span = program % SPAN_COUNT
    worker = tl.program_id(1)
    first = tl.load(Spans + span * 3).to(tl.int64)
    count = tl.load(Spans + span * 3 + 1)
    owner = tl.load(Spans + span * 3 + 2).to(tl.int64)
    row = first + tl.arange(0, GROUP_ROWS)
    row_valid = tl.arange(0, GROUP_ROWS) < count
    lengths = tl.load(
        Lengths + row // NEXT_N * LENGTH_STRIDE,
        row_valid,
        other=0,
    )
    lengths = tl.minimum(tl.maximum(lengths, 0), REGIONS)
    max_live = tl.max(lengths, 0)
    matrix_row = tl.arange(0, GROUP_ROWS * HEADS)
    query_row = first + matrix_row // HEADS
    head = matrix_row % HEADS
    d = tl.arange(0, 256)
    query = tl.load(
        Q.to(tl.pointer_type(tl.uint8))
        + group * Q_GROUP
        + query_row[:, None] * Q_ROW
        + head[:, None] * Q_HEAD
        + d[None, :],
        (matrix_row[:, None] // HEADS) < count,
        other=0,
    ).to(tl.float8e4nv, bitcast=True)
    weights = tl.load(
        W + group * W_GROUP + query_row * W_ROW + head * W_HEAD,
        matrix_row // HEADS < count,
        other=0,
    ).to(tl.float32)
    for tile in tl.range(worker, tl.cdiv(REGIONS, TILE), GRID_TILES):
        region = tile.to(tl.int64) * TILE + tl.arange(0, TILE)
        if tile * TILE < max_live:
            valid = (region < REGIONS) & (region < max_live)
            physical = owner * PAGES + region // 32
            key = tl.load(
                KV
                + physical[None, :] * KV_PAGE
                + (region[None, :] % 32) * 256
                + d[:, None],
                valid[None, :],
                other=0,
            ).to(tl.float8e4nv, bitcast=True)
            dot = tl.dot(query, key, out_dtype=tl.float32, max_num_imprecise_acc=0)
            weighted = tl.maximum(dot, 0.0) * weights[:, None]
            total = tl.sum(weighted.reshape((GROUP_ROWS, HEADS, TILE)), 1)
            scales = tl.load(
                (KV + physical * KV_PAGE + 32 * 256 + (region % 32) * 4).to(
                    tl.pointer_type(tl.float32)
                ),
                valid,
                other=0,
            )
            result = tl.where(
                region[None, :] < lengths[:, None],
                total * scales[None, :],
                float("-inf"),
            )
        else:
            result = tl.full((GROUP_ROWS, TILE), float("-inf"), tl.float32)
        tl.store(
            Output + group * OUTPUT_GROUP + row[:, None] * REGIONS + region[None, :],
            result,
            row_valid[:, None] & (region[None, :] < REGIONS),
        )


def _submit_scores(
    kernel, grid, launch_args, config, values, backend, pd_type="unknown"
):
    from vllm_sail import envs

    if not envs.VLLM_SAIL_NVTX_PROFILE:
        return kernel[grid](*launch_args, **config)
    from vllm_sail.profiling.step5_sparse import trace_call

    return trace_call(
        "score",
        kernel.__name__,
        backend,
        values,
        lambda: kernel[grid](*launch_args, **config),
        scope="kernel",
        pd_type=pd_type,
    )


@profiled("score", backend="compact_grouped", pd_type="prefill")
def grouped_paged_logits(
    query,
    weights,
    packed,
    lengths,
    spans,
    regions,
    pages,
    heads,
    next_n,
    *,
    group_rows=None,
    tile=32,
    grid_cap=32,
    warps=2,
    groups=1,
):
    """Inputs come from a validated packed plan; config overrides are for offline acceptance only."""
    if heads not in (4, 8, 16) or isinstance(heads, bool):
        raise ValueError("unsupported grouped heads")
    group_rows = 64 // heads if group_rows is None else group_rows
    if (
        group_rows not in (4, 8, 16)
        or isinstance(group_rows, bool)
        or tile not in (32, 64, 128)
        or grid_cap not in (8, 16, 32)
        or warps not in (2, 4)
        or next_n not in (1, 2)
    ):
        raise ValueError("unsupported grouped score configuration")
    if (
        not isinstance(groups, int)
        or isinstance(groups, bool)
        or groups < 1
        or query.shape[1] != groups
        or weights.shape != query.shape[:3]
    ):
        raise ValueError("unsupported grouped score geometry")
    shape = (
        (groups, query.shape[0], regions) if groups > 1 else (query.shape[0], regions)
    )
    output = torch.empty(shape, dtype=torch.float32, device=query.device)
    if query.shape[0] and regions:
        workers = min(triton.cdiv(regions, tile), grid_cap)
        _submit_scores(
            _grouped_scores_sm89,
            (groups * spans.shape[0], workers),
            (
                query,
                packed,
                weights,
                lengths,
                spans,
                output,
                query.stride(0),
                query.stride(2),
                weights.stride(0),
                weights.stride(2),
                packed.stride(0),
                lengths.stride(0),
                regions,
                pages,
                heads,
                next_n,
                group_rows,
                tile,
                workers,
                spans.shape[0],
                query.stride(1),
                weights.stride(1),
                query.shape[0] * regions,
            ),
            dict(num_warps=warps),
            dict(
                query=query,
                packed=packed,
                weights=weights,
                lengths=lengths,
                spans=spans,
                output=output,
                regions=regions,
                pages=pages,
                heads=heads,
                next_n=next_n,
                groups=groups,
                group_rows=group_rows,
                tile=tile,
                workers=workers,
                num_warps=warps,
            ),
            "compact_grouped",
            "prefill",
        )
    return output


@profiled("score", backend="resident")
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
    _submit_scores(
        _resident_scores_sm90,
        (rows * workers,),
        (
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
        ),
        dict(num_warps=warps),
        dict(
            query=q,
            packed=kv_fused,
            weights=weights,
            lengths=seq_lens,
            block_table=block_tables,
            output=output,
            max_seq_len=max_seq_len,
            heads=heads,
            next_n=next_n,
            tile=tile,
            workers=workers,
            num_warps=warps,
        ),
        "resident",
    )
    return output
