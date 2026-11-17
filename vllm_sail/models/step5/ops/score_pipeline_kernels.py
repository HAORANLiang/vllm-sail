# SPDX-License-Identifier: Apache-2.0
"""Queries of one request read the summary directly; only the explicit SM89
flow calls this module.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _scores(
    Q,
    W,
    Mean,
    Table,
    Lengths,
    Tiles,
    Output,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    W_ROW: tl.constexpr,
    MEAN_PAGE: tl.constexpr,
    CACHE_PAGES: tl.constexpr,
    REGIONS_PER_BLOCK: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    REGIONS: tl.constexpr,
    HEADS: tl.constexpr,
    NEXT_N: tl.constexpr,
    BQ: tl.constexpr,
    R: tl.constexpr,
):
    tile = tl.program_id(0)
    first = tl.load(Tiles + tile * 3)
    count = tl.load(Tiles + tile * 3 + 1)
    owner = tl.load(Tiles + tile * 3 + 2)
    row = first + tl.arange(0, BQ)
    regions = tl.program_id(1) * R + tl.arange(0, R)
    length = tl.load(Lengths + row // NEXT_N, tl.arange(0, BQ) < count, other=0)
    col = regions // REGIONS_PER_BLOCK
    physical = tl.load(
        Table + owner * TABLE_ROW + col,
        (col < TABLE_COLS) & (regions < REGIONS),
        other=-1,
    ).to(tl.int64)
    valid_page = (regions < REGIONS) & (physical >= 0) & (physical < CACHE_PAGES)
    d = tl.arange(0, 256)
    key = tl.load(
        Mean
        + physical[:, None] * MEAN_PAGE
        + (regions % REGIONS_PER_BLOCK)[:, None] * 256
        + d[None, :],
        valid_page[:, None],
        other=0,
    )
    n = tl.arange(0, BQ * 16)
    qrow, head = first + n // 16, n % 16
    if Q.dtype.element_ty == tl.float16:
        ptr = Q
    else:
        ptr = Q.to(tl.pointer_type(tl.uint8))
    query = tl.load(
        ptr + qrow[None, :] * Q_ROW + head[None, :] * Q_HEAD + d[:, None],
        (n[None, :] // 16 < count) & (head[None, :] < HEADS),
        other=0,
    )
    if Mean.dtype.element_ty == tl.uint8:
        key = key.to(tl.float8e4nv, bitcast=True)
        query = query.to(tl.float8e4nv, bitcast=True)
    dot = tl.dot(key, query, out_dtype=tl.float32, max_num_imprecise_acc=0)
    dot = tl.maximum(dot, 0.0).reshape((R, BQ, 16))
    h = tl.arange(0, 16)
    weights = tl.load(
        W + row[:, None] * W_ROW + h[None, :],
        (tl.arange(0, BQ)[:, None] < count) & (h[None, :] < HEADS),
        other=0,
    )
    sums = tl.full((R, BQ, 4), 0, tl.float32)
    lane = tl.arange(0, 4)
    for group in tl.static_range(HEADS // 4):
        index = group * 4 + lane
        a = tl.gather(dot, tl.broadcast_to(index[None, None, :], (R, BQ, 4)), 2)
        w = tl.gather(weights, tl.broadcast_to(index[None, :], (BQ, 4)), 1)
        sums = tl.fma(a, w[None, :, :], sums)
    a = tl.gather(sums, tl.full((R, BQ, 1), 0, tl.int32), 2).reshape((R, BQ))
    b = tl.gather(sums, tl.full((R, BQ, 1), 1, tl.int32), 2).reshape((R, BQ))
    c = tl.gather(sums, tl.full((R, BQ, 1), 2, tl.int32), 2).reshape((R, BQ))
    e = tl.gather(sums, tl.full((R, BQ, 1), 3, tl.int32), 2).reshape((R, BQ))
    result = ((a + b) + c) + e
    valid = (regions[:, None] < REGIONS) & (regions[:, None] < length[None, :])
    tl.store(
        Output + row[None, :] * REGIONS + regions[:, None],
        tl.where(valid, result, float("-inf")),
        (regions[:, None] < REGIONS) & (tl.arange(0, BQ)[None, :] < count),
    )


def launch(
    query, weights, mean, table, lengths, regions, full, tails, bq, heads, next_n
):
    output = torch.empty(
        (query.shape[0], regions), dtype=torch.float32, device=query.device
    )
    for tiles, block_q in ((full, bq), (tails, 1)):
        if tiles.shape[0]:
            _scores[(tiles.shape[0], triton.cdiv(regions, 32))](
                query,
                weights,
                mean,
                table,
                lengths,
                tiles,
                output,
                query.stride(0),
                query.stride(2),
                weights.stride(0),
                mean.stride(0),
                mean.shape[0],
                mean.shape[1],
                table.stride(0),
                table.shape[1],
                regions,
                heads,
                next_n,
                block_q,
                32,
                num_warps=4,
            )
    return output
