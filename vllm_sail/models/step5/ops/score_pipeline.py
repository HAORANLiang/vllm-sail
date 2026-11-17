# SPDX-License-Identifier: Apache-2.0
"""Multi-query dataflow of scored prefill; host checks never read device values."""

from __future__ import annotations

import functools
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar

_PLAN_ATTRIBUTE = "step5_sail_score_plan"
_CONTEXT = ContextVar("step5_score_pipeline", default=None)
_EXPERIMENT = ContextVar("step5_score_experiment", default=False)
# The candidate is called only from the standalone acceptance context, never
# from production patches.
_STATS = Counter()


@contextmanager
def _experimental():
    """Standalone acceptance allows candidate shapes without whitelisting them for
    production dispatch.
    """
    token = _EXPERIMENT.set(True)
    try:
        yield
    finally:
        _EXPERIMENT.reset(token)


def _runtime_key(runtime):
    return (
        id(runtime),
        tuple(id(value) for value in runtime),
        tuple(id(meta.lengths) for meta in runtime[4]),
        tuple(id(pair[0]) for pair in runtime[8]) if runtime[8] is not None else (),
    )


def _dispatch_key(starts, rows, regions, paired, bq):
    return (
        tuple(b - a for a, b in zip(starts, starts[1:], strict=False)),
        tuple(rows),
        regions,
        paired,
        bq,
    )


def stats(reset=False):
    result = dict(_STATS)
    if reset:
        _STATS.clear()
    return result


def query_tiles(starts, chunk_rows, bq):
    """Each tile is (start within chunk, valid rows, request); tail rows use Q1."""
    if bq not in (2, 4, 8) or isinstance(bq, bool):
        raise ValueError("BQ must be one of 2/4/8")
    starts, chunk_rows = tuple(starts), tuple(chunk_rows)
    if (
        not starts
        or starts[0] != 0
        or any(b < a for a, b in zip(starts, starts[1:], strict=False))
        or any(n <= 0 for n in chunk_rows)
        or sum(chunk_rows) != starts[-1]
    ):
        raise ValueError("incomplete request/chunk boundaries")
    result, begin = [], 0
    for size in chunk_rows:
        end, full, tails = begin + size, [], []
        for owner, (left, right) in enumerate(zip(starts, starts[1:], strict=False)):
            left, right = max(begin, left), min(end, right)
            while left + bq <= right:
                full.append((left - begin, bq, owner))
                left += bq
            tails.extend((row - begin, 1, owner) for row in range(left, right))
        result.append((begin, end, tuple(full), tuple(tails)))
        begin = end
    return tuple(result)


class ScorePlan:
    def __init__(self, runtime, starts, bq, device):
        import torch

        rows = tuple(meta.current.numel() for meta in runtime[4])
        self.host = query_tiles(starts, rows, bq)
        self.runtime_key, self.bq = _runtime_key(runtime), bq
        self.dispatch_key = _dispatch_key(
            starts, rows, runtime[2], runtime[8] is not None, bq
        )
        flat = [tile for _, _, full, tails in self.host for tile in (*full, *tails)]
        # Upload once per step; request metadata is not rebuilt per layer/chunk.
        self.tiles = (
            torch.tensor(flat, dtype=torch.int32)
            .reshape(-1, 3)
            .pin_memory()
            .to(device, non_blocking=True)
        )
        self.chunks, cursor = [], 0
        for begin, end, full, tails in self.host:
            middle, last = cursor + len(full), cursor + len(full) + len(tails)
            self.chunks.append(
                (begin, end, self.tiles[cursor:middle], self.tiles[middle:last])
            )
            cursor = last


def attach_plan(metadata, runtime, ppu, *, bq=4):
    """Build a plan with an explicit BQ, only in the standalone experiment context.

    Production uploads no extra metadata.
    """
    setattr(metadata, _PLAN_ATTRIBUTE, None)
    if not _EXPERIMENT.get():
        _STATS["reject_unverified_dispatch"] += 1
        return runtime
    if (
        not ppu
        or runtime is None
        or not runtime[4]
        or getattr(runtime[4][0], "selected", None) is not None
    ):
        return runtime
    import torch

    device = metadata.slot_mapping.device
    if tuple(torch.cuda.get_device_capability(device)) != (8, 9):
        _STATS["reject_plan_capability"] += 1
        return runtime
    if getattr(metadata.query_start_loc_cpu, "is_cuda", False):
        _STATS["reject_host_starts"] += 1
        return runtime
    starts = metadata.query_start_loc_cpu[: metadata.num_actual_reqs + 1].tolist()
    plan = ScorePlan(runtime, starts, bq, metadata.slot_mapping.device)
    setattr(metadata, _PLAN_ATTRIBUTE, plan)
    return runtime


def contract_reason(query, weights, mean, table, plan, runtime, paired_heads):
    import torch

    if plan is None or runtime is None or plan.runtime_key != _runtime_key(runtime):
        return "runtime"
    if query is None or weights is None or mean is None or table is None:
        return "missing_input"
    tensors = (query, weights, mean, table, plan.tiles)
    if not query.is_cuda or any(t.device != query.device for t in tensors):
        return "device"
    if query.ndim != 4 or weights.ndim != 3 or mean.ndim != 3 or table.ndim != 2:
        return "rank"
    rows = plan.host[-1][1] if plan.host else 0
    paired = runtime[8] is not None
    if (
        query.shape[0] != rows
        or query.shape[1] != 1
        or query.shape[-1] != 256
        or query.shape[2] != 8
        or weights.shape != query.shape[:3]
        or (paired and paired_heads != 4)
    ):
        return "query_geometry"
    if (query.dtype, mean.dtype) not in (
        (torch.float8_e4m3fn, torch.uint8),
        (torch.float16, torch.float16),
    ) or weights.dtype != torch.float32:
        return "dtype"
    if (
        mean.shape[1] not in (2, 4, 8, 16)
        or mean.shape[2] != 256
        or mean.stride(1) != 256
        or mean.stride(2) != 1
        or mean.stride(0) < mean.shape[1] * 256
        or query.stride(3) != 1
        or query.stride(2) != 256
        or query.stride(0) < 8 * 256
        or weights.stride(2) != 1
        or weights.stride(0) < 8
        or table.stride(1) != 1
        or table.dtype != torch.int32
        or table.shape[1] <= 0
    ):
        return "layout"
    owners = [t[2] for _, _, full, tails in plan.host for t in (*full, *tails)]
    if owners and max(owners) >= table.shape[0]:
        return "request_table"
    if tuple(torch.cuda.get_device_capability(query.device)) != (8, 9):
        return "capability"
    if runtime[2] <= 0 or runtime[2] > 2147483647:
        return "regions"
    if paired and len(runtime[8]) != len(plan.chunks):
        return "chunk_lengths"
    for index, (begin, end, _, _) in enumerate(plan.chunks):
        meta = runtime[4][index]
        lengths = runtime[8][index][0] if paired else meta.lengths
        if (
            lengths.dtype != torch.int32
            or lengths.ndim != 1
            or lengths.numel() != (end - begin) // (2 if paired else 1)
            or lengths.device != query.device
            or lengths.stride(0) != 1
            or (paired and (begin % 2 or (end - begin) % 2))
            or (end - begin) * runtime[2] * 4 > 128 << 20
        ):
            return "chunk_lengths"
    return None


@contextmanager
def bind(metadata, runtime, adapter, query, weights, mean, table, paired_heads, ppu):
    # Outside the experiment context, never import torch or query device properties.
    context = None
    if _EXPERIMENT.get() and ppu:
        plan = getattr(metadata, _PLAN_ATTRIBUTE, None)
        reason = contract_reason(
            query, weights, mean, table, plan, runtime, paired_heads
        )
        if reason is None:
            context = ScoreContext(
                adapter,
                plan,
                mean,
                table,
                runtime[2],
                paired_heads,
                query,
                weights,
                runtime,
            )
        else:
            _STATS["reject_" + reason] += 1
    elif not _EXPERIMENT.get():
        _STATS["reject_unverified_dispatch"] += 1
    token = _CONTEXT.set(context)
    try:
        yield context
    finally:
        _CONTEXT.reset(token)
        if context is not None:
            context.release()


def wrap_plan(original, platform, *, bq=4):
    @functools.wraps(original)
    def planned(self, attn_metadata, n, lengths, layout, positions=None):
        runtime = original(self, attn_metadata, n, lengths, layout, positions)
        return attach_plan(attn_metadata, runtime, platform.is_ppu(), bq=bq)

    return planned


def wrap_forward(original, platform):
    @functools.wraps(original)
    def forward(
        self,
        query,
        kv_cache,
        attn_metadata,
        output,
        dsa_proxy_query,
        dsa_proxy_key,
        dsa_proxy_weights,
        dsa_proxy_z,
        **kwargs,
    ):
        args = (
            self,
            query,
            kv_cache,
            attn_metadata,
            output,
            dsa_proxy_query,
            dsa_proxy_key,
            dsa_proxy_weights,
            dsa_proxy_z,
        )
        if not _EXPERIMENT.get() or not platform.is_ppu():
            return original(*args, **kwargs)
        with bind(
            attn_metadata,
            attn_metadata.step5_runtime,
            self.adapter,
            dsa_proxy_query,
            dsa_proxy_weights,
            self.summary_cache.mean,
            attn_metadata.block_table[: attn_metadata.num_actual_reqs],
            self.indexer_heads,
            True,
        ):
            return original(*args, **kwargs)

    return forward


class ScoreContext:
    def __init__(
        self, adapter, plan, mean, table, regions, paired_heads, query, weights, runtime
    ):
        self.adapter, self.plan, self.mean, self.table = adapter, plan, mean, table
        self.query, self.weights, self.runtime = query, weights, runtime
        self.regions, self.paired_heads, self.cursor = regions, paired_heads, 0

    def release(self):
        self.adapter = self.mean = self.table = self.plan = self.query = (
            self.weights
        ) = self.runtime = None


class RawSummary:
    """Materialize lazily through the original interface.

    Lives only for this layer/group and keeps the packed Tensor contract.
    """

    def __init__(self, context, original):
        self.context, self.original, self.packed = context, original, None

    def materialize(self):
        if self.packed is None:
            c = self.context
            self.packed = self.original(c.adapter, c.mean, c.table, c.regions)
            _STATS["fallback_pack"] += 1
        return self.packed


def pack_or_reference(original, adapter, mean, table, regions):
    c = _CONTEXT.get()
    if (
        not _EXPERIMENT.get()
        or c is None
        or c.adapter is not adapter
        or c.mean is not mean
        or c.regions != regions
        or c.table.data_ptr() != table.data_ptr()
        or c.table.shape != table.shape
        or c.table.stride() != table.stride()
    ):
        return original(adapter, mean, table, regions)
    _STATS["raw_pack"] += 1
    return RawSummary(c, original), ((regions + 127) // 128) * 4


def raw_logits(raw, adapter, query, weights, lengths, regions, pairs, paired_heads):
    """None means fall back before launch; JIT/device errors are not caught."""
    import torch

    c = raw.context
    if (
        not _EXPERIMENT.get()
        or c is not _CONTEXT.get()
        or c.adapter is not adapter
        or raw.packed is not None
    ):
        return None
    if c.cursor >= len(c.plan.chunks):
        return None
    begin, end, full, tails = c.plan.chunks[c.cursor]
    paired = c.runtime[8] is not None
    valid = (
        query.shape == (end - begin, 1, 8, 256)
        and weights.shape == query.shape[:3]
        and regions == c.regions
        and query.stride() == c.query.stride()
        and weights.stride() == c.weights.stride()
        and query.data_ptr() == c.query[begin:end].data_ptr()
        and weights.data_ptr() == c.weights[begin:end].data_ptr()
        and weights.dtype == torch.float32
        and query.device == c.mean.device
        and weights.device == c.mean.device
        and (query.dtype, c.mean.dtype)
        in ((torch.float8_e4m3fn, torch.uint8), (torch.float16, torch.float16))
        and (pairs is not None) == paired
        and (not paired or paired_heads == 4)
    )
    c.cursor += 1
    effective = pairs[0] if paired and pairs is not None else lengths
    expected = (
        c.runtime[8][c.cursor - 1][0] if paired else c.runtime[4][c.cursor - 1].lengths
    )
    if not valid or effective is not expected:
        _STATS["reject_logits"] += 1
        return None
    from vllm_sail.models.step5.ops.score_pipeline_kernels import launch

    result = launch(
        query,
        weights,
        c.mean,
        c.table,
        effective,
        regions,
        full,
        tails,
        c.plan.bq,
        4 if paired else 8,
        2 if paired else 1,
    )
    _STATS["mqa_chunks"] += 1
    _STATS["mqa_tiles"] += full.shape[0]
    _STATS["q1_tiles"] += tails.shape[0]
    return result
