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
_PACKED_ATTRIBUTE = "step5_sail_packed_score_plan"
_PACKED_CONTEXT = ContextVar("step5_packed_score", default=None)
PACKED_REASON = (
    "PPU SM89 Step5 prefill groups query/head rows within a request and reuses "
    "the existing packed K; enabled by VLLM_SAIL_STEP5_DSA_SCORES=v2/v3, "
    "v0/v1 build no compact plan."
)
PACKED_REMOVE_WHEN = (
    "Upstream Step5 prefill scorer reuses packed K across queries within a request."
)


_LAYER_ATTRIBUTE = "step5_sail_layer_score_plan"
_LAYER_CONTEXT = ContextVar("step5_layer_score", default=None)
_LAYER_LOGGED = set()
LAYER_REASON = (
    "PPU FP8 prefill scores all actual queries and local KV groups once per layer "
    "under v1/v2, reusing shared packed summaries within the forward only. "
    "The lazy SCORE_MAX_BYTES budget rejects consolidation before allocation."
)
LAYER_REMOVE_WHEN = "Upstream Step5 consolidates all prefill group/chunk scores."


def layer_enabled():
    from vllm_sail import envs

    return envs.VLLM_SAIL_STEP5_DSA_SCORES in ("v1", "v2")


def _view_key(tensor):
    if tensor is None:
        return None
    return (
        id(tensor),
        tuple(tensor.shape),
        tensor.stride(),
        tensor.device,
        tensor.dtype,
        tensor.data_ptr(),
    )


def _layer_runtime_key(runtime):
    if runtime is None or len(runtime) != 9:
        return None
    return (
        _runtime_key(runtime),
        tuple(
            tuple(
                _view_key(getattr(meta, name, None))
                for name in (
                    "current",
                    "lengths",
                    "block_table",
                    "topk_lengths",
                    "selected",
                    "schedule",
                )
            )
            for meta in runtime[4]
        ),
        tuple(tuple(_view_key(t) for t in pair) for pair in runtime[8])
        if runtime[8] is not None
        else (),
    )


def _integer_tensor(tensor, shape, device, *, wide=False):
    import torch

    return (
        tensor is not None
        and tensor.dtype in ((torch.int32, torch.int64) if wide else (torch.int32,))
        and tensor.shape == shape
        and tensor.device == device
        and tensor.stride(-1) == 1
    )


def _layer_compact(query, starts, regions, heads, mode):
    import torch

    return (
        mode == "v2"
        and tuple(torch.cuda.get_device_capability(query.device)) == (8, 9)
        and heads in (4, 8)
        and starts[-1] >= 256
        and any(b - a >= 64 // heads for a, b in zip(starts, starts[1:], strict=False))
        and query.stride(2) == 256
        and regions > 512
    )


def layer_contract(metadata, runtime, query, weights, groups, real_heads):
    """Validate every chunk before uploading any consolidated metadata."""
    from vllm_sail import envs

    rows = metadata.num_actual_tokens
    if runtime is None or len(runtime) != 9 or not runtime[4]:
        return "runtime", None
    regions = runtime[2]
    if not isinstance(regions, int) or not 0 < regions <= 2147483647:
        return "regions", None
    budget = (
        0
        if envs.VLLM_SAIL_STEP5_DSA_SCORES == "v3"
        else envs.VLLM_SAIL_STEP5_DSA_SCORE_MAX_BYTES
    )
    if budget == 0 or rows * groups * regions * 4 > budget:
        return "budget", None
    import torch

    if (
        query is None
        or weights is None
        or query.ndim != 4
        or weights.ndim != 3
        or query.shape[0] < rows
        or query.shape[1] != groups
        or groups < 1
        or query.shape[2] not in (4, 8, 16)
        or query.shape[3] != 256
        or weights.shape != query.shape[:3]
    ):
        return "query_geometry", None
    if (
        envs.VLLM_STEP5_INDEXER_FP8_AS_FP16
        or query.dtype != torch.float8_e4m3fn
        or weights.dtype != torch.float32
    ):
        return "dtype", None
    device = query.device
    if (
        not query.is_cuda
        or weights.device != device
        or metadata.slot_mapping.device != device
    ):
        return "device", None
    if tuple(torch.cuda.get_device_capability(device)) < (8, 9):
        return "capability", None
    if (
        any(s <= 0 for s in query.stride())
        or any(s <= 0 for s in weights.stride())
        or query.stride(3) != 1
    ):
        return "layout", None
    if getattr(metadata.query_start_loc_cpu, "is_cuda", False):
        return "host_starts", None
    starts = metadata.query_start_loc_cpu[: metadata.num_actual_reqs + 1].tolist()
    sizes = tuple(meta.current.numel() for meta in runtime[4])
    paired = runtime[8] is not None
    try:
        host = packed_spans(starts, sizes, 16 if paired else 8)
    except ValueError:
        return "boundaries", None
    if len(starts) != metadata.num_actual_reqs + 1 or starts[-1] != rows:
        return "boundaries", None
    if paired and (
        real_heads != 4
        or len(runtime[8]) != len(sizes)
        or any(n % 2 for n in (*starts, *sizes))
    ):
        return "pairs", None
    heads = 4 if paired else query.shape[2]
    if (rows * groups - 1) * regions > 2147483647 and not _layer_compact(
        query, starts, regions, heads, envs.VLLM_SAIL_STEP5_DSA_SCORES
    ):
        # Preserve the resident kernel's signed row * REGIONS arithmetic.
        return "resident_index_range", None
    pages = ((regions + 127) // 128) * 4
    if not _integer_tensor(runtime[0], (rows,), device, wide=True):
        return "owners", None
    for index, size in enumerate(sizes):
        meta = runtime[4][index]
        if (
            getattr(meta, "selected", None) is not None
            or not _integer_tensor(meta.current, (size,), device, wide=True)
            or not _integer_tensor(meta.lengths, (size,), device)
            or not _integer_tensor(meta.topk_lengths, (size,), device)
            or not _integer_tensor(meta.block_table, (size, pages), device)
        ):
            return "chunk_metadata", None
        if paired:
            pair = runtime[8][index]
            if (
                len(pair) != 3
                or not _integer_tensor(pair[0], (size // 2,), device)
                or not _integer_tensor(pair[1], (size // 2, pages), device)
            ):
                return "pairs", None
    return None, (tuple(starts), host)


class LayerScorePlan:
    """Step-local immutable metadata; never owns layer Q, packed K or scores."""

    def __init__(self, runtime, starts, host, query, groups, mode):
        import torch

        self.rows, self.groups, self.regions = starts[-1], groups, runtime[2]
        self.next_n = 2 if runtime[8] is not None else 1
        self.heads = 4 if self.next_n == 2 else query.shape[2]
        self.group_rows = 64 // self.heads
        self.compact = _layer_compact(query, starts, self.regions, self.heads, mode)
        self.key = (
            _layer_runtime_key(runtime),
            groups,
            query.shape[2],
            self.next_n,
            mode,
            query.device,
            starts,
            query.stride(2),
        )
        lengths = (
            [pair[0] for pair in runtime[8]]
            if self.next_n == 2
            else [meta.lengths for meta in runtime[4]]
        )
        self.lengths = lengths[0] if len(lengths) == 1 else torch.cat(lengths)
        self.spans = self.table = None
        self.chunks = tuple((begin, end) for begin, end, _ in host)
        if self.compact:
            host = packed_spans(
                starts, [b - a for a, b in self.chunks], self.group_rows
            )
            flat = [
                (begin + first, count, owner)
                for begin, _, spans in host
                for first, count, owner in spans
            ]
            self.spans = (
                torch.tensor(flat, dtype=torch.int32)
                .reshape(-1, 3)
                .pin_memory()
                .to(query.device, non_blocking=True)
            )
        else:
            tables = (
                [pair[1] for pair in runtime[8]]
                if self.next_n == 2
                else [meta.block_table for meta in runtime[4]]
            )
            self.table = tables[0] if len(tables) == 1 else torch.cat(tables)
            if groups > 1:
                self.lengths = self.lengths.repeat(groups)
                self.table = self.table.repeat(groups, 1)


class LayerScoreContext:
    def __init__(self, impl, metadata, query, weights, groups):
        from vllm_sail import envs

        self.impl, self.metadata = impl, metadata
        self.adapter, self.query, self.weights = impl.adapter, query, weights
        self.groups, self.mode = groups, envs.VLLM_SAIL_STEP5_DSA_SCORES
        self.plan = self.runtime = self.runtime_key = self.contract = None
        self.packed = self.pack_key = self.scores = None
        self.group, self.cursor = -1, 0
        self.rejected = None

    def reject(self, reason):
        if self.scores is not None:
            raise RuntimeError(
                f"Step5 layer score contract changed after launch: {reason}"
            )
        if self.rejected is None:
            self.rejected = reason
            _STATS["layer_reject_" + reason] += 1
            if reason == "budget" and reason not in _LAYER_LOGGED:
                from vllm.logger import init_logger

                init_logger(__name__).info(
                    "Step5 layer scores exceed SCORE_MAX_BYTES (or budget is zero); "
                    "falling back to the original 128 MiB group/chunk scoring path."
                )
                _LAYER_LOGGED.add(reason)
        return False

    def enter_group(self, metadata, query, weights):
        if self.rejected is not None:
            return False
        runtime = metadata.step5_runtime
        if (
            self.group >= 0
            and self.plan is not None
            and self.cursor != len(self.plan.chunks)
        ):
            return self.reject("incomplete_slices")
        if metadata is not self.metadata or self.group + 1 >= self.groups:
            return self.reject("group_mapping")
        if self.runtime is not None and _layer_runtime_key(runtime) != self.runtime_key:
            if self.scores is not None:
                return self.reject("runtime")
            self.plan = self.contract = None
            self.packed = self.pack_key = None
        reason, contract = layer_contract(
            metadata,
            runtime,
            self.query,
            self.weights,
            self.groups,
            self.impl.indexer_heads,
        )
        if reason is not None:
            return self.reject(reason)
        rows = metadata.num_actual_tokens
        group = self.group + 1
        if not _same_view(
            query, self.query[:rows, group : group + 1]
        ) or not _same_view(weights, self.weights[:rows, group : group + 1]):
            return self.reject("group_view")
        self.runtime, self.runtime_key, self.contract = (
            runtime,
            _layer_runtime_key(runtime),
            contract,
        )
        self.group, self.cursor = group, 0
        return True

    def pack(self, original, adapter, mean, table, regions):
        if self.rejected is not None or adapter is not self.adapter:
            return original(adapter, mean, table, regions)
        expected_table = self.metadata.block_table[: self.metadata.num_actual_reqs]
        if (
            self.group < 0
            or mean is not self.impl.summary_cache.mean
            or not _same_view(table, expected_table)
            or regions != self.runtime[2]
        ):
            self.reject("summary_mapping")
            return original(adapter, mean, table, regions)
        import torch

        if (
            mean.ndim != 3
            or mean.shape[1] not in (2, 4, 8)
            or mean.shape[2] != 256
            or mean.dtype != torch.uint8
            or mean.device != self.query.device
            or mean.stride(1) != 256
            or mean.stride(2) != 1
            or mean.stride(0) < mean.shape[1] * 256
            or table.ndim != 2
            or table.dtype != torch.int32
            or table.device != mean.device
            or table.stride(1) != 1
            or table.shape[1] * mean.shape[1] < regions
        ):
            self.reject("summary_contract")
            return original(adapter, mean, table, regions)
        key = (_view_key(mean), table.shape, table.stride(), table.data_ptr(), regions)
        if self.packed is not None and key != self.pack_key:
            self.reject("summary_changed")
            return original(adapter, mean, table, regions)
        if self.packed is None:
            self.packed = original(adapter, mean, table, regions)
            self.pack_key = key
            _STATS["layer_packs"] += 1
        return self.packed

    def logits(
        self,
        adapter,
        query,
        weights,
        packed,
        pages,
        requests,
        lengths,
        regions,
        *,
        pairs=None,
        paired_heads=0,
        block_table=None,
        **kwargs,
    ):
        if self.rejected is not None or adapter is not self.adapter:
            return None
        runtime = self.metadata.step5_runtime
        runtime_key = _layer_runtime_key(runtime)
        if runtime_key != self.runtime_key and self.scores is not None:
            self.reject("runtime")
            return None
        if self.scores is None:
            reason, contract = layer_contract(
                self.metadata,
                runtime,
                self.query,
                self.weights,
                self.groups,
                self.impl.indexer_heads,
            )
            if reason is not None:
                self.reject(reason)
                return None
            if runtime_key != self.runtime_key:
                self.plan = None
            self.runtime, self.runtime_key, self.contract = (
                runtime,
                runtime_key,
                contract,
            )
        starts, host = self.contract
        if self.cursor >= len(host):
            self.reject("extra_slice")
            return None
        index = self.cursor
        begin, end, _ = host[index]
        meta = runtime[4][index]
        expected_pairs = runtime[8][index] if runtime[8] is not None else None
        if (
            self.packed is None
            or packed is not self.packed[0]
            or pages != self.packed[1]
            or regions != runtime[2]
            or regions != self.pack_key[-1]
            or pairs is not expected_pairs
            or (pairs is not None and paired_heads != 4)
            or not _same_view(query, self.query[begin:end, self.group : self.group + 1])
            or not _same_view(
                weights, self.weights[begin:end, self.group : self.group + 1]
            )
            or not _same_view(requests, runtime[0][begin:end])
            or lengths is not meta.lengths
            or not _same_view(block_table, meta.block_table)
        ):
            self.reject("slice_contract")
            return None
        import torch

        if (
            pages != ((regions + 127) // 128) * 4
            or packed.dtype != torch.uint8
            or packed.device != self.query.device
            or packed.shape != (len(starts[:-1]) * pages, 32, 1, 260)
            or not packed.is_contiguous()
        ):
            self.reject("packed_contract")
            return None
        if self.scores is None:
            key = (
                self.runtime_key,
                self.groups,
                self.query.shape[2],
                2 if pairs is not None else 1,
                self.mode,
                self.query.device,
                starts,
                self.query.stride(2),
            )
            plan = getattr(self.metadata, _LAYER_ATTRIBUTE, None)
            if plan is None or plan.key != key:
                plan = LayerScorePlan(
                    runtime, *self.contract, self.query, self.groups, self.mode
                )
                setattr(self.metadata, _LAYER_ATTRIBUTE, plan)
                _STATS["layer_plans"] += 1
            self.plan = plan
            from vllm_sail.models.step5.ops.sparse_attention_opt import (
                grouped_paged_logits,
                paged_mqa_logits_sm90,
            )

            rows, heads = plan.rows, plan.heads
            if plan.compact:
                self.scores = grouped_paged_logits(
                    self.query[:rows],
                    self.weights[:rows],
                    packed,
                    plan.lengths,
                    plan.spans,
                    regions,
                    pages,
                    heads,
                    plan.next_n,
                    group_rows=plan.group_rows,
                    groups=self.groups,
                ).view(self.groups, rows, regions)
                _STATS["layer_compact_launches"] += 1
            else:
                # Try a group-major view first. Only Q/weights may be normalized;
                # metadata is already step-local and the shared K is never copied.
                q = self.query[:rows, :, :heads].transpose(0, 1)
                w = self.weights[:rows, :, :heads].transpose(0, 1)
                shape = (self.groups * rows // plan.next_n, plan.next_n, heads, 256)
                try:
                    q = q.view(*shape)
                except RuntimeError:
                    q = q.contiguous().view(*shape)
                    _STATS["layer_query_copies"] += 1
                try:
                    w = w.view(self.groups * rows, heads)
                except RuntimeError:
                    w = w.contiguous().view(self.groups * rows, heads)
                    _STATS["layer_weight_copies"] += 1
                self.scores = paged_mqa_logits_sm90(
                    q, packed, w, plan.table, plan.lengths, regions
                ).view(self.groups, rows, regions)
                _STATS["layer_resident_launches"] += 1
            _STATS["layer_launches"] += 1
        self.cursor += 1
        _STATS["layer_slices"] += 1
        return self.scores[self.group, begin:end]

    def release(self):
        self.impl = self.metadata = self.adapter = self.query = self.weights = None
        self.plan = self.runtime = self.runtime_key = self.contract = None
        self.packed = self.pack_key = self.scores = None


@contextmanager
def bind_layer(impl, metadata, query, weights, groups):
    c = LayerScoreContext(impl, metadata, query, weights, groups)
    token = _LAYER_CONTEXT.set(c)
    try:
        yield c
    finally:
        _LAYER_CONTEXT.reset(token)
        c.release()


def wrap_layer_forward(original, platform):
    @functools.wraps(original, updated=())
    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
        dsa_proxy_query=None,
        dsa_proxy_key=None,
        dsa_proxy_weights=None,
        dsa_proxy_z=None,
        dsa_summary_updated=False,
        defer_inactive_clear=False,
        dsa_gate=None,
    ):
        args = (
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
            dsa_proxy_query,
            dsa_proxy_key,
            dsa_proxy_weights,
            dsa_proxy_z,
            dsa_summary_updated,
            defer_inactive_clear,
            dsa_gate,
        )
        if _LAYER_CONTEXT.get() is not None:
            token = _LAYER_CONTEXT.set(None)
            try:
                return forward(*args)
            finally:
                _LAYER_CONTEXT.reset(token)
        if not layer_enabled() or not platform.is_ppu() or _EXPERIMENT.get():
            return original(*args)
        # Gate recursion creates exactly one scoring lifetime in the ungated call.
        if (
            attn_metadata is None
            or not attn_metadata.num_actual_tokens
            or dsa_gate is not None
            or self.adapter is None
            or self.summary_cache is None
            or (
                attn_metadata.num_actual_tokens
                == attn_metadata.num_actual_reqs * attn_metadata.max_query_len
                and attn_metadata.max_query_len <= self.num_speculative_tokens + 1
            )
        ):
            return original(*args)
        with bind_layer(
            self, attn_metadata, dsa_proxy_query, dsa_proxy_weights, kv_cache.shape[1]
        ) as context:
            result = original(*args)
            if context.scores is not None and (
                context.group + 1 != context.groups
                or context.cursor != len(context.plan.chunks)
            ):
                raise RuntimeError("Step5 layer scores were not completely consumed")
            return result

    return forward


def wrap_layer_pack(original):
    @functools.wraps(original, updated=())
    def pack(self, mean, block_table, regions):
        c = _LAYER_CONTEXT.get()
        if c is not None:
            return c.pack(original, self, mean, block_table, regions)
        return original(self, mean, block_table, regions)

    return pack


def compact_enabled():
    from vllm_sail import envs

    return envs.VLLM_SAIL_STEP5_DSA_SCORES in ("v2", "v3")


def grouped_enabled():
    from vllm_sail import envs

    return envs.VLLM_SAIL_STEP5_DSA_SCORES == "v3"


def packed_spans(starts, chunk_rows, group_rows):
    """Split by request and chunk; tail groups keep only real rows, owner 0 is valid."""
    if group_rows not in (4, 8, 16) or isinstance(group_rows, bool):
        raise ValueError("unsupported grouped rows")
    starts, chunk_rows = tuple(starts), tuple(chunk_rows)
    if (
        not starts
        or starts[0] != 0
        or any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in starts)
        or any(b < a for a, b in zip(starts, starts[1:], strict=False))
        or any(
            not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in chunk_rows
        )
        or sum(chunk_rows) != starts[-1]
    ):
        raise ValueError("incomplete request/chunk boundaries")
    chunks, begin = [], 0
    for size in chunk_rows:
        end, spans = begin + size, []
        for owner, (left, right) in enumerate(zip(starts, starts[1:], strict=False)):
            left, right = max(left, begin), min(right, end)
            while left < right:
                count = min(group_rows, right - left)
                spans.append((left - begin, count, owner))
                left += count
        chunks.append((begin, end, tuple(spans)))
        begin = end
    return tuple(chunks)


class PackedScorePlan:
    def __init__(self, runtime, starts, heads, device, *, group_rows=None):
        import torch

        if heads not in (4, 8, 16) or isinstance(heads, bool):
            raise ValueError("unsupported grouped heads")
        self.heads = heads
        self.group_rows = 64 // heads if group_rows is None else group_rows
        self.runtime_key = _runtime_key(runtime)
        self.requests = len(starts) - 1
        self.host = packed_spans(
            starts, [m.current.numel() for m in runtime[4]], self.group_rows
        )
        flat = [span for _, _, spans in self.host for span in spans]
        self.spans = (
            torch.tensor(flat, dtype=torch.int32)
            .reshape(-1, 3)
            .pin_memory()
            .to(device, non_blocking=True)
        )
        self.chunks, cursor = [], 0
        for begin, end, spans in self.host:
            self.chunks.append((begin, end, self.spans[cursor : cursor + len(spans)]))
            cursor += len(spans)


def attach_packed_plan(metadata, runtime, ppu, heads, *, group_rows=None):
    """No extra upload when disabled; per-step metadata is reused across layers, never K values."""
    if not ppu or not compact_enabled():
        return runtime
    from vllm_sail import envs

    setattr(metadata, _PACKED_ATTRIBUTE, None)
    if envs.VLLM_STEP5_INDEXER_FP8_AS_FP16:
        return runtime
    if (
        runtime is None
        or not runtime[4]
        or not 512 < runtime[2] <= 2147483647
        or heads not in (4, 8, 16)
        or (heads == 16 and not _EXPERIMENT.get())
        or getattr(runtime[4][0], "selected", None) is not None
    ):
        return runtime
    sizes = [m.current.numel() for m in runtime[4]]
    if max(sizes) < 256 or getattr(metadata.query_start_loc_cpu, "is_cuda", False):
        return runtime
    import torch

    if tuple(torch.cuda.get_device_capability(metadata.slot_mapping.device)) != (8, 9):
        return runtime
    starts = metadata.query_start_loc_cpu[: metadata.num_actual_reqs + 1].tolist()
    if (
        len(starts) != metadata.num_actual_reqs + 1
        or starts[-1] != sum(sizes)
        or (
            runtime[8] is not None
            and (
                len(runtime[8]) != len(sizes)
                or any(n % 2 for n in starts)
                or any(n % 2 for n in sizes)
            )
        )
    ):
        return runtime
    rows = 64 // heads if group_rows is None else group_rows
    if not any(b - a >= rows for a, b in zip(starts, starts[1:], strict=False)):
        return runtime
    plan = PackedScorePlan(
        runtime, starts, heads, metadata.slot_mapping.device, group_rows=group_rows
    )
    setattr(metadata, _PACKED_ATTRIBUTE, plan)
    return runtime


def _same_view(actual, expected):
    return (
        actual is not None
        and expected is not None
        and actual.shape == expected.shape
        and actual.stride() == expected.stride()
        and actual.device == expected.device
        and actual.dtype == expected.dtype
        and actual.data_ptr() == expected.data_ptr()
    )


@contextmanager
def bind_packed(metadata, runtime, adapter, query, weights, ppu):
    """Bind views for this forward only; the outer context is restored even on error."""
    context = None
    if ppu and compact_enabled():
        plan = getattr(metadata, _PACKED_ATTRIBUTE, None)
        if (
            plan is not None
            and runtime is not None
            and plan.runtime_key == _runtime_key(runtime)
        ):
            import torch

            rows = plan.host[-1][1]
            if (
                query is not None
                and weights is not None
                and plan.host
                and query.ndim == 4
                and weights.ndim == 3
                and query.shape[0] >= rows
                and query.shape[1] == 1
                and query.shape[-1] == 256
                and (
                    query.shape[2] in (8, 16)
                    or (query.shape[2] == 4 and grouped_enabled())
                )
                and weights.shape == query.shape[:3]
                and query.dtype == torch.float8_e4m3fn
                and weights.dtype == torch.float32
                and query.is_cuda
                and query.device == weights.device == plan.spans.device
                and query.stride(3) == 1
                and query.stride(2) == 256
                and query.stride(0) >= query.shape[2] * 256
                and weights.stride(2) == 1
                and weights.stride(0) >= query.shape[2]
                and plan.heads == (4 if runtime[8] is not None else query.shape[2])
            ):
                context = PackedScoreContext(plan, runtime, adapter, query, weights)
    token = _PACKED_CONTEXT.set(context)
    try:
        yield context
    finally:
        _PACKED_CONTEXT.reset(token)
        if context is not None:
            context.release()


class PackedScoreContext:
    def __init__(self, plan, runtime, adapter, query, weights):
        self.plan, self.runtime, self.adapter = plan, runtime, adapter
        self.query, self.weights, self.cursor = query, weights, 0

    def release(self):
        self.plan = self.runtime = self.adapter = self.query = self.weights = None


def packed_logits(
    adapter,
    query,
    weights,
    packed,
    pages,
    requests,
    lengths,
    regions,
    *,
    pairs=None,
    paired_heads=0,
    block_table=None,
    config=None,
):
    """Returning None falls back before launch; device/JIT exceptions propagate."""
    c = _PACKED_CONTEXT.get()
    if not compact_enabled() or c is None or c.adapter is not adapter:
        return None
    if c.query.shape[2] == 4 and not grouped_enabled():
        return None
    if c.plan.runtime_key != _runtime_key(c.runtime) or c.cursor >= len(c.plan.chunks):
        return None
    index = c.cursor
    c.cursor += 1
    begin, end, spans = c.plan.chunks[index]
    host_spans = c.plan.host[index][2]
    if (
        end - begin < 256
        or regions != c.runtime[2]
        or (end - begin) * regions * 4 > 128 << 20
        or not any(count == c.plan.group_rows for _, count, _ in host_spans)
    ):
        return None
    import torch

    paired = c.runtime[8] is not None
    expected_pairs = c.runtime[8][index] if paired else None
    if (paired and (pairs is not expected_pairs or paired_heads != 4)) or (
        not paired and pairs is not None
    ):
        return None
    effective = pairs[0] if paired else lengths
    expected = expected_pairs[0] if paired else c.runtime[4][index].lengths
    if block_table is not None and not _same_view(
        block_table, c.runtime[4][index].block_table
    ):
        return None
    if (
        not _same_view(query, c.query[begin:end])
        or not _same_view(weights, c.weights[begin:end])
        or not _same_view(requests, c.runtime[0][begin:end])
        or effective is not expected
        or lengths is not c.runtime[4][index].lengths
        or effective.ndim != 1
        or effective.dtype != torch.int32
        or effective.device != query.device
        or effective.stride(0) != 1
        or effective.numel() != (end - begin) // (2 if paired else 1)
        or pages != ((regions + 127) // 128) * 4
        or packed.dtype != torch.uint8
        or packed.device != query.device
        or packed.shape != (c.plan.requests * pages, 32, 1, 260)
        or not packed.is_contiguous()
    ):
        return None
    from vllm_sail.models.step5.ops.sparse_attention_opt import grouped_paged_logits

    result = grouped_paged_logits(
        query,
        weights,
        packed,
        effective,
        spans,
        regions,
        pages,
        c.plan.heads,
        2 if paired else 1,
        group_rows=c.plan.group_rows,
        **(config or {}),
    )
    _STATS["compact_chunks"] += 1
    _STATS["compact_spans"] += spans.shape[0]
    return result


def wrap_packed_plan(original, platform):
    @functools.wraps(original, updated=())
    def planned(self, attn_metadata, n, lengths, layout, positions=None):
        runtime = original(self, attn_metadata, n, lengths, layout, positions)
        if _LAYER_CONTEXT.get() is not None:
            return runtime
        # Production outer-forward admission is lazy: do not upload a legacy
        # compact plan before the complete Q/K and score-output budget checks.
        if (
            platform.is_ppu()
            and (layer_enabled() or grouped_enabled())
            and not _EXPERIMENT.get()
            and getattr(layout, "groups", None) is not None
        ):
            return runtime
        if platform.is_ppu() and compact_enabled():
            heads = (
                4
                if runtime is not None and runtime[8] is not None
                else max(self.indexer_heads, 8)
            )
            attach_packed_plan(attn_metadata, runtime, True, heads)
        return runtime

    return planned


def wrap_packed_forward(original, platform):
    @functools.wraps(original, updated=())
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
        c = _LAYER_CONTEXT.get()
        if c is not None and c.adapter is self.adapter:
            if c.enter_group(attn_metadata, dsa_proxy_query, dsa_proxy_weights):
                return original(*args, **kwargs)
        if not platform.is_ppu() or not compact_enabled():
            return original(*args, **kwargs)
        runtime = attn_metadata.step5_runtime
        plan = getattr(attn_metadata, _PACKED_ATTRIBUTE, None)
        if runtime is not None:
            heads = 4 if runtime[8] is not None else dsa_proxy_query.shape[2]
            if (
                plan is None
                or plan.runtime_key != _runtime_key(runtime)
                or (grouped_enabled() and plan.heads != heads)
            ):
                attach_packed_plan(attn_metadata, runtime, True, heads)
        with bind_packed(
            attn_metadata,
            runtime,
            self.adapter,
            dsa_proxy_query,
            dsa_proxy_weights,
            True,
        ):
            return original(*args, **kwargs)

    return forward


def wrap_packed_logits(original, platform=None):
    @functools.wraps(original, updated=())
    def logits(
        self, query, weights, packed, pages, requests, lengths, regions, **kwargs
    ):
        c = _LAYER_CONTEXT.get()
        if c is not None:
            result = c.logits(
                self,
                query,
                weights,
                packed,
                pages,
                requests,
                lengths,
                regions,
                **kwargs,
            )
            if result is not None:
                return result
        selected_platform = platform
        if compact_enabled() and selected_platform is None:
            from vllm.platforms import current_platform

            selected_platform = current_platform
        if compact_enabled() and selected_platform.is_ppu():
            result = packed_logits(
                self,
                query,
                weights,
                packed,
                pages,
                requests,
                lengths,
                regions,
                pairs=kwargs.get("pairs"),
                paired_heads=kwargs.get("paired_heads", 0),
                block_table=kwargs.get("block_table"),
            )
            if result is not None:
                return result
        return original(
            self, query, weights, packed, pages, requests, lengths, regions, **kwargs
        )

    return logits


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
