# SPDX-License-Identifier: Apache-2.0
"""20261110 BF16 paged DSA: prebuilt ops only, no device reads and no JIT."""

from __future__ import annotations

import logging
import math
import sys
from collections import Counter

import torch

from vllm_sail.step5_sparse_cases import profiled

_REJECTS = Counter()
_SUBMISSIONS = Counter()
_ROUTE_LOGGED = set()
logger = logging.getLogger(__name__)
_INT_MAX = 2147483647


def compiled_op(kind):
    # torch.ops._ppu_C is reached only through _custom_ops (its extension import
    # registers the namespace).
    from vllm_sail import _custom_ops

    return _custom_ops.step5_dsa_native_op(kind)


def _submit_attention(op, kind, *args):
    from vllm_sail import envs

    if not envs.VLLM_SAIL_NVTX_PROFILE:
        return op(*args)
    from vllm_sail.profiling.step5_sparse import trace_call, trace_range

    names = (
        (
            "query",
            "kv_cache",
            "indices",
            "block_table",
            "positions",
            "active",
            "output",
            "scale",
            "query_len",
            "clear_inactive",
            "gate",
        )
        if kind == "decode"
        else (
            "query",
            "kv_cache",
            "indices",
            "block_table",
            "owners",
            "positions",
            "tiles",
            "output",
            "max_seq_len",
            "scale",
            "query_tile",
        )
    )
    values = dict(zip(names, args, strict=False))
    impl = "step5_dsa_native_" + kind
    with trace_range("attention", impl, "native_20261110", values, pd_type=kind):
        return trace_call(
            "attention",
            impl,
            "native_20261110",
            values,
            lambda: op(*args),
            scope="kernel",
            pd_type=kind,
        )


def reject(reason):
    _REJECTS[reason] += 1


def stats():
    return {
        "implementation": "native_20261110",
        "rejects": dict(_REJECTS),
        "submissions": dict(_SUBMISSIONS),
        "counter_scope": "host submissions only, not device hits",
    }


def cache_context(cache):
    """Read host geometry only; offsets are in BF16 elements, not bytes."""
    context = {
        "cache_shape": tuple(cache.shape),
        "cache_stride": tuple(cache.stride(i) for i in range(cache.ndim)),
        "cache_dtype": str(cache.dtype),
    }
    if cache.ndim == 4:
        last = (
            (cache.shape[0] - 1) * cache.stride(0)
            + (cache.shape[1] - 1) * cache.stride(1)
            + (cache.shape[2] - 8) * cache.stride(2)
        )
        context.update(
            max_region_offset_elements=last,
            r10_offset_limit_elements=2**34,
            physical_regions=cache.shape[0] * cache.shape[2] // 8,
            physical_region_limit=2**28,
            address_encoding="physical_region_28b_int64_stride",
        )
    return context


def record_submission(kind):
    """Count only after the op returns.

    Async device completion and actual kernel hits still need a trace to confirm.
    """
    _SUBMISSIONS[kind] += 1


def record_route(kind, branch, reason, stage, context=None):
    """Log each decision once; no device reads, and config is never a device hit."""
    key = (kind, branch, reason, stage)
    if key in _ROUTE_LOGGED:
        return
    _ROUTE_LOGGED.add(key)
    package = sys.modules.get("vllm_sail")
    extension = sys.modules.get("vllm_sail._C")
    route = dict(context or {})
    route.update(
        kind=kind,
        requested="native",
        branch=branch,
        reason=reason,
        stage=stage,
        implementation="native_20261110",
        source=__file__,
        version=getattr(package, "__version__", None),
        extension=getattr(extension, "__file__", None),
        counter_scope="host routing only, not device hits",
    )
    emit = logger.warning if branch != "native_20261110" else logger.info
    emit("Step5 DSA native route (not a device hit): %s", route)


def device_reason(device):
    if torch.device(device).type != "cuda":
        return "device"
    try:
        if tuple(torch.cuda.get_device_capability(device)) != (
            8,
            9,
        ) or "890P" not in torch.cuda.get_device_name(device):
            return "non_890p"
    except (RuntimeError, AssertionError):
        return "device_properties"
    return None


def valid_scale(scale):
    return (
        math.isfinite(float(scale)) and 2**-150 < float(scale) <= 3.4028234663852886e38
    )


def request_spans(starts, begin, end):
    for request, (left, right) in enumerate(zip(starts, starts[1:], strict=False)):
        left, right = max(begin, left), min(end, right)
        if left < right:
            yield request, left, right


def candidate_reason(shape, dtype, device, block, width, scale):
    if (
        len(shape) != 3
        or shape[0] < 0
        or shape[1] not in (8, 16)
        or shape[2] != 192
        or shape[0] * shape[1] * 32 > _INT_MAX
        or block not in (16, 32, 64, 128)
        or not 1 <= width <= 513
    ):
        return "geometry"
    if dtype != torch.bfloat16:
        return "bf16"
    if not valid_scale(scale):
        return "scale"
    reason = device_reason(device)
    if reason is not None:
        return reason
    return "missing_op" if compiled_op("prefill") is None else None


def plan_reason(metadata, n, lengths, layout, positions, slots, scale):
    starts = metadata.query_start_loc_cpu[: metadata.num_actual_reqs + 1].tolist()
    if (
        not starts
        or starts[0] != 0
        or starts[-1] != n
        or len(starts) != len(lengths) + 1
        or any(b < a for a, b in zip(starts, starts[1:], strict=False))
    ):
        return "request_starts"
    if layout.quantized or layout.groups < 1:
        return "layout"
    if any(
        not b - a <= int(seq) <= _INT_MAX - 8
        for a, b, seq in zip(starts, starts[1:], lengths, strict=False)
    ):
        return "context_capacity"
    return candidate_reason(
        (n, layout.heads_per_group, layout.head_size),
        layout.dtype,
        metadata.slot_mapping.device,
        layout.block_size,
        slots,
        scale,
    )


def _tensor_reason(t, q, dtype, rank, name):
    """Check the named input contract first so shape/stride never index a bad rank."""
    if not t.is_cuda or t.device != q.device:
        return name + "_device"
    if t.ndim != rank:
        return name + "_rank"
    if t.dtype != dtype:
        return name + "_dtype"
    if t.layout != torch.strided:
        return name + "_layout"
    return None


def _tensor_capacity_reason(t, name):
    """int64 address checks added by the port; not in the reference dtype/rank table."""
    span = 1
    for axis, size in enumerate(t.shape):
        stride = t.stride(axis)
        if (
            size > _INT_MAX
            or stride < 0
            or stride > (2**63 - 1) // 2
            or (size > 1 and stride <= 0)
        ):
            return name + "_stride_capacity"
        span += max(0, size - 1) * stride
    return name + "_address_capacity" if span > (2**63 - 1) // 8 else None


def reference_reason(q, cache, regions, table, out, *, width=None):
    """Mirror _supports_inputs/supports_paged_attention of 20261110/ppu_sparse.py.

    positions is not in the reference public input table; regions=None is only for
    the bind precheck before the selector. Strided CUDA tensors are a precondition
    of the current prebuilt ops; the FP8 Triton fallback is not native support.
    """
    tensors = [
        ("query", q, torch.bfloat16, 3),
        ("output", out, torch.bfloat16, 3),
        ("cache", cache, torch.bfloat16, 4),
        ("table", table, torch.int32, 2),
    ]
    if regions is not None:
        tensors.append(("regions", regions, torch.int64, 2))
    for name, t, dtype, rank in tensors:
        reason = _tensor_reason(t, q, dtype, rank, name)
        if reason:
            return reason
    rows, heads, dim = q.shape
    if dim != 192 or heads not in (8, 16) or out.shape != q.shape:
        return "head_output_shape"
    if (
        cache.shape[0] < 1
        or cache.shape[1] < 1
        or cache.shape[3] != 384
        or cache.shape[2] not in (16, 32, 64, 128)
    ):
        return "cache_shape"
    if regions is not None:
        width = regions.shape[1]
        if regions.shape[0] != rows or regions.stride(1) != 1:
            return "regions"
    if width is None or not 1 <= width <= 513:
        return "region_width"
    if table.stride(1) != 1:
        return "table_capacity"
    if (
        cache.data_ptr() % 128
        or cache.stride(3) != 1
        or any(cache.stride(i) % 64 for i in range(3))
    ):
        return "cache_alignment"
    if cache.shape[0] * cache.shape[2] // 8 >= 2**28:
        return "offset_capacity"
    for t in (q, out):
        if t.stride(2) != 1 or t.stride(0) % 2 or t.stride(1) % 2 or t.data_ptr() % 4:
            return "query_output_alignment"
    return device_reason(q.device)


def common_reason(q, cache, regions, table, pos, out, scale, *, width=None):
    """Rejection model: reference admission contract + normalized positions +
    current kernel safety checks.
    """
    reason = reference_reason(q, cache, regions, table, out, width=width)
    if reason:
        return reason
    # The reference prefill converts to int64/contiguous before launch; this
    # adapter's fixup_positions already did. Decode, like the reference
    # try_decode_attention, requires int64/contiguous directly.
    reason = _tensor_reason(pos, q, torch.int64, 1, "positions")
    if reason:
        return reason
    tensors = [
        ("query", q),
        ("output", out),
        ("cache", cache),
        ("table", table),
        ("positions", pos),
    ]
    if regions is not None:
        tensors.append(("regions", regions))
    for name, t in tensors:
        reason = _tensor_capacity_reason(t, name)
        if reason:
            return reason
    rows, heads, _ = q.shape
    if pos.numel() != rows or not pos.is_contiguous():
        return "positions"
    if (
        table.shape[0] < 1
        or table.shape[1] < 1
        or table.stride(0) < table.shape[1]
        or table.shape[1] * cache.shape[2] > _INT_MAX - 8
    ):
        return "table_capacity"
    # A single group's stride is not used for addressing; for LBNHC/BLNHC that
    # stride may be just 384.
    if (
        cache.stride(2) < 384
        or (cache.shape[1] > 1 and cache.stride(1) < cache.shape[2] * cache.stride(2))
        or cache.stride(0)
        < (cache.shape[1] - 1) * cache.stride(1) + cache.shape[2] * cache.stride(2)
    ):
        return "cache_overlap"
    # Tags store the physical region and addresses are rebuilt from the int64
    # stride; the capacity check does not restore the old 2**34 offset limit.
    if out.numel() and (
        out.stride(1) < 192 or out.stride(0) < (heads - 1) * out.stride(1) + 192
    ):
        return "output_overlap"
    inputs = (q, cache, table, pos) + ((regions,) if regions is not None else ())
    if any(torch._C._overlaps(out, t) for t in inputs):
        return "output_alias"
    if rows * heads * 32 > _INT_MAX or not valid_scale(scale):
        return "launch_scale"
    return None


def prefill_reason(
    q,
    cache,
    regions,
    table,
    owners,
    pos,
    tiles,
    out,
    logical_regions,
    scale,
    query_tile=0,
    *,
    width=None,
):
    reason = common_reason(q, cache, regions, table, pos, out, scale, width=width)
    if reason:
        return reason
    if (
        cache.shape[1] != 1
        or q.stride(1) != 192
        or q.stride(0) % 64
        or q.data_ptr() % 128
    ):
        return "prefill_layout"
    for name, t, dtype, rank in (
        ("owners", owners, torch.int64, 1),
        ("tiles", tiles, torch.int32, 2),
    ):
        reason = _tensor_reason(t, q, dtype, rank, name) or _tensor_capacity_reason(
            t, name
        )
        if reason:
            return reason
        if torch._C._overlaps(out, t):
            return "output_alias"
    if (
        owners.numel() != q.shape[0]
        or not owners.is_contiguous()
        or tiles.shape[0] != 2
        or not tiles.is_contiguous()
        or not 0 <= tiles.shape[1] <= q.shape[0]
        or (q.shape[0] and tiles.shape[1] == 0)
    ):
        return "owners_tiles"
    if (
        not 1 <= logical_regions <= _INT_MAX // 8
        or type(query_tile) is not int
        or query_tile not in (0, 1, 2, 4, 8)
    ):
        return "prefill_capacity"
    return None


def decode_reason(
    q, cache, regions, table, pos, active, out, scale, query_len=1, gate=None
):
    reason = common_reason(q, cache, regions, table, pos, out, scale)
    if reason:
        return reason
    reason = _tensor_reason(
        active, q, torch.bool, 1, "active"
    ) or _tensor_capacity_reason(active, "active")
    if reason:
        return reason
    groups = cache.shape[1]
    if (
        type(query_len) is not int
        or not 1 <= query_len <= _INT_MAX // groups
        or q.shape[0] % (query_len * groups)
        or table.shape[0] < q.shape[0] // (query_len * groups)
        or active.numel() != q.shape[0]
        or not active.is_contiguous()
    ):
        return "decode_requests"
    if torch._C._overlaps(out, active):
        return "output_alias"
    if gate is not None:
        reason = _tensor_reason(
            gate, q, torch.bfloat16, 2, "gate"
        ) or _tensor_capacity_reason(gate, "gate")
        if reason:
            return reason
        if gate.shape != (q.shape[0] // groups, groups * q.shape[1]):
            return "gate_shape"
        if torch._C._overlaps(out, gate):
            return "output_alias"
    return None


def request_tiles(starts, begin, end, group_rows=16):
    """Return chunk-local bounds; each tile has at most 16 rows within one request."""
    if group_rows not in (8, 16) or not 0 <= begin <= end <= starts[-1]:
        raise ValueError("invalid chunk/tile boundaries")
    lefts, rights = [], []
    for _, left, right in request_spans(starts, begin, end):
        for first in range(left, right, group_rows):
            lefts.append(first - begin)
            rights.append(min(right, first + group_rows) - begin)
    return [lefts, rights]


class NativePrefill:
    """Prebuilt consumer keeping upstream's nine-field runtime adapter protocol."""

    query_buffer = True

    def __init__(
        self,
        query_shape,
        device,
        host_positions,
        region_count,
        scale,
        query_tile=0,
        *,
        batch_attention=False,
    ):
        if type(query_tile) is not int or query_tile not in (0, 1, 2, 4, 8):
            raise ValueError("query_tile must be one of 0/1/2/4/8")
        self.query_tile = query_tile
        self.query_shape = tuple(query_shape)
        self.region_count = region_count
        self.scale = float(scale)
        self.host_positions = tuple(int(x) for x in host_positions)
        self.positions = None
        self.reference = None
        self.batch_attention = bool(batch_attention)
        self._bound = None
        self._batch = None
        self._finished = False

    def page_metadata(self, *args, **kwargs):
        return None

    def release(self):
        self._bound = None
        self._batch = None
        self._next_start = 0
        self._finished = False

    def fixup_positions(self, positions):
        """Same pre-launch conversion as the reference prefill.

        No device values are read, and int64 inputs are not copied again.
        """
        self.positions = positions.reshape(-1).to(torch.int64).contiguous()

    def bind(self, query, cache, table, starts, output, k_scale, v_scale):
        if self._bound is not None:
            raise RuntimeError("native prefill cannot be bound twice")
        if compiled_op("prefill") is None:
            return "missing_op"
        n = self.query_shape[0]
        if (
            tuple(query.shape) != self.query_shape
            or self.positions is None
            or len(self.host_positions) != n
            or output.ndim != 3
            or output.shape[0] < n
        ):
            return "query_shape"
        if (
            not starts
            or starts[0] != 0
            or starts[-1] != n
            or any(b < a for a, b in zip(starts, starts[1:], strict=False))
            or table.ndim != 2
            or table.shape[0] != len(starts) - 1
        ):
            return "request_starts"
        chunks = tuple(getattr(self, "chunk_rows", (n,)))
        if any(size <= 0 for size in chunks) or sum(chunks) != n:
            return "chunk_rows"
        owners = getattr(self, "owners", None)
        if owners is None:
            return "owners"
        reason = common_reason(
            query,
            cache,
            None,
            table,
            self.positions,
            output[:n],
            self.scale,
            width=self.region_count,
        )
        if reason:
            return reason
        logical = max(1, (max(self.host_positions, default=-1) + 8) // 8)
        if any(p < 0 or p > _INT_MAX - 8 for p in self.host_positions):
            return "context_capacity"
        if max(self.host_positions, default=-1) >= table.shape[1] * cache.shape[2]:
            return "table_capacity"
        group_rows = 8 if n >= 8192 and query.shape[1] == 16 else 16
        # Check the whole group before summary/selector; only a few host tiles are
        # uploaded and query/KV are never moved.
        all_tiles = torch.tensor(
            request_tiles(starts, 0, n, group_rows),
            dtype=torch.int32,
            device=query.device,
        )
        reason = prefill_reason(
            query,
            cache,
            None,
            table,
            owners,
            self.positions,
            all_tiles,
            output[:n],
            logical,
            self.scale,
            self.query_tile,
            width=self.region_count,
        )
        if reason:
            return reason
        tiled, begin = {}, 0
        for size in chunks:
            end = begin + size
            tiles = (
                None
                if self.batch_attention
                else torch.tensor(
                    request_tiles(starts, begin, end, group_rows),
                    dtype=torch.int32,
                    device=query.device,
                )
            )
            tiled[begin] = (end, tiles)
            begin = end
        # Cache only the selections, not query/KV; separate storage keeps the reused
        # selector buffer from overwriting the previous chunk.
        batch = None
        if self.batch_attention:
            regions = torch.empty(
                (n, self.region_count), dtype=torch.int64, device=query.device
            )
            batch = (query, cache, table, owners, regions, all_tiles)
        self._bound = (output[:n], tiled, logical)
        self._batch = batch
        self._next_start = 0
        self._finished = False
        return None

    @staticmethod
    def _same_view(actual, expected):
        return (
            actual.data_ptr() == expected.data_ptr()
            and actual.shape == expected.shape
            and actual.dtype == expected.dtype
            and actual.device == expected.device
            and tuple(actual.stride(i) for i in range(actual.ndim))
            == tuple(expected.stride(i) for i in range(expected.ndim))
        )

    def prepare_chunk(self, start, query, cache, regions, table, owners):
        if (
            self._bound is None
            or self._finished
            or start != self._next_start
            or start not in self._bound[1]
        ):
            raise RuntimeError("native chunk lifecycle error")
        output, tiled, logical = self._bound
        end, tiles = tiled[start]
        if query.shape[0] != end - start or regions.shape != (
            end - start,
            self.region_count,
        ):
            raise ValueError("native chunk geometry mismatch")
        if self.batch_attention:
            full_query, full_cache, full_table, full_owners, selected, _ = self._batch
            # bind already validated the full prefill; a chunk must be a raw view of the
            # same batch.
            if not all(
                self._same_view(actual, expected)
                for actual, expected in (
                    (query, full_query[start:end]),
                    (cache, full_cache),
                    (table, full_table),
                    (owners, full_owners[start:end]),
                )
            ):
                raise ValueError("native chunk inputs do not belong to the bound batch")
            reason = common_reason(
                query,
                cache,
                regions,
                table,
                self.positions[start:end],
                output[start:end],
                self.scale,
            )
            if reason:
                raise ValueError("native selection contract failed: " + reason)
            selected[start:end].copy_(regions)
            self._next_start = end
            return
        reason = prefill_reason(
            query,
            cache,
            regions,
            table,
            owners,
            self.positions[start:end],
            tiles,
            output[start:end],
            logical,
            self.scale,
            self.query_tile,
        )
        if reason:
            raise ValueError("native launch contract failed: " + reason)
        op = compiled_op("prefill")
        if op is None:
            raise RuntimeError("native op became unavailable after binding")
        _submit_attention(
            op,
            "prefill",
            query,
            cache,
            regions,
            table,
            owners,
            self.positions[start:end],
            tiles,
            output[start:end],
            logical,
            self.scale,
            self.query_tile,
        )
        record_submission("prefill")
        self._next_start = end

    def finish(self, query, cache, out):
        if (
            self._bound is None
            or self._finished
            or self._next_start != self.query_shape[0]
            or out.data_ptr() != self._bound[0].data_ptr()
            or tuple(query.shape) != self.query_shape
        ):
            raise RuntimeError("native chunks are incomplete or already finished")
        if self.batch_attention:
            full_query, full_cache, table, owners, regions, tiles = self._batch
            if not all(
                self._same_view(actual, expected)
                for actual, expected in (
                    (query, full_query),
                    (cache, full_cache),
                    (out, self._bound[0]),
                )
            ):
                raise ValueError(
                    "native finish inputs do not belong to the bound batch"
                )
            logical = self._bound[2]
            reason = prefill_reason(
                query,
                cache,
                regions,
                table,
                owners,
                self.positions,
                tiles,
                out,
                logical,
                self.scale,
                self.query_tile,
            )
            if reason:
                raise ValueError("native batched launch contract failed: " + reason)
            op = compiled_op("prefill")
            if op is None:
                raise RuntimeError("native op became unavailable after binding")
            # Never resubmit, even after a failed launch; the outer finally releases.
            self._finished = True
            _submit_attention(
                op,
                "prefill",
                query,
                cache,
                regions,
                table,
                owners,
                self.positions,
                tiles,
                out,
                logical,
                self.scale,
                self.query_tile,
            )
            record_submission("prefill")
        self._finished = True


def wrap_decode(original, backend, platform):
    reference = profiled("attention", backend="upstream_triton", pd_type="decode")(
        original
    )

    def sparse_decode_attention(
        query,
        cache,
        regions,
        block_table,
        positions,
        active,
        output,
        *,
        scale,
        query_len=1,
        splits,
        clear_inactive=True,
        k_scale=None,
        v_scale=None,
        gate=None,
    ):
        if backend() == "native":
            reason = (
                decode_reason(
                    query,
                    cache,
                    regions,
                    block_table,
                    positions,
                    active,
                    output,
                    scale,
                    query_len,
                    gate,
                )
                if platform.is_ppu()
                else "non_ppu"
            )
            op = compiled_op("decode") if reason is None else None
            if reason is None and op is None:
                reason = "missing_op"
            if reason is None:
                record_route("decode", "native_20261110", None, "dispatch")
                _submit_attention(
                    op,
                    "decode",
                    query,
                    cache,
                    regions,
                    block_table,
                    positions,
                    active,
                    output,
                    scale,
                    query_len,
                    clear_inactive,
                    gate,
                )
                record_submission("decode")
                return None
            reject("decode:" + reason)
            record_route("decode", "upstream", reason, "dispatch", cache_context(cache))
        return reference(
            query,
            cache,
            regions,
            block_table,
            positions,
            active,
            output,
            scale=scale,
            query_len=query_len,
            splits=splits,
            clear_inactive=clear_inactive,
            k_scale=k_scale,
            v_scale=v_scale,
            gate=gate,
        )

    return profiled("attention", pd_type="decode")(sparse_decode_attention)
