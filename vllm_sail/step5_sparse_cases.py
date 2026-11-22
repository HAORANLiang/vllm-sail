# SPDX-License-Identifier: Apache-2.0
"""Versioned Step5 sparse op metadata; standard library only."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import math
import re
from pathlib import Path

PREFIX = "[FW_SPARSE] "
OPS = {"score", "topk", "attention"}
DTYPES = {
    "torch.bfloat16": 2,
    "torch.float16": 2,
    "torch.float32": 4,
    "torch.float8_e4m3fn": 1,
    "torch.uint8": 1,
    "torch.int32": 4,
    "torch.int64": 8,
    "torch.bool": 1,
}
TENSORS = {
    "query",
    "weights",
    "mean",
    "packed",
    "scores",
    "indices",
    "current",
    "block_table",
    "lengths",
    "positions",
    "slots",
    "active",
    "output",
    "k_scale",
    "v_scale",
    "gate",
    "schedule_meta",
    "spans",
    "tiles",
    "owners",
    "indptr",
    "pages",
    "topk_lengths",
    "starts",
    "workspace",
    "context_lengths",
    "partial",
    "lse",
    "key",
    "value",
}
INTS = {
    "max_seq_len",
    "row_groups",
    "query_len",
    "next_n",
    "heads",
    "topk",
    "splits",
    "num_warps",
    "tile",
    "workers",
    "groups",
    "block_size",
    "page_stride",
    "group_rows",
    "query_tile",
    "paired_heads",
    "page_count",
    "grid_cap",
    "region_block",
    "page_size",
    "threads",
}
BOOLS = {"sort", "clear_inactive", "capture", "packed_mode"}
FLOATS = {"scale"}
FIXED = {
    "v",
    "op",
    "fw",
    "model",
    "pd_type",
    "scope",
    "impl",
    "backend",
    "requested_backend",
}
ALIASES = {
    "q": "query",
    "w": "weights",
    "kv_fused": "packed",
    "cache": "kv_cache",
    "k": "key",
    "v": "value",
    "paged_kv_cache": "kv_cache",
    "kv_cache": "kv_cache",
    "table": "block_table",
    "block_tables": "block_table",
    "seq_lens": "lengths",
    "out": "output",
    "region_count": "topk",
    "capacity": "max_seq_len",
    "warps": "num_warps",
    "packed_mode": "packed_mode",
    "query_tile": "query_tile",
    "bq": "query_tile",
    "pages": "page_count",
    "backend": "requested_backend",
}
TENSORS.add("kv_cache")


def tensor_metadata(value):
    if value is None:
        return None
    shape = tuple(value.shape)
    return {
        "shape": list(shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
        "offset": value.storage_offset(),
    }


def _vector(raw):
    if not re.fullmatch(r"\(\s*(?:\d+\s*(?:,\s*\d+\s*)*,?\s*)?\)", raw):
        raise ValueError("invalid shape/stride vector")
    values = [int(v) for v in re.findall(r"\d+", raw)]
    if len(values) > 8 or any(v > 2147483647 for v in values):
        raise ValueError("shape/stride exceeds protocol limits")
    return values


def storage_elements(item):
    shape, stride = item["shape"], item["stride"]
    if len(shape) != len(stride) or item["offset"] < 0 or item["offset"] > 2147483647:
        raise ValueError("shape/stride/offset mismatch")
    if len(shape) > 8 or any(
        type(v) is not int or not 0 <= v <= 2147483647 for v in (*shape, *stride)
    ):
        raise ValueError("invalid shape/stride values")
    active = sorted(
        (step, size) for size, step in zip(shape, stride, strict=False) if size > 1
    )
    occupied = 1
    for step, size in active:
        if step < occupied:
            raise ValueError("overlapping or broadcast layout cannot be replayed")
        occupied += (size - 1) * step
    span = (
        item["offset"]
        + 1
        + sum((size - 1) * step for size, step in zip(shape, stride, strict=False))
    )
    if any(size == 0 for size in shape):
        span = item["offset"]
    if span > 1 << 40:
        raise ValueError("tensor storage exceeds protocol limits")
    return span


def make_label(
    op, impl, backend, values, *, pd_type="unknown", scope="op", capture=False
):
    fields = [
        f"{PREFIX}op:{op}",
        "v:1",
        "fw:vllm",
        "model:step5",
        f"pd_type:{pd_type}",
        f"scope:{scope}",
        f"impl:{impl}",
        f"backend:{backend}",
        f"capture:{int(capture)}",
    ]
    normalized = {}
    for key, value in values.items():
        if (
            key in ("paged_kv_cache", "kv_cache")
            and type(value) is tuple
            and len(value) == 2
        ):
            normalized["key"], normalized["value"] = value
            continue
        name = ALIASES.get(key, key)
        if key == "regions":
            name = (
                "indices"
                if getattr(value, "shape", None) is not None
                else "max_seq_len"
            )
        if key == "pages" and getattr(value, "shape", None) is not None:
            name = "pages"
        if key == "paged_prefill" and value is not None:
            for attr in (
                "owners",
                "block_table",
                "indices",
                "indptr",
                "block_size",
                "page_stride",
            ):
                normalized[attr if attr != "indices" else "pages"] = getattr(
                    value, attr, None
                )
        elif name in TENSORS | INTS | BOOLS | FLOATS | {"requested_backend"}:
            normalized[name] = value
    for name in sorted(normalized):
        value = normalized[name]
        if name == "capture":
            continue
        if name in TENSORS:
            if value is None:
                fields.append(f"{name}:None")
            elif getattr(value, "shape", None) is not None:
                item = tensor_metadata(value)
                fields.extend(
                    (
                        f"{name}:{tuple(item['shape'])}",
                        f"{name}_dtype:{item['dtype']}",
                        f"{name}_stride:{tuple(item['stride'])}",
                        f"{name}_offset:{item['offset']}",
                    )
                )
        elif name == "requested_backend":
            if type(value) is str and re.fullmatch(r"[\w.-]+", value):
                fields.append(f"{name}:{value}")
        elif value is None:
            fields.append(f"{name}:None")
        elif name in BOOLS and type(value) is bool:
            fields.append(f"{name}:{int(value)}")
        elif name in INTS and type(value) is int:
            fields.append(f"{name}:{value}")
        elif name in FLOATS and type(value) in (float, int) and math.isfinite(value):
            fields.append(f"{name}:{value}")
    return ",".join(fields)


def parse_label(label):
    if not label.startswith(PREFIX):
        raise ValueError("not a sparse op label")
    raw = {}
    for part in re.split(r",(?=[A-Za-z_][\w.]*:)", label[len(PREFIX) :].strip()):
        key, sep, value = part.partition(":")
        if not sep or key in raw or not re.fullmatch(r"[A-Za-z_][\w.]*", key):
            raise ValueError("duplicate or malformed field")
        raw[key] = value
    if (
        raw.get("v") != "1"
        or raw.get("op") not in OPS
        or raw.get("fw") != "vllm"
        or raw.get("model") != "step5"
    ):
        raise ValueError("unsupported version/op/fixed fields")
    if raw.get("pd_type") not in ("prefill", "decode", "unknown") or raw.get(
        "scope"
    ) not in ("op", "kernel"):
        raise ValueError("unsupported phase or scope")
    for name in ("impl", "backend", "requested_backend"):
        if name == "requested_backend" and name not in raw:
            continue
        pattern = r"(?:[\w.-]|<locals>|<lambda>)+" if name == "impl" else r"[\w.-]+"
        if not re.fullmatch(pattern, raw.get(name, "")):
            raise ValueError(f"invalid {name}")
    case = {key: raw[key] for key in FIXED if key in raw}
    case.update(schema=1, tensors={}, scalars={})
    for key, value in raw.items():
        if key in FIXED:
            continue
        if key in TENSORS:
            related = {key + suffix for suffix in ("_dtype", "_stride", "_offset")}
            if value == "None":
                if related & raw.keys():
                    raise ValueError("absent tensor carries metadata")
                case["tensors"][key] = None
            else:
                if not related <= raw.keys() or raw[key + "_dtype"] not in DTYPES:
                    raise ValueError("missing tensor metadata or unsupported dtype")
                offset = raw[key + "_offset"]
                if not re.fullmatch(r"\d+", offset):
                    raise ValueError("invalid tensor offset")
                item = {
                    "shape": _vector(value),
                    "stride": _vector(raw[key + "_stride"]),
                    "dtype": raw[key + "_dtype"],
                    "offset": int(offset),
                }
                storage_elements(item)
                case["tensors"][key] = item
        elif any(
            key.endswith(suffix) and key[: -len(suffix)] in TENSORS
            for suffix in ("_dtype", "_stride", "_offset")
        ):
            base = key.rsplit("_", 1)[0]
            if base not in raw:
                raise ValueError("orphan tensor metadata")
        elif key in INTS | BOOLS | FLOATS:
            if value == "None" and key != "capture":
                case["scalars"][key] = None
            elif key in FLOATS:
                number = float(value)
                if not math.isfinite(number) or number <= 0:
                    raise ValueError("scale must be finite and positive")
                case["scalars"][key] = number
            else:
                if not re.fullmatch(r"\d+", value) or int(value) > 2147483647:
                    raise ValueError("invalid integer scalar")
                number = int(value)
                if key in BOOLS and number not in (0, 1):
                    raise ValueError("boolean scalar must be 0/1")
                case["scalars"][key] = number
        else:
            raise ValueError(f"unknown field: {key}")
    if "capture" not in case["scalars"]:
        raise ValueError("missing capture flag")
    canonical = json.dumps(case, sort_keys=True, separators=(",", ":"))
    case["id"] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    case["raw_label"] = label
    return case


def extract_cases(paths):
    unique = {}
    for path in paths:
        path = Path(path).resolve()
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        for line_number, line in enumerate(content.decode("utf-8").splitlines(), 1):
            match = re.search(r"\[FW_SPARSE\] op:(score|topk|attention)(?=,|$)", line)
            if not match:
                continue
            try:
                case = parse_label(line[match.start() :].strip())
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            item = unique.setdefault(
                case["id"], {**case, "observations": 0, "sources": []}
            )
            item["observations"] += 1
            item["sources"].append(
                {"path": str(path), "sha256": digest, "line": line_number}
            )
    return {
        "schema": 1,
        "synthetic": True,
        "frequency_note": "Exported ranges are not graph replay invocation counts.",
        "cases": sorted(unique.values(), key=lambda item: item["id"]),
    }


def estimate_memory(case):
    sizes = [
        storage_elements(t) * DTYPES[t["dtype"]]
        for t in case["tensors"].values()
        if t is not None
    ]
    return sum(sizes)


def profiled(op, *, backend="dispatch", scope="op", pd_type="unknown", names=None):
    """Compose instrumentation with a functional patch without copying its marker."""

    def decorate(original):
        signature = None

        @functools.wraps(original, updated=())
        def wrapped(*args, **kwargs):
            from vllm_sail import envs

            if not envs.VLLM_SAIL_NVTX_PROFILE:
                return original(*args, **kwargs)
            nonlocal signature
            from vllm_sail.profiling.step5_sparse import trace_call

            if names is None:
                try:
                    if signature is None:
                        signature = inspect.signature(original)
                    bound = signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    values = bound.arguments
                except (TypeError, ValueError):
                    return original(*args, **kwargs)
            else:
                values = dict(zip(names, args, strict=False))
                values.update(kwargs)
            owner = values.get("self")
            qualname = getattr(original, "__qualname__", type(original).__qualname__)
            if owner is not None and "Step5FlashInferSparse." in qualname:
                from vllm_sail.profiling.step5_sparse import instrument_adapter

                instrument_adapter(owner)
            elif owner is not None and "CompactPagedPrefill.finish" in qualname:
                from vllm_sail.profiling.step5_sparse import _instrument_run

                _instrument_run(owner.wrapper, "flashinfer_paged")
            module = getattr(original, "__module__", type(original).__module__)
            return trace_call(
                op,
                f"{module}.{qualname}",
                backend,
                values,
                lambda: original(*args, **kwargs),
                scope=scope,
                pd_type=pd_type,
            )

        wrapped._sail_sparse_profile = True
        return wrapped

    return decorate
