# SPDX-License-Identifier: Apache-2.0
"""Step5 Norm metadata protocol; stdlib only, never recovers real activations."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

DTYPES = {"torch.bfloat16": "bf16", "torch.float16": "fp16", "torch.float32": "fp32"}
_LABEL_PREFIX = "[FW_NATIVE] op:_Step5Norm"
_FIELDS = (
    "v",
    "op",
    "M",
    "H",
    "shape",
    "stride",
    "X",
    "W",
    "wshape",
    "wstride",
    "R",
    "rshape",
    "rstride",
    "Y",
    "eps",
    "Z",
    "out",
    "oshape",
    "ostride",
    "capture",
)


def _vector(values) -> str:
    return "x".join(str(int(v)) for v in values)


def _tensor(tensor) -> tuple[str, str, str]:
    if tensor is None:
        return "-", "-", "-"
    return (
        DTYPES.get(str(tensor.dtype), str(tensor.dtype)),
        _vector(tensor.shape),
        _vector(tensor.stride()),
    )


def make_label(
    x,
    weight,
    eps,
    zero_centered=False,
    residual=None,
    out=None,
    out_dtype=None,
    capture=False,
) -> str:
    """Query shape/stride/dtype only; no data reads and no device sync.

    Y is the requested output dtype.
    """
    xd, shape, stride = _tensor(x)
    wd, ws, wt = _tensor(weight)
    rd, rs, rt = _tensor(residual)
    _, os, ot = _tensor(out)
    yd = DTYPES.get(str(out_dtype or x.dtype), str(out_dtype or x.dtype))
    values = (
        "1",
        "add_rms" if residual is not None else "rms",
        str(math.prod(x.shape[:-1])),
        str(x.shape[-1]),
        shape,
        stride,
        xd,
        wd,
        ws,
        wt,
        rd,
        rs,
        rt,
        yd,
        repr(float(eps)),
        str(int(zero_centered)),
        str(int(out is not None)),
        os,
        ot,
        str(int(capture)),
    )
    kind = "D" if capture else "P"
    return f"{_LABEL_PREFIX},type:{kind}," + ",".join(
        f"{key}={value}" for key, value in zip(_FIELDS, values, strict=False)
    )


def _parse_vector(raw: str) -> list[int]:
    values = [int(v) for v in raw.split("x")]
    if not values or any(v < 0 for v in values):
        raise ValueError("shape/stride must be vectors of non-negative integers")
    return values


def is_norm_label(label) -> bool:
    """Match the Norm candidate prefix only; parse_label still checks the fields."""
    return isinstance(label, str) and label.startswith(
        (_LABEL_PREFIX + ",", "D_Step5Norm,", "P_Step5Norm,")
    )


def parse_label(label: str) -> dict[str, Any]:
    """Strict versioned parsing; corrupt or unknown versions never become benchmark
    input silently.
    """
    prefix, *parts = label.split(",")
    if prefix == _LABEL_PREFIX:
        if not parts or parts[0] not in ("type:D", "type:P"):
            raise ValueError(
                "Norm type must be D/P and separated from the metadata by a comma"
            )
        expected_capture = int(parts.pop(0) == "type:D")
    elif prefix in ("D_Step5Norm", "P_Step5Norm"):
        expected_capture = int(prefix.startswith("D"))
    else:
        raise ValueError("not a Step5 Norm label")
    pairs = [part.split("=", 1) for part in parts]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("malformed label field")
    raw = dict(pairs)
    if len(raw) != len(pairs) or set(raw) != set(_FIELDS) or raw["v"] != "1":
        raise ValueError("duplicate/missing fields or unsupported schema version")
    if raw["op"] not in ("rms", "add_rms"):
        raise ValueError("unsupported Norm op")
    result: dict[str, Any] = dict(raw)
    for key in ("v", "M", "H", "Z", "out", "capture"):
        result[key] = int(raw[key])
    if any(result[key] not in (0, 1) for key in ("Z", "out", "capture")):
        raise ValueError("boolean fields must be 0/1")
    if result["capture"] != expected_capture:
        raise ValueError("capture does not match the label prefix")
    result["eps"] = float(raw["eps"])
    if not math.isfinite(result["eps"]) or result["eps"] <= 0:
        raise ValueError("epsilon must be finite and positive")
    for key in (
        "shape",
        "stride",
        "wshape",
        "wstride",
        "rshape",
        "rstride",
        "oshape",
        "ostride",
    ):
        result[key] = None if raw[key] == "-" else _parse_vector(raw[key])
    for dtype, shape, stride, required in (
        ("X", "shape", "stride", True),
        ("W", "wshape", "wstride", True),
        ("R", "rshape", "rstride", result["op"] == "add_rms"),
        (None, "oshape", "ostride", bool(result["out"])),
    ):
        if required:
            if (
                result[shape] is None
                or result[stride] is None
                or len(result[shape]) != len(result[stride])
            ):
                raise ValueError("tensor shape/stride mismatch")
            if dtype and raw[dtype] not in DTYPES.values():
                raise ValueError("unsupported dtype")
        elif (
            result[shape] is not None
            or result[stride] is not None
            or (dtype and raw[dtype] != "-")
        ):
            raise ValueError("absent tensor carries metadata")
    if raw["Y"] not in DTYPES.values():
        raise ValueError("unsupported output dtype")
    if (
        result["H"] <= 0
        or result["H"] != result["shape"][-1]
        or result["M"] != math.prod(result["shape"][:-1])
    ):
        raise ValueError("M/H does not match the full shape")
    if result["wshape"] != [result["H"]]:
        raise ValueError("weight must be a hidden-size vector")
    for key in ("rshape", "oshape"):
        if result[key] is not None and result[key] != result["shape"]:
            raise ValueError("residual/out shape must equal input")
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["id"] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return result


def labels_from_file(path: str | Path):
    """Accept torch/NVTX-exported JSON or one label per line, optionally gzipped."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        text = stream.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        yield from (
            line.strip() for line in text.splitlines() if is_norm_label(line.strip())
        )
        return
    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    if not isinstance(events, list):
        raise ValueError("JSON must contain a traceEvents list")
    for event in events:
        if not isinstance(event, dict):
            continue
        # Prefer name within one event so a label in both name and args counts once.
        args = event.get("args", {})
        candidates = [event.get("name", "")]
        if isinstance(args, dict):
            candidates += [args.get(key, "") for key in ("Text", "text", "message")]
        for name in candidates:
            if is_norm_label(name):
                yield name
                break


def extract_cases(paths) -> dict[str, Any]:
    unique: dict[str, dict[str, Any]] = {}
    for path in paths:
        source = str(Path(path).resolve())
        for label in labels_from_file(path):
            case = parse_label(label)
            item = unique.setdefault(
                case["id"], {**case, "sources": [], "observations": 0}
            )
            if source not in item["sources"]:
                item["sources"].append(source)
            item["observations"] += 1
    return {
        "schema": 1,
        "input_values": "shape-matched synthetic (not real activations)",
        "frequency_note": "observations counts exported ranges; capture/replay cannot infer the real call frequency",
        "cases": sorted(unique.values(), key=lambda case: case["id"]),
    }


def load_cases(path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != 1 or not isinstance(data.get("cases"), list):
        raise ValueError("unsupported cases.json schema")
    result = []
    for case in data["cases"]:
        values = []
        for key in _FIELDS:
            value = case[key]
            if key.endswith("shape") or key.endswith("stride"):
                value = "-" if value is None else _vector(value)
            values.append(f"{key}={value}")
        kind = "D" if case["capture"] else "P"
        checked = parse_label(f"{_LABEL_PREFIX},type:{kind}," + ",".join(values))
        if checked["id"] != case["id"]:
            raise ValueError("case id does not match the metadata")
        result.append({**case, **checked})
    return result
