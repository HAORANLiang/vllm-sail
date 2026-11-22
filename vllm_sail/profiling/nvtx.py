# SPDX-License-Identifier: Apache-2.0
"""Shared NVTX helpers for the opt-in profiling patches.

The in-tree PPU fork duplicated an identical import/no-op block in
``v1/worker/gpu_model_runner.py``, ``v1/engine/core.py`` and
``v1/core/sched/scheduler.py``. This module implements those helpers once,
with graceful degradation per optional dependency:

* ``nvtx`` (PyPI, ``vllm-sail[profiling]``) -- ``annotate`` and ``mark``
* ``model_prof`` (PPU SDK)                 -- the ``prof_iter`` hook
* ``torch.cuda.nvtx``                      -- ``range_push`` / ``range_pop``

Every helper is always importable and always callable; whatever is missing
becomes a no-op, so profiling can never crash the engine. The backends are
resolved once at import time and held in module-level slots, which tests can
swap to observe calls.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar

from vllm.logger import init_logger

logger = init_logger(__name__)

_F = TypeVar("_F", bound=Callable[..., Any])

try:
    from nvtx import annotate as _nvtx_annotate
    from nvtx import mark as _nvtx_mark
except ImportError:
    _nvtx_annotate = None
    _nvtx_mark = None

try:
    from model_prof import prof_iter as _model_prof_iter
except ImportError:
    _model_prof_iter = None

try:
    from torch.cuda.nvtx import range_pop as _torch_range_pop
    from torch.cuda.nvtx import range_push as _torch_range_push
except ImportError:
    _torch_range_pop = None
    _torch_range_push = None

#: True when the ``nvtx`` PyPI package (annotate/mark) is usable.
has_nvtx = _nvtx_mark is not None
#: True when ``model_prof`` from the PPU SDK provides ``prof_iter``.
has_model_prof = _model_prof_iter is not None
#: True when ``torch.cuda.nvtx`` imported; calls may still fail on a
#: CPU-only torch build, which :func:`range_push` swallows.
has_torch_nvtx = _torch_range_push is not None

_torch_nvtx_warned = False


def mark(message: str) -> None:
    """Emit an instantaneous NVTX mark, or no-op without the ``nvtx`` package."""
    if _nvtx_mark is not None:
        _nvtx_mark(message)


def annotate(name: str) -> Callable[[_F], _F]:
    """Decorator/context manager naming an NVTX range.

    Without the ``nvtx`` package this degrades to an identity decorator that
    preserves the wrapped callable's metadata, so ``@annotate`` sites stay
    unconditional.
    """
    if _nvtx_annotate is not None:
        return _nvtx_annotate(name)

    def decorator(func: _F) -> _F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def range_push(label: str) -> None:
    """Open a ``torch.cuda.nvtx`` range, or no-op when unavailable.

    Call-time errors (e.g. CPU-only torch raising "NVTX functions not
    installed") are swallowed with a one-time warning: an observability
    helper must never take down inference.
    """
    global _torch_nvtx_warned
    if _torch_range_push is None:
        return
    try:
        _torch_range_push(label)
    except Exception as exc:
        if not _torch_nvtx_warned:
            _torch_nvtx_warned = True
            logger.warning(
                "torch.cuda.nvtx is unusable in this build (%s); NVTX ranges "
                "are disabled for this process.",
                exc,
            )


def range_pop() -> None:
    """Close the range opened by the matching :func:`range_push`."""
    if _torch_range_pop is None:
        return
    try:
        _torch_range_pop()
    except Exception:
        # Already reported by range_push; a pop without a working push must
        # stay silent to keep the pair balanced.
        pass


def step5_sparse_label(
    pd_type: str, args: tuple, kwargs: dict, *, dump_dict: bool = False
) -> str:
    """Record shape/dtype only; dicts are opt-in; tensor/scalar values are not read."""
    proxy_prefix = "dsa_proxy" if pd_type == "prefill" else "proxy"
    names = (
        "self",
        "query",
        "kv_cache",
        "attn_metadata" if pd_type == "prefill" else "metadata",
        "output",
        f"{proxy_prefix}_query",
        f"{proxy_prefix}_key",
        f"{proxy_prefix}_weights",
        f"{proxy_prefix}_z",
    )
    values = dict(zip(names, args, strict=False))
    values.update(kwargs)
    fields = [
        f"[FW_SPARSE] op:_forward_{pd_type}",
        "fw:vllm",
        "model:step5",
        "backend:default",
        f"pd_type:{pd_type}",
    ]

    active_dicts = set()

    def tensor_field(name, value, depth=0):
        if isinstance(value, dict):
            # Dicts are not walked when off; when on, the raw dict, tensor values and
            # arbitrary object keys are still never formatted.
            if not dump_dict or depth >= 4 or id(value) in active_dicts:
                return
            active_dicts.add(id(value))
            try:
                for key, item in value.items():
                    if type(key) not in (str, int):
                        continue
                    if (
                        isinstance(item, dict)
                        or getattr(item, "shape", None) is not None
                    ):
                        tensor_field(f"{name}.{key}", item, depth + 1)
            finally:
                active_dicts.remove(id(value))
            return
        shape = getattr(value, "shape", None)
        fields.append(f"{name}:{tuple(shape) if shape is not None else None}")
        if shape is not None and hasattr(value, "dtype"):
            fields.append(f"{name}_dtype:{value.dtype}")

    for name in ("query", "kv_cache", "output", *names[5:]):
        tensor_field(name, values.get(name))
    metadata = values.get(names[3])
    if isinstance(metadata, dict):
        tensor_field(names[3], metadata)
    else:
        for name in ("block_table", "slot_mapping", "seq_lens", "query_start_loc"):
            tensor_field(name, getattr(metadata, name, None))
    scales = values.get("kv_scales", (None, None))
    if isinstance(scales, dict):
        tensor_field("kv_scales", scales)
    else:
        tensor_field("k_scale", scales[0])
        tensor_field("v_scale", scales[1])
    if pd_type == "decode":
        tensor_field("gate", values.get("gate"))
    return ",".join(fields)


@contextmanager
def step5_op_phase(pd_type):
    from vllm_sail.profiling.step5_sparse import phase

    with phase(pd_type):
        yield


def prof_iter(iteration: int) -> None:
    """Per-iteration hook for the PPU SDK profiler; no-op without model_prof."""
    if _model_prof_iter is not None:
        _model_prof_iter(iteration)


def sche_mark(sche_output: Any) -> None:
    """Emit NVTX marks describing one scheduler output.

    Mirrors the fork's ``sche_mark`` verbatim (labels included) so existing
    PPU profiling workflows read the same events.
    """
    if _nvtx_mark is None:
        return
    if len(sche_output.scheduled_new_reqs) > 0:
        reqs = [req.req_id for req in sche_output.scheduled_new_reqs]
        mark(f"new_reqs: {reqs}")
    mark(
        f"total_tokens={sche_output.total_num_scheduled_tokens},"
        f"req_id:num_tokens={sche_output.num_scheduled_tokens}"
    )
    if len(sche_output.finished_req_ids) > 0:
        mark(f"finish_req: {sche_output.finished_req_ids}")


def batch_label(sche_output: Any) -> str:
    """Label of the fork's ``total bs=..., P bs=...`` batch range.

    The prefill/decode split is derived from the scheduler output alone, so
    both model runners label their forward range identically even though they
    hand it different arguments.
    """
    spec_tokens = sche_output.scheduled_spec_decode_tokens
    total_bs = len(sche_output.num_scheduled_tokens)
    p_bs = sum(
        1
        for req_id, num_tokens in sche_output.num_scheduled_tokens.items()
        if num_tokens - len(spec_tokens.get(req_id, ())) > 1
    )
    return f"total bs={total_bs}, P bs={p_bs}"


def instrument_compute_logits(model: Any) -> None:
    """Add the fork's shape-aware ``compute_logits`` range to ``model``.

    Called before every forward and idempotent per model, because the model is
    only available after the weights are loaded. ``model`` may be ``None``
    (early calls), which is a no-op.
    """
    compute_logits = getattr(model, "compute_logits", None)
    if compute_logits is None or getattr(compute_logits, "_ppu_nvtx_logits", False):
        return

    @functools.wraps(compute_logits)
    def traced(hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
        range_push(f"[FW_GEMM] op:compute_logits,hidden_states:{hidden_states.shape}")
        try:
            return compute_logits(hidden_states, *args, **kwargs)
        finally:
            range_pop()

    traced._ppu_nvtx_logits = True
    model.compute_logits = traced
