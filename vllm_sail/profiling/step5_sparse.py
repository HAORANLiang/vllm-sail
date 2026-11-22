# SPDX-License-Identifier: Apache-2.0
"""Common, opt-in Step5 sparse dispatch instrumentation."""

from __future__ import annotations

import functools
import importlib
import sys
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar

from vllm.logger import init_logger

from vllm_sail import envs
from vllm_sail.patch.utils import PATCH_MARKER, patch
from vllm_sail.step5_sparse_cases import make_label, profiled

logger = init_logger(__name__)
_PHASE = ContextVar("sail_sparse_phase", default="unknown")
_RANGES = ContextVar("sail_sparse_ranges", default=())
_TOPK_CONTEXT = ContextVar("sail_sparse_topk_launch", default=None)
_REASON = (
    "Opt-in Step5 sparse op metadata ranges without data reads or dispatch changes."
)
_REMOVE = "Upstream Step5 emits versioned score/topk/attention metadata ranges."
_TARGETS = (
    (
        "flashinfer_sparse",
        "Step5FlashInferSparse.decode_logits",
        "score",
        "dispatch",
        "op",
        "decode",
    ),
    (
        "flashinfer_sparse",
        "Step5FlashInferSparse.logits",
        "score",
        "dispatch",
        "op",
        "prefill",
    ),
    (
        "flashinfer_sparse",
        "Step5FlashInferSparse.select_regions",
        "topk",
        "dispatch",
        "op",
        "unknown",
    ),
    (
        "flashinfer_sparse",
        "Step5FlashInferSparse.run_attention",
        "attention",
        "flashinfer_ragged",
        "op",
        "prefill",
    ),
    (
        "ops.sparse_indexer_scores",
        "decode_summary_scores",
        "score",
        "dispatch",
        "op",
        "decode",
    ),
    (
        "ops.sparse_indexer_scores",
        "paged_mqa_logits_sm90",
        "score",
        "upstream_paged",
        "op",
        "unknown",
    ),
    ("ops.sparse_indexer", "select_regions", "topk", "dispatch", "op", "unknown"),
    (
        "ops.sparse_indexer",
        "select_all_regions",
        "topk",
        "select_all",
        "kernel",
        "unknown",
    ),
    (
        "ops.sparse_decode_attention",
        "sparse_decode_attention",
        "attention",
        "upstream_triton",
        "op",
        "decode",
    ),
    (
        "ops.sparse_prefill_pages",
        "CompactPagedPrefill.finish",
        "attention",
        "flashinfer_paged",
        "op",
        "prefill",
    ),
)


@contextmanager
def phase(kind):
    token = _PHASE.set(kind)
    try:
        yield
    finally:
        _PHASE.reset(token)


@contextmanager
def trace_range(op, impl, backend, values, *, scope="op", pd_type="unknown"):
    if not envs.VLLM_SAIL_NVTX_PROFILE:
        yield
        return
    key = (op, scope)
    active = _RANGES.get()
    if key in active:
        yield
        return
    from vllm_sail.profiling import nvtx

    try:
        torch = sys.modules.get("torch")
        cuda_tensor = next(
            (v for v in values.values() if getattr(v, "is_cuda", False)), None
        )
        capture = (
            bool(torch.cuda.is_current_stream_capturing())
            if torch and cuda_tensor is not None
            else False
        )
        label = make_label(
            op,
            impl,
            backend,
            values,
            scope=scope,
            capture=capture,
            pd_type=_PHASE.get() if pd_type == "unknown" else pd_type,
        )
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
        yield
        return
    token = _RANGES.set((*active, key))
    try:
        nvtx.range_push(label)
        try:
            yield
        finally:
            nvtx.range_pop()
    finally:
        _RANGES.reset(token)


def trace_call(op, impl, backend, values, call, *, scope="op", pd_type="unknown"):
    if not envs.VLLM_SAIL_NVTX_PROFILE:
        return call()
    with trace_range(op, impl, backend, values, scope=scope, pd_type=pd_type):
        return call()


def _aliases(module, attribute, original):
    result = [(module, attribute)]
    for name, loaded in list(sys.modules.items()):
        if loaded is None or not name.startswith(
            ("vllm.models.step5.nvidia", "vllm.models.step5.common")
        ):
            continue
        for attr, value in list(vars(loaded).items()):
            if value is original and (name, attr) not in result:
                result.append((name, attr))
    return result


def _install_one(module_name, attribute, replacement):
    patch(
        module_name,
        attribute,
        reason=_REASON,
        affected_versions=">=0.34.0,<0.35.0",
        remove_when=_REMOVE,
    )(replacement)


def install():
    if not envs.VLLM_SAIL_NVTX_PROFILE:
        return False
    installed = False
    for suffix, attr, op, backend, scope, kind in _TARGETS:
        prefix = (
            "vllm.models.step5.common."
            if suffix
            in (
                "ops.sparse_indexer_scores",
                "ops.sparse_indexer",
                "ops.sparse_decode_attention",
            )
            else "vllm.models.step5.nvidia."
        )
        module_name = prefix + suffix
        try:
            module = importlib.import_module(module_name)
            owner, name = module, attr
            if "." in attr:
                cls, name = attr.split(".", 1)
                owner = getattr(module, cls)
            original = getattr(owner, name)
            if getattr(original, "_sail_sparse_profile", False):
                installed = True
                continue
            if getattr(original, PATCH_MARKER, None) is not None:
                continue
            replacement = profiled(op, backend=backend, scope=scope, pd_type=kind)(
                original
            )
            targets = (
                [(module_name, attr)]
                if owner is not module
                else _aliases(module_name, attr, original)
            )
            for target, target_attr in targets:
                _install_one(target, target_attr, replacement)
            installed = True
        except Exception as exc:
            logger.debug("Step5 sparse profiling target unavailable: %s: %s", attr, exc)
    _install_topk_factory()
    _install_adapter_logits()
    _install_launches()
    return installed


def _install_topk_factory():
    name = "vllm.models.step5.common.ops.sparse_indexer"
    try:
        module = importlib.import_module(name)
        original = module.get_indexer_topk
        if getattr(original, "_sail_sparse_profile", False) or getattr(
            original, PATCH_MARKER, None
        ):
            return

        @functools.wraps(original, updated=())
        def factory(backend):
            run = original(backend)
            if not envs.VLLM_SAIL_NVTX_PROFILE or ("topk", "op") not in _RANGES.get():
                return run
            resolver = getattr(run, "resolve_backend", None)
            if resolver is None:
                return profiled(
                    "topk",
                    backend="unresolved",
                    scope="kernel",
                    names=(
                        "scores",
                        "lengths",
                        "next_n",
                        "indices",
                        "topk",
                        "max_seq_len",
                    ),
                )(run)
            if not getattr(resolver, "_sail_sparse_profile", False):

                @functools.wraps(resolver, updated=())
                def resolve(*args, **kwargs):
                    actual = resolver(*args, **kwargs)
                    frame = _TOPK_CONTEXT.get()
                    if frame is not None:
                        stack, values = frame
                        impl = (
                            type(run).__module__
                            + "."
                            + type(run).__qualname__
                            + ".forward"
                        )
                        stack.enter_context(
                            trace_range("topk", impl, actual, values, scope="kernel")
                        )
                    return actual

                resolve._sail_sparse_profile = True
                run.resolve_backend = resolve

            def launch(*args, **kwargs):
                if not envs.VLLM_SAIL_NVTX_PROFILE:
                    return run(*args, **kwargs)
                values = dict(
                    zip(
                        (
                            "scores",
                            "lengths",
                            "next_n",
                            "indices",
                            "topk",
                            "max_seq_len",
                        ),
                        args,
                        strict=False,
                    )
                )
                values.update(kwargs, requested_backend=backend)
                with ExitStack() as stack:
                    token = _TOPK_CONTEXT.set((stack, values))
                    try:
                        return run(*args, **kwargs)
                    finally:
                        _TOPK_CONTEXT.reset(token)

            return launch

        factory._sail_sparse_profile = True
        _install_one(name, "get_indexer_topk", factory)
    except Exception as exc:
        logger.debug("Step5 topk profiling unavailable: %s", exc)


def _install_adapter_logits():
    name = "vllm.models.step5.nvidia.flashinfer_sparse"
    try:
        module = importlib.import_module(name)
        original = module.Step5FlashInferSparse.__init__
        if getattr(original, "_sail_sparse_profile", False) or getattr(
            original, PATCH_MARKER, None
        ):
            return

        @functools.wraps(original, updated=())
        def initialize(self, *args, **kwargs):
            original(self, *args, **kwargs)
            instrument_adapter(self)

        initialize._sail_sparse_profile = True
        _install_one(name, "Step5FlashInferSparse.__init__", initialize)
    except Exception as exc:
        logger.debug("Step5 logits profiling unavailable: %s", exc)


def instrument_adapter(adapter):
    """Instrument only an actual FlashInfer provider, never a dispatch wrapper."""
    if not envs.VLLM_SAIL_NVTX_PROFILE:
        return
    attention = getattr(adapter, "_attention", None)
    if attention is not None:
        _instrument_run(attention, "flashinfer_ragged")
    run = getattr(adapter, "_logits", None)
    if (
        run is not None
        and not getattr(run, "_sail_sparse_profile", False)
        and getattr(run, "__module__", "").startswith("flashinfer")
    ):
        adapter._logits = profiled("score", backend="flashinfer", scope="kernel")(run)


def _instrument_run(wrapper, backend):
    run = getattr(wrapper, "run", None)
    if run is not None and not getattr(run, "_sail_sparse_profile", False):
        wrapper.run = profiled(
            "attention", backend=backend, scope="kernel", pd_type="prefill"
        )(run)


class LaunchObserver:
    """Delegate a Step5 JIT launch without copying or changing its device body."""

    _sail_sparse_profile = True

    def __init__(self, original, op, backend, names, scalars, pd_type="decode"):
        self.original, self.op, self.backend = original, op, backend
        self.names, self.scalars, self.pd_type = names, scalars, pd_type

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __getitem__(self, grid):
        launch = self.original[grid]

        def run(*args, **kwargs):
            if not envs.VLLM_SAIL_NVTX_PROFILE:
                return launch(*args, **kwargs)
            values = dict(zip(self.names, args, strict=False))
            values.update(
                {
                    name: args[index]
                    for name, index in self.scalars.items()
                    if index < len(args)
                }
            )
            values["num_warps"] = kwargs.get("num_warps")
            return trace_call(
                self.op,
                self.original.__name__,
                self.backend,
                values,
                lambda: launch(*args, **kwargs),
                scope="kernel",
                pd_type=self.pd_type,
            )

        return run


def _install_launches():
    target = "vllm.models.step5.common.ops.sparse_decode_attention"
    try:
        module = importlib.import_module(target)
        for name, names, scalars in (
            (
                "_sparse_decode_split",
                (
                    "query",
                    "kv_cache",
                    "indices",
                    "block_table",
                    "positions",
                    "active",
                    "k_scale",
                    "v_scale",
                    "partial",
                    "lse",
                    "output",
                    "gate",
                    "scale",
                ),
                {
                    "heads": 28,
                    "topk": 30,
                    "block_size": 31,
                    "splits": 32,
                    "clear_inactive": 15,
                },
            ),
            (
                "_sparse_decode_combine",
                ("partial", "lse", "active", "output", "gate"),
                {"heads": 7, "splits": 8, "clear_inactive": 10},
            ),
        ):
            original = getattr(module, name)
            if getattr(original, "_sail_sparse_profile", False) or getattr(
                original, PATCH_MARKER, None
            ):
                continue
            _install_one(
                target,
                name,
                LaunchObserver(
                    original, "attention", "upstream_triton", names, scalars
                ),
            )
    except Exception as exc:
        logger.debug("Step5 attention launches unavailable: %s", exc)
    target = "vllm.models.step5.common.ops.sparse_indexer_scores"
    try:
        module = importlib.import_module(target)
        original = module._paged_scores_sm90
        if not getattr(original, "_sail_sparse_profile", False) and not getattr(
            original, PATCH_MARKER, None
        ):
            observer = LaunchObserver(
                original,
                "score",
                "upstream_paged",
                ("query", "packed", "weights", "lengths", "block_table", "output"),
                {
                    "max_seq_len": 15,
                    "next_n": 16,
                    "heads": 17,
                    "page_size": 18,
                    "tile": 19,
                },
                "unknown",
            )
            for name, attr in _aliases(target, "_paged_scores_sm90", original):
                _install_one(name, attr, observer)
    except Exception as exc:
        logger.debug("Step5 score launch unavailable: %s", exc)
    target = "vllm.models.step5.nvidia.ops.sparse_prefill_pages"
    try:
        module = importlib.import_module(target)
        original = module.CompactPagedPrefill.__init__
        if getattr(original, "_sail_sparse_profile", False) or getattr(
            original, PATCH_MARKER, None
        ):
            return

        @functools.wraps(original, updated=())
        def initialize(self, *args, **kwargs):
            original(self, *args, **kwargs)
            if envs.VLLM_SAIL_NVTX_PROFILE:
                _instrument_run(self.wrapper, "flashinfer_paged")

        initialize._sail_sparse_profile = True
        _install_one(target, "CompactPagedPrefill.__init__", initialize)
    except Exception as exc:
        logger.debug("Step5 paged attention profiling unavailable: %s", exc)
