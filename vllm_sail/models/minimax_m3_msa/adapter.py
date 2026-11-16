# SPDX-License-Identifier: Apache-2.0
"""Lazy access to the external ``msa_for_sail`` JIT kernels.

``msa_for_sail`` compiles its hgcc variants on first use and is not installed
in every PPU environment, so nothing here imports it at module scope. A missing
package disables the SAIL MSA path (the upstream Triton implementation keeps
working) instead of failing plugin import.

The JIT cache (``~/.cache/minfer/msa_for_sail``, or ``MINFER_FMHA_CACHE_DIR``)
is shared by every TP worker. ``msa_for_sail`` skips a target once its ``.so``
exists but does not guard the *writing*: when all workers reach a target at the
same time (first warmup batch of a tensor-parallel run), one worker can load a
half-written ``.so`` and fail with ``Module has no function ...``. Each entry
point is therefore wrapped so that its first call per process *per compile
signature* holds an exclusive ``flock`` on a lock file next to the cache; the
other workers wait and then find the finished artifact. Compilation thus
happens inside ``vllm serve`` startup, on the first real call, instead of
requiring a separate prewarm process.
"""

from __future__ import annotations

import fcntl
import functools
import os
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_LOCK_NAME = ".vllm_sail_msa_compile.lock"
#: Signatures whose first call already went through the compile lock.
_warmed: set[tuple] = set()


@dataclass(frozen=True)
class SailMSAOps:
    fmha_sail_plan: Callable[..., Any]
    fmha_sail: Callable[..., Any]
    sparse_topk_select: Callable[..., Any]


def _cache_base() -> Path:
    """Directory ``msa_for_sail`` builds into (``jit.CACHE_BASE``)."""
    try:
        from msa_for_sail.jit import CACHE_BASE

        return Path(CACHE_BASE)
    except Exception:
        explicit = os.environ.get("MINFER_FMHA_CACHE_DIR")
        if explicit:
            return Path(explicit)
        return Path(os.path.expanduser("~/.cache/minfer/msa_for_sail"))


@contextmanager
def _compile_lock():
    path = _cache_base() / _LOCK_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info(
                "Waiting for another worker to finish an msa_for_sail JIT "
                "build (lock: %s)",
                path,
            )
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _plan_variant(plan: dict[str, Any]) -> tuple:
    """The sub-plan fields ``fmha_sail`` picks its compiled variant from."""
    return (
        plan.get("qo_tile_size"),
        plan.get("kv_tile_size"),
        plan.get("pack_factor"),
        int(plan.get("max_qo_len", 0)) <= 64,
        plan.get("workspace_o") is not None,  # split-KV
    )


def _signature_part(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return str(value.dtype)
    if isinstance(value, bool | str | torch.dtype) or value is None:
        return value
    if isinstance(value, int):
        # Lens, page and head counts: never a variant selector.
        return "int"
    if isinstance(value, dict):
        return _plan_variant(value)
    if isinstance(value, tuple):
        # ``fmha_sail_plan`` result: (mixed, split, batch, decode, prefill).
        return tuple(_plan_variant(v) for v in value if isinstance(v, dict))
    return type(value).__name__


def _compile_signature(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple:
    """Identity of a JIT call: what can select a distinct compiled module.

    ``fmha_sail`` picks its variant from the q dtype, the output flags and the
    plan's tile/pack/split fields; the tiles follow the batch lengths, so a new
    step can need an unbuilt variant at an unchanged dtype. The plan and top-k
    entry points build one module each. Shapes and integers are left out: they
    vary every step without selecting a variant, and each new key costs every
    TP worker a turn on the shared lock.
    """
    parts = [_signature_part(value) for value in args]
    parts.extend((name, _signature_part(kwargs[name])) for name in sorted(kwargs))
    return tuple(parts)


def _locked(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Serialize the first (compiling) call of each signature across workers."""

    @functools.wraps(fn)
    def call(*args: Any, **kwargs: Any) -> Any:
        key = (name,) + _compile_signature(args, kwargs)
        if key in _warmed:
            return fn(*args, **kwargs)
        with _compile_lock():
            result = fn(*args, **kwargs)
        _warmed.add(key)
        return result

    return call


@functools.cache
def sail_msa_ops() -> SailMSAOps | None:
    """The ``msa_for_sail`` entry points, or ``None`` when unusable."""
    try:
        from msa_for_sail import fmha_sail, fmha_sail_plan, sparse_topk_select
    except ImportError:
        logger.warning_once(
            "MiniMax M3 SAIL MSA is available on this PPU but the "
            "`msa_for_sail` package is not importable; sparse attention stays "
            "on the Triton implementation. Install MSA-for-sail "
            "(`pip install -e python/`) to enable it."
        )
        return None
    return SailMSAOps(
        fmha_sail_plan=_locked("plan", fmha_sail_plan),
        fmha_sail=_locked("fmha", fmha_sail),
        sparse_topk_select=_locked("topk", sparse_topk_select),
    )


def sail_msa_available() -> bool:
    return sail_msa_ops() is not None


def require_sail_msa_ops() -> SailMSAOps:
    """The entry points for a backend the gating policy already selected."""
    ops = sail_msa_ops()
    if ops is None:
        raise RuntimeError(
            "MiniMax M3 SAIL MSA was selected but `msa_for_sail` is not importable"
        )
    return ops
