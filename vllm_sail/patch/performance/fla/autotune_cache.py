# SPDX-License-Identifier: Apache-2.0
"""PPU patch: enable ``cache_results=True`` on FLA's autotuned Triton kernels.

The fork adds ``cache_results=True`` to the ``@triton.autotune`` decorator of
FLA kernels across eight ``vllm.third_party.flash_linear_attention.ops``
modules. This enables Triton's disk cache of tuning results, in addition to
its normal process-local cache.

The plugin rebuilds each Autotuner through the live ``triton.autotune`` API
with the same configuration plus ``cache_results=True`` and retains any
surrounding Heuristics wrappers. Same-module launcher wrappers resolve the
kernel name through module globals at call time, so the
rebind takes effect everywhere.

NOTE: this rebuild preserves the autotuner attributes it can read off the
object (``key``, ``prune_configs_by``, ``reset_to_zero``, ``restore_value``,
``pre_hook``, ``post_hook``, ``use_cuda_graph``). Triton's Autotuner internals
are version-dependent and cannot be exercised in a triton-less test
environment, so on a real PPU host the first launch of each kernel is the
effective check. None of the FLA autotune sites use ``warmup``/``rep``.

With vLLM's triton placeholder (no triton installed) the kernels are plain
functions and there is nothing to rebuild; the module then installs no patches
and says so at debug level.

This is an UNCONDITIONAL upstream behaviour change (the fork applies it for all
platforms, not just PPU); it is reproduced as-in-fork and flagged in the
Phase-4 report.
"""

from __future__ import annotations

from copy import copy

from vllm.logger import init_logger
from vllm.triton_utils.importing import HAS_TRITON

from vllm_sail.patch.utils import patch

logger = init_logger(__name__)

_AFFECTED = ">=0.33.0,<0.34.0"

#: Upstream FLA ops modules and the autotuned kernels the fork decorates with
#: cache_results=True. Kept explicit so an upstream rename fails loudly.
_TARGETS: dict[str, tuple[str, ...]] = {
    "vllm.third_party.flash_linear_attention.ops.chunk_delta_h": (
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    ),
    "vllm.third_party.flash_linear_attention.ops.chunk_o": ("chunk_fwd_kernel_o",),
    "vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt": (
        "chunk_scaled_dot_kkt_fwd_kernel",
    ),
    "vllm.third_party.flash_linear_attention.ops.cumsum": (
        "chunk_local_cumsum_scalar_kernel",
        "chunk_local_cumsum_vector_kernel",
    ),
    "vllm.third_party.flash_linear_attention.ops.kda": (
        "chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter",
        "chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra",
        "recompute_w_u_fwd_kernel",
        "chunk_gla_fwd_kernel_o",
        "kda_gate_fwd_kernel",
    ),
    "vllm.third_party.flash_linear_attention.ops.l2norm": (
        "l2norm_fwd_kernel1",
        "l2norm_fwd_kernel",
    ),
    "vllm.third_party.flash_linear_attention.ops.solve_tril": (
        "solve_tril_16x16_kernel",
        "merge_16x16_to_32x32_inverse_kernel",
        "merge_16x16_to_64x64_inverse_kernel",
    ),
    "vllm.third_party.flash_linear_attention.ops.wy_fast": (
        "recompute_w_u_fwd_kernel",
    ),
}

#: Autotuner attributes forwarded to the rebuilt autotuner. The FLA sites only
#: ever set `key` (plus `use_cuda_graph` on one), but forwarding the rest
#: keeps this correct if upstream starts using them. Triton's Autotuner
#: stores the key list under ``keys`` (plural), hence the alias.
_PRESERVED_ATTRS = (
    "configs",
    "key",
    "prune_configs_by",
    "reset_to_zero",
    "restore_value",
    "pre_hook",
    "post_hook",
    "use_cuda_graph",
    "do_bench",
)
_ATTR_ALIASES = {"key": "keys", "do_bench": "_do_bench"}


def _is_autotuner(obj) -> bool:
    # PPU triton builds rename the Autotuner class; the attribute pair is
    # the stable duck-type across triton variants.
    return hasattr(obj, "fn") and hasattr(obj, "configs")


def _autotuner_chain(kernel):
    """Find the tuner inside Triton's outer Heuristics decorators."""
    wrappers = []
    seen = set()
    while not _is_autotuner(kernel):
        if id(kernel) in seen or not (
            hasattr(kernel, "fn") and hasattr(kernel, "values")
        ):
            return None
        seen.add(id(kernel))
        wrappers.append(kernel)
        kernel = kernel.fn
    return wrappers, kernel


def _with_cache_results(tuner):
    """Rebuild ``tuner`` with identical settings plus cache_results=True.

    PPU triton builds expose a different ``autotune`` signature than
    upstream Triton (required positionals, no ``cache_results``), so the
    call is adapted from the live signature instead of assuming one.
    """
    import inspect

    from vllm.triton_utils import triton

    preserved = {}
    for attr in _PRESERVED_ATTRS:
        # Reading do_bench can initialize Triton's driver. Its stored callback
        # preserves a custom benchmark without invoking that cached property.
        value = getattr(tuner, _ATTR_ALIASES.get(attr, attr), None)
        if value is None and attr in _ATTR_ALIASES:
            if attr != "do_bench":
                value = getattr(tuner, attr, None)
        # Generated reset/restore hooks close over the old tuner. Let the new
        # constructor recreate those; only explicit user hooks are portable.
        if attr in ("pre_hook", "post_hook") and not getattr(
            tuner, f"user_defined_{attr}", True
        ):
            continue
        if value is not None:
            preserved[attr] = value
    if "prune_configs_by" not in preserved:
        pruning = {
            key: getattr(tuner, attr, None)
            for key, attr in (
                ("perf_model", "perf_model"),
                ("top_k", "configs_top_k"),
                ("early_config_prune", "early_config_prune"),
            )
        }
        preserved["prune_configs_by"] = {
            key: value for key, value in pruning.items() if value is not None
        }
    params = inspect.signature(triton.autotune).parameters
    has_varkw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    def accepted(name: str) -> bool:
        return has_varkw or name in params

    kwargs = {name: value for name, value in preserved.items() if accepted(name)}
    if accepted("cache_results"):
        kwargs["cache_results"] = True
    else:
        logger.warning(
            "vllm-sail: this triton autotune() has no cache_results parameter; "
            "skipping the FLA autotune cache patch."
        )
        return None

    args = []
    for name, param in params.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if name not in kwargs and param.default is param.empty:
            if hasattr(tuner, name):
                kwargs[name] = getattr(tuner, name)
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            if name in kwargs:
                args.append(kwargs.pop(name))
            elif param.default is not param.empty:
                args.append(param.default)
            else:
                raise TypeError(f"cannot recover required autotune argument {name!r}")
    return triton.autotune(*args, **kwargs)(tuner.fn)


def _install() -> None:
    if not HAS_TRITON:
        logger.debug(
            "vllm-sail: triton is unavailable; the FLA kernels are plain "
            "functions and there is no autotuner to add cache_results to."
        )
        return

    import importlib

    for module_path, kernel_names in _TARGETS.items():
        module = importlib.import_module(module_path)
        for kernel_name in kernel_names:
            kernel = getattr(module, kernel_name)
            chain = _autotuner_chain(kernel)
            if chain is None:
                logger.warning(
                    "vllm-sail: %s.%s has no triton Autotuner in its "
                    "decorator chain (%r); "
                    "skipping the cache_results patch for it.",
                    module_path,
                    kernel_name,
                    type(kernel),
                )
                continue
            wrappers, tuner = chain
            try:
                rebuilt = _with_cache_results(tuner)
            except TypeError as exc:
                logger.warning(
                    "vllm-sail: cannot rebuild %s.%s with this triton's "
                    "autotune signature (%s); skipping the cache_results "
                    "patch for it.",
                    module_path,
                    kernel_name,
                    exc,
                )
                continue
            if rebuilt is None:
                continue
            # Copy wrappers instead of mutating the original: patch metadata
            # must retain the complete upstream object, including its tuner.
            for wrapper in reversed(wrappers):
                replacement = copy(wrapper)
                replacement.fn = rebuilt
                rebuilt = replacement
            patch(
                module_path,
                kernel_name,
                reason=(
                    "Fork adds cache_results=True to this kernel's "
                    "@triton.autotune so benchmark results are cached across "
                    "calls. The decorator cannot be mutated after the fact, so "
                    "the autotuner is rebuilt with the same configuration plus "
                    "cache_results=True. UNCONDITIONAL in the fork (all "
                    "platforms), reproduced as-in-fork."
                ),
                affected_versions=_AFFECTED,
                remove_when=(
                    "upstream enables cache_results on the FLA autotune sites, "
                    "or the fork drops the cache_results additions."
                ),
            )(rebuilt)


_install()
