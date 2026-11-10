# SPDX-License-Identifier: Apache-2.0
"""Disable the MiniMax fused allreduce+RMSNorm QK kernel on PPU.

The fork unconditionally sets ``_MINIMAX_FUSED_AR_RMS_QK = None`` in
``vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py`` (was
``getattr(torch.ops._C, "minimax_allreduce_rms_qk", None)``): with cuda.bindings
13.x, ``cudaMalloc`` returns ``cudaErrorInvalidValue`` even with a valid CUDA
context, which makes the Lamport workspace initialization fail on PPU hosts.
Forcing ``None`` routes both usage sites — the ``minimax_qk_norm_fusion`` custom
op and ``MiniMaxText01RMSNormTP.__init__``'s workspace setup — onto the
functionally equivalent pure-PyTorch allreduce + RMSNorm fallback. Both sites
re-read the module global at call/construction time, so a single attribute
override suffices; nothing outside this module imports the name.

``@patch`` cannot install ``None`` (the replacement must carry the patch marker)
and ``patch_value`` rejects an existing differing value, so the override is done
imperatively with explicit ``PATCH_REGISTRY`` bookkeeping matching ``patch_value``
records. The plugin only loads on PPU hosts, so the override is unconditional at
install time, as-in-fork.
"""

from __future__ import annotations

from vllm_sail.patch.utils import PATCH_REGISTRY, PatchRecord

_MODULE = "vllm.model_executor.layers.minimax_rms_norm.rms_norm_tp"
_TARGET = f"{_MODULE}._MINIMAX_FUSED_AR_RMS_QK"
REASON = (
    "On PPU hosts with cuda.bindings 13.x, cudaMalloc returns "
    "cudaErrorInvalidValue even with a valid CUDA context, so the Lamport "
    "workspace used by the fused allreduce+RMSNorm QK kernel fails to "
    "initialize. The fork disables the kernel module-wide; the pure-PyTorch "
    "allreduce + RMSNorm fallback is functionally equivalent."
)
AFFECTED_VERSIONS = ">=0.33.0,<0.34.0"
REMOVE_WHEN = (
    "PPU's cuda.bindings/cudaMalloc stack accepts the Lamport workspace "
    "allocation (the upstream try/except in MiniMaxText01RMSNormTP.__init__ "
    "then suffices), or upstream gates the kernel on a platform capability "
    "PPU answers correctly."
)

#: (target, reason, affected versions, remove_when) for every patch installed
#: by this module. Read by the metadata-shape unit tests.
METADATA = ((_TARGET, REASON, AFFECTED_VERSIONS, REMOVE_WHEN),)

_installed = False


def install() -> None:
    """Force the fused kernel handle to None. Requires vLLM; idempotent."""
    global _installed
    if _installed:
        return

    import importlib

    from vllm.logger import init_logger

    logger = init_logger(__name__)
    module = importlib.import_module(_MODULE)
    if getattr(module, "_MINIMAX_FUSED_AR_RMS_QK", None) is None:
        _installed = True
        return

    module._MINIMAX_FUSED_AR_RMS_QK = None
    PATCH_REGISTRY.append(
        PatchRecord(
            target=_TARGET,
            reason=REASON,
            affected_versions=AFFECTED_VERSIONS,
            remove_when=REMOVE_WHEN,
            kind="value",
            was_missing=False,
            original_source=None,
        )
    )
    _installed = True
    logger.debug("Disabled MiniMax fused allreduce+RMSNorm QK kernel for PPU")
