# SPDX-License-Identifier: Apache-2.0
"""SAIL MSA backend for MiniMax M3 sparse attention on PPU.

Attributes are resolved lazily (PEP 562) because importing the backend modules
requires both torch and the upstream ``vllm.models.minimax_m3`` package; the
CPU-only unit-test tier imports :mod:`vllm_sail.models.minimax_m3_msa.gating`
without either installed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_sail.models.minimax_m3_msa.attention import (
        MiniMaxM3SparseSailMSABackend,
        MiniMaxM3SparseSailMSAImpl,
    )
    from vllm_sail.models.minimax_m3_msa.gating import (
        SailMSASelection,
        decide_from_config,
        decide_from_current_config,
        default_indexer_kv_dtype,
    )
    from vllm_sail.models.minimax_m3_msa.indexer import (
        MiniMaxM3IndexerSailMSABackend,
        MiniMaxM3IndexerSailMSAImpl,
    )

__all__ = [
    "MiniMaxM3IndexerSailMSABackend",
    "MiniMaxM3IndexerSailMSAImpl",
    "MiniMaxM3SparseSailMSABackend",
    "MiniMaxM3SparseSailMSAImpl",
    "SailMSASelection",
    "decide_from_config",
    "decide_from_current_config",
    "default_indexer_kv_dtype",
]

_EXPORTS = {
    "MiniMaxM3IndexerSailMSABackend": (".indexer", "MiniMaxM3IndexerSailMSABackend"),
    "MiniMaxM3IndexerSailMSAImpl": (".indexer", "MiniMaxM3IndexerSailMSAImpl"),
    "MiniMaxM3SparseSailMSABackend": (".attention", "MiniMaxM3SparseSailMSABackend"),
    "MiniMaxM3SparseSailMSAImpl": (".attention", "MiniMaxM3SparseSailMSAImpl"),
    "SailMSASelection": (".gating", "SailMSASelection"),
    "decide_from_config": (".gating", "decide_from_config"),
    "decide_from_current_config": (".gating", "decide_from_current_config"),
    "default_indexer_kv_dtype": (".gating", "default_indexer_kv_dtype"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
