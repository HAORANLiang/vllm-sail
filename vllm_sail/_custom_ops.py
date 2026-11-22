# SPDX-License-Identifier: Apache-2.0
"""Python wrappers for the plugin's compiled extensions.

Phase-1 decision D2 (spec 1.8): PPU ops live under the plugin-private
namespaces ``torch.ops._ppu_C`` / ``torch.ops._ppu_moe_C`` — never under
upstream's ``_C`` / ``_moe_C`` — so the plugin installs cleanly beside a
stock vLLM wheel and can never shadow an upstream op. PPU code calls these
wrappers directly; there is deliberately **no** patch of upstream's
``vllm._custom_ops``.

The fork's only ``vllm/_custom_ops.py`` addition, ``ep_scatter_2_cuda``,
calls ``torch.ops._moe_C``; here it calls ``torch.ops._ppu_moe_C``.

The extensions only exist when the package was built against a PPU SDK
(``PPU_SDK`` set; see ``setup.py``). Importing this module always works —
on a CPU-only install the wrappers raise an actionable error on first call.
"""

from __future__ import annotations

import importlib
import logging

import torch

logger = logging.getLogger(__name__)

_ext_state: dict[str, bool | None] = {"_C": None, "_moe_C": None}


def _ext_loaded(name: str) -> bool:
    """Whether the named extension imported, cached per process.

    Importing registers the ops into their ``torch.ops._ppu_*`` namespaces
    as a static-initializer side effect of the shared library.
    """
    loaded = _ext_state[name]
    if loaded is None:
        try:
            importlib.import_module(f"vllm_sail.{name}")
            loaded = True
        except ImportError as exc:
            logger.info(
                "vllm_sail.%s unavailable (%s); built without PPU_SDK?", name, exc
            )
            loaded = False
        _ext_state[name] = loaded
    return bool(loaded)


def is_available() -> bool:
    """Whether any compiled PPU extension is loaded or loadable."""
    return _ext_loaded("_C") or _ext_loaded("_moe_C")


def _require(name: str, op: str) -> None:
    if not _ext_loaded(name):
        raise RuntimeError(
            f"torch.ops._ppu{name}.{op} is unavailable: vllm_sail.{name} was "
            "not built. Rebuild vllm-sail with PPU_SDK set (see setup.py); "
            "CPU-only installs cannot run PPU kernels."
        )


def ep_scatter_2_cuda(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor | None,
    recv_topk_ids: torch.Tensor,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_index: torch.Tensor,
    output_tensor_scale: torch.Tensor | None,
    with_scale: bool,
) -> None:
    """Scatter EP-received tokens into expert-contiguous layout (fork's
    ``vllm._custom_ops.ep_scatter_2_cuda``, on the plugin namespace)."""
    _require("_moe_C", "ep_scatter_2_cuda")
    torch.ops._ppu_moe_C.ep_scatter_2_cuda(
        recv_x,
        recv_x_scale,
        recv_topk_ids,
        expert_start_loc,
        output_tensor,
        output_index,
        output_tensor_scale,
        with_scale,
    )


def step5_dsa_native_prefill(
    query,
    cache,
    regions,
    table,
    owners,
    positions,
    tiles,
    output,
    logical_regions,
    scale,
    query_tile=0,
) -> None:
    """Explicit API: bad metadata or a missing op raises; no implicit JIT/fallback."""
    from vllm_sail.models.step5.ops import dsa_native

    reason = dsa_native.prefill_reason(
        query,
        cache,
        regions,
        table,
        owners,
        positions,
        tiles,
        output,
        logical_regions,
        scale,
        query_tile,
    )
    if reason is not None:
        raise ValueError("native prefill unsupported: " + reason)
    op = dsa_native.compiled_op("prefill")
    if op is None:
        raise RuntimeError("native prefill precompiled op is unavailable")
    op(
        query,
        cache,
        regions,
        table,
        owners,
        positions,
        tiles,
        output,
        logical_regions,
        scale,
        query_tile,
    )


def step5_dsa_native_decode(
    query,
    cache,
    regions,
    table,
    positions,
    active,
    output,
    scale,
    query_len=1,
    clear_inactive=True,
    gate=None,
) -> None:
    """Explicit decode API; the gate keeps the BF16 double-rounding semantics."""
    from vllm_sail.models.step5.ops import dsa_native

    reason = dsa_native.decode_reason(
        query, cache, regions, table, positions, active, output, scale, query_len, gate
    )
    if reason is not None:
        raise ValueError("native decode unsupported: " + reason)
    op = dsa_native.compiled_op("decode")
    if op is None:
        raise RuntimeError("native decode precompiled op is unavailable")
    op(
        query,
        cache,
        regions,
        table,
        positions,
        active,
        output,
        scale,
        query_len,
        clear_inactive,
        gate,
    )


def step5_dsa_native_op(kind: str):
    """Prebuilt step5_dsa_native_<kind> op, or None if not loadable or registered."""
    name = "step5_dsa_native_" + kind
    op = getattr(torch.ops._ppu_C, name, None)
    if op is None and _ext_loaded("_C"):
        op = getattr(torch.ops._ppu_C, name, None)
    return op


def step5_dsa_selector_v2(
    scores,
    positions,
    table,
    workspace,
    regions,
    starts,
    region_block,
    page_size,
    topk,
    packed,
    threads,
) -> None:
    """Step5 selector v2 raw candidate selection (prebuilt op)."""
    _require("_C", "step5_dsa_selector_v2")
    torch.ops._ppu_C.step5_dsa_selector_v2(
        scores,
        positions,
        table,
        workspace,
        regions,
        starts,
        region_block,
        page_size,
        topk,
        packed,
        threads,
    )


def top_k_per_row_prefill_bf16(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    top_k: int,
) -> None:
    """BF16 top-k prefill sampler kernel (fork's ``sampler.cu`` addition,
    on the plugin namespace)."""
    _require("_C", "top_k_per_row_prefill_bf16")
    torch.ops._ppu_C.top_k_per_row_prefill_bf16(
        logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
    )
