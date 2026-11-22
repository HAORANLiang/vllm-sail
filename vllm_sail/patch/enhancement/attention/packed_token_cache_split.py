# SPDX-License-Identifier: Apache-2.0
"""Split packed caches into their kernel blocks.

Kimi-K3.1 packs the DSA indexer pages beside the MLA pages of the same block row
(``KimiK3IndexerBackend`` only supports BLHNC), so a KDA state page sizes the
whole row and ``Platform._align_hybrid_block_size`` raises the manager block
size far past the 64-token pages the indexer kernels address. The indexer
therefore writes its keys at slots derived from the block table
(``KimiK3IndexerMetadataBuilder``): the contract
``AttentionGroup.map_kv_cache_to_kernel_blocks`` describes as "compressed
caches, written from their block table".

The guard there derives that contract from ``tokens_per_state`` alone and
refuses every cache that stores one state per token, so binding the packed
indexer cache fails with "cannot be split into". A backend that declares
``writes_kv_cache_from_block_table`` meets the contract without compression, so
take the same split path for it. Every other cache, and every other platform,
keeps upstream's check.

A dense MLA draft (EAGLE3/DSpark on Kimi-K3.1, retargeted to FLASHMLA on PPU)
adds a second per-token cache to the packed group: its pages ride at the head
of the same block rows, so the indexer's 64-token kernel blocks force a split
of the draft's cache too, and upstream's guard raises there as well. Unlike the
indexer, the draft's layers write through manager-block slot mappings, so the
split alone would misaddress every write: this patch also splits the draft's
cache view and registers its layers in ``_slot_remaps``, and the patched
``build_slot_mappings_by_layer`` rewrites those layers' slot mappings from
manager-block addressing to kernel-block addressing (a manager slot
``b * block_size + o`` addresses kernel page ``b * kernel_block_stride +
o // kernel_block_size`` of the split view, whose flat row index is
``b * span + o`` with ``span = kernel_block_stride * kernel_block_size``).

The rewrite must NOT happen in place on the shared per-group row of the
persistent ``slot_mappings`` buffer: a packed group also serves backends that
keep manager addressing (target FLASHMLA_SPARSE alongside the draft's
force-split FLASHMLA), and they alias the same row, so rewriting it would
corrupt their writes. Instead the patched ``build_slot_mappings_by_layer``
gives remapped layers their own persistent rows (allocated once, so CUDA-graph
capture bakes a stable address into the captured cache-write ops), and each
step recomputes those rows from the freshly computed manager-block slots
(``_compute_slot_mappings_kernel`` rewrites every element of the shared row,
padding included). Layers of one group alias the same shared row, so the
per-step recompute deduplicates by storage address.

Reads need no fix: the metadata builders already consume the kernel block
table that ``AttentionGroup.map_to_kernel_block_table`` derives from the split.

The speculator's context-KV write is a third consumer that bypasses both the
metadata and the per-layer dict: its input-prepare kernel fills
``_context_slot_mappings`` with manager-block slots and
``precompute_and_store_context_kv`` writes the draft's context K/V straight
from that buffer. The patched ``DFlashSpeculator._precompute_context_kv``
remaps those slots onto kernel addressing first (the remap is captured with
the FULL graph, so it runs on every replay).

Every rewrite is gated twice — on PPU and on a non-empty ``_slot_remaps`` —
so a platform, or a model that never force-splits a cache, runs upstream's
code unchanged. Each rewrite carries the ``(block_size, span)`` of the layers
it serves instead of reading one geometry from the global registry, and falls
back with one warning when a group mixes geometries, which one math cannot
address; the speculator path needs a single geometry across every split cache
for the same reason. That patch is also installed imperatively and only on
PPU (see ``_install_precompute_context_kv_patch``), so importing this module
does not pull the speculative-decoding stack into models served without a
draft.

Backends that natively run on the manager blocks (FLASHMLA_SPARSE declares
``MultipleOf(64)``) are never force-split: their group only splits because a
cogroup backend needs smaller kernel pages, and their kernels read manager-sized
pages themselves, so they keep upstream's behavior — including the unmodified
shared slot-mapping row, which is exactly why the remap writes to separate
persistent rows instead of in place.

Known limitation: with ubatching (``num_ubatches > 1``) the runner takes
``ubatch_runner.prepare`` and never calls ``build_slot_mappings_by_layer``, so
the rewrite would silently not apply and MLA writes would misaddress. The
configurations this patch targets (single-ubatch serving, as on Kimi-K3.1 PPU)
do not enable ubatching; do not combine ``--enable-ubatch`` with a force-split
group without porting the rewrite into the ubatch slot-mapping path first.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterable
from dataclasses import replace

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.attn_utils import (
    build_slot_mappings_by_layer as _upstream_build_slot_mappings_by_layer,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_sail.patch.utils import original_of, patch

logger = init_logger(__name__)

_AFFECTED = ">=0.34.0,<0.35.0"

# Captured before patching so the delegating path stays upstream.
_upstream_map_kv_cache_to_kernel_blocks = AttentionGroup.map_kv_cache_to_kernel_blocks
_upstream_with_kernel_block_table = AttentionGroup._with_kernel_block_table

_reported: set[str] = set()

#: Per shared-row storage, a persistent private row holding the remapped
#: (kernel-addressed) slots for the force-split layers that alias it. Allocated
#: once per storage so CUDA graphs capture a stable address; values are
#: recomputed every step from the shared row.
_remapped_rows: dict[int, torch.Tensor] = {}

#: Layers whose cache view was split while their writes go through
#: manager-block slot mappings, mapped to ``(block_size, span)``: a manager
#: slot ``s`` addresses kernel page ``s // block_size`` and row
#: ``(s // block_size) * span + s % block_size`` of the split view, with
#: ``span = kernel_block_stride * kernel_block_size``. Populated by the
#: force-split below and consumed by all three write paths (the per-layer dict
#: from ``build_slot_mappings_by_layer``, the metadata swap in
#: ``_with_kernel_block_table``, and the speculator's context slots).
#: Consumers pass their own geometry (``_group_geometry`` /
#: ``_unique_geometry``) rather than reading one entry from here.
_slot_remaps: dict[str, tuple[int, int]] = {}

#: Per speculator, the kernel-addressed twin of its ``_context_slot_mappings``
#: buffer. The speculator's input-prepare kernel writes manager-block slots
#: there, and the context-KV write consumes that buffer directly (it never
#: passes through ``build_slot_mappings_by_layer``), so the split draft cache
#: needs this remapped copy.
_context_slot_rows: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _report_once(key: str, message: str, *args, warn: bool = True) -> None:
    """One line per backend, so a split that falls back stays diagnosable."""
    if key in _reported:
        return
    _reported.add(key)
    (logger.warning if warn else logger.info)(message, *args)


def _group_geometry(layer_names: Iterable[str]) -> tuple[int, int] | None:
    """The single ``(block_size, span)`` a group's force-split layers share.

    ``None`` when the group has no force-split layer, or when it mixes
    geometries: one rewrite cannot address both, so the caller keeps upstream
    behavior for the whole group instead of mis-addressing the layers whose
    geometry it guessed wrong.
    """
    geometries = {_slot_remaps[name] for name in layer_names if name in _slot_remaps}
    if len(geometries) > 1:
        _report_once(
            f"mixed-geometry:{sorted(geometries)}",
            "Force-split layers share one metadata row but use different "
            "kernel geometries %s; their slot mappings stay on manager-block "
            "addressing and misaddress the split cache.",
            sorted(geometries),
        )
        return None
    return next(iter(geometries)) if geometries else None


def _unique_geometry() -> tuple[int, int] | None:
    """The one geometry every force-split group shares.

    The speculator's context-slot buffer is addressed per draft group, not per
    layer, so it can only be remapped when the split caches agree — use
    :func:`_group_geometry` where the layers are known.
    """
    geometries = set(_slot_remaps.values())
    if len(geometries) > 1:
        _report_once(
            f"mixed-geometry-context:{sorted(geometries)}",
            "Force-split caches use different kernel geometries %s; the "
            "speculator's context slots stay on manager-block addressing.",
            sorted(geometries),
        )
        return None
    return next(iter(geometries)) if geometries else None


def _force_split_token_cache(self, kv_cache: torch.Tensor) -> torch.Tensor | None:
    """Split a slot-mapping-written token cache whose group was forced onto
    smaller kernel pages than its manager blocks.

    A cogroup backend that writes from its block table (the packed DSA
    indexer) drags the whole group's kernel block size below the manager
    block size. A token cache sharing the packed block row then needs the
    same split to bind at all: its reads already go through the kernel block
    table each backend group derives for itself, and its writes are fixed by
    registering the layers in ``_slot_remaps`` for the patched
    ``build_slot_mappings_by_layer``. Returns ``None`` when no split is
    required or the layout is not addressable, so the caller keeps upstream's
    behavior (and error message).
    """
    assert self.kernel_block_size is not None
    spec = self.kv_cache_spec
    if spec.tokens_per_state != 1 or self.kernel_block_size == spec.block_size:
        return None
    kernel_rows = spec.get_num_kernel_states(self.kernel_block_size)
    if kv_cache.shape[-2] == kernel_rows:
        return None
    if kv_cache.shape[1] != 1 or kv_cache.stride(0) % (
        kernel_rows * kv_cache.stride(-2)
    ):
        return None
    # Same geometry as the indexer's split: kernel block j of block b is
    # ``b * kernel_block_stride + j``. The view spans the other layers' pages
    # between blocks (never addressing them), which ``view`` cannot express.
    page_stride = kernel_rows * kv_cache.stride(-2)
    self.kernel_block_stride = kv_cache.stride(0) // page_stride
    num_pages = (kv_cache.shape[0] - 1) * self.kernel_block_stride + (
        kv_cache.shape[-2] // kernel_rows
    )
    span = self.kernel_block_stride * kernel_rows
    for name in self.layer_names:
        _slot_remaps[name] = (spec.block_size, span)
    backend_name = self.backend.get_name()
    _report_once(
        f"split:{backend_name}",
        "Force-split %s token cache into %d-token kernel blocks (a cogroup "
        "backend needs smaller kernel pages): block_size=%d shape=%s "
        "strides=%s pages=%d kernel_block_stride=%d; slot mappings of its "
        "%d layer(s) are remapped.",
        backend_name,
        self.kernel_block_size,
        spec.block_size,
        tuple(kv_cache.shape),
        kv_cache.stride(),
        num_pages,
        self.kernel_block_stride,
        len(self.layer_names),
    )
    return kv_cache.as_strided(
        (num_pages, *kv_cache.shape[1:-2], kernel_rows, kv_cache.shape[-1]),
        (page_stride, *kv_cache.stride()[1:]),
    )


@patch(
    "vllm.v1.worker.gpu.attn_utils",
    "build_slot_mappings_by_layer",
    reason=(
        "A token cache force-split into kernel blocks (see the packed DSA "
        "group's indexer split) is addressed by kernel pages, but its layers "
        "write through manager-block slot mappings, so every write would land "
        "in the wrong row. Those layers get their own persistent rows with "
        "slot mappings rewritten from manager-block to kernel-block addressing; "
        "the rows are persistent so CUDA graphs keep reading the buffer they "
        "captured, and separate from the shared row because cogroup backends "
        "with manager addressing alias it."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "map_kv_cache_to_kernel_blocks (or the layers) rewrites slot mappings "
        "for split token caches into per-layer rows."
    ),
)
def build_slot_mappings_by_layer(slot_mappings, kv_cache_config):
    by_layer = _upstream_build_slot_mappings_by_layer(slot_mappings, kv_cache_config)
    if not _slot_remaps or not current_platform.is_ppu():
        return by_layer
    # Layers of one KV cache group alias the same row of the persistent
    # slot-mappings buffer; recompute each distinct storage once per step.
    seen_storages: set[int] = set()
    for name, geometry in _slot_remaps.items():
        slot_mapping = by_layer.get(name)
        if slot_mapping is None:
            continue
        if slot_mapping.data_ptr() in seen_storages:
            continue
        seen_storages.add(slot_mapping.data_ptr())
        # A manager slot ``s`` addresses kernel row ``(s // block_size) * span
        # + s % block_size`` of the split view. PAD slots stay negative so the
        # cache-write kernels keep skipping them. The remapped slots go to a
        # persistent private row, NOT in place: cogroup backends that keep
        # manager addressing (e.g. the target's FLASHMLA_SPARSE) alias the
        # same row and must keep seeing the manager-block slots. The private
        # row's address is stable, so graphs captured against it stay valid,
        # and its values are recomputed from the shared row every step.
        private_row = _remapped_private_row(slot_mapping, geometry)
        for other, mapping in by_layer.items():
            if (
                other in _slot_remaps
                and mapping.data_ptr() == slot_mapping.data_ptr()
            ):
                by_layer[other] = private_row
    return by_layer


def _remapped_private_row(
    slot_mapping: torch.Tensor, geometry: tuple[int, int]
) -> torch.Tensor:
    """Recompute the kernel-addressed twin of a shared manager-slot row.

    ``geometry`` is the ``(block_size, span)`` of the layers that consume this
    row (see :func:`_group_geometry`): a parameter rather than a global lookup,
    so a process that split caches with different geometries cannot remap one
    with another's span.

    Idempotent per step: both consumers (the per-layer dict update and the
    metadata swap below) call this on the same shared row, in whatever order
    they run in (the speculator's prepare builds metadata before refreshing
    the per-layer dict), so neither can observe the other's stale values.
    """
    block_size, span = geometry
    private_row = _remapped_rows.get(slot_mapping.data_ptr())
    if private_row is None or private_row.shape != slot_mapping.shape:
        private_row = torch.empty_like(slot_mapping)
        _remapped_rows[slot_mapping.data_ptr()] = private_row
    remapped = (slot_mapping // block_size) * span + slot_mapping % block_size
    private_row.copy_(torch.where(slot_mapping < 0, slot_mapping, remapped))
    return private_row


@patch(
    "vllm.v1.worker.utils",
    "AttentionGroup._with_kernel_block_table",
    reason=(
        "A token cache force-split into kernel blocks is bound as the split "
        "view, so its cache-write kernels must receive kernel-addressed "
        "slots. The cache-write kernels consume CommonAttentionMetadata's "
        "group-wide slot_mapping row (manager-block slots, shared with "
        "cogroup backends that keep manager addressing), not the per-layer "
        "dict that build_slot_mappings_by_layer returns — so the metadata's "
        "slot_mapping must be swapped for the remapped private row for the "
        "force-split groups only."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "map_kv_cache_to_kernel_blocks (or the metadata builders) rewrites "
        "slot mappings for split token caches."
    ),
)
def _with_kernel_block_table(self, common_attn_metadata, ubatch_idx=0):
    metadata = _upstream_with_kernel_block_table(self, common_attn_metadata, ubatch_idx)
    if not _slot_remaps or not current_platform.is_ppu():
        return metadata
    geometry = _group_geometry(self.layer_names)
    if self.kernel_block_size is None or geometry is None:
        return metadata
    # A group with force-split layers writes its cache through the group's
    # shared slot-mapping row; swap it for the remapped private row. Recomputed
    # here from the shared row so ordering versus the per-layer refresh does
    # not matter. PAD slots stay negative so cache-write kernels skip them.
    slot_mapping = getattr(metadata, "slot_mapping", None)
    if slot_mapping is None:
        return metadata
    metadata = replace(
        metadata, slot_mapping=_remapped_private_row(slot_mapping, geometry)
    )
    return metadata


def _remapped_context_slots(
    speculator, group_idx: int, start: int, end: int, geometry: tuple[int, int]
) -> torch.Tensor:
    """Kernel-addressed twin of a speculator's context slot rows."""
    source = speculator._context_slot_mappings[group_idx][start:end]
    rows = _context_slot_rows.get(speculator)
    if rows is None or rows.shape != speculator._context_slot_mappings.shape:
        rows = torch.empty_like(speculator._context_slot_mappings)
        _context_slot_rows[speculator] = rows
    target = rows[group_idx][start:end]
    block_size, span = geometry
    remapped = (source // block_size) * span + source % block_size
    target.copy_(torch.where(source < 0, source, remapped))
    return target


_DFLASH_CONTEXT_KV_TARGET = (
    "vllm.v1.worker.gpu.spec_decode.dflash.speculator"
    ".DFlashSpeculator._precompute_context_kv"
)

_dflash_precompute_upstream: Callable | None = None


def _precompute_context_kv(self, start: int, end: int, dummy_run: bool = False):
    global _dflash_precompute_upstream
    upstream = _dflash_precompute_upstream
    if upstream is None:
        upstream = original_of(_precompute_context_kv, _DFLASH_CONTEXT_KV_TARGET)
        _dflash_precompute_upstream = upstream
    if dummy_run or not _slot_remaps or not current_platform.is_ppu():
        return upstream(self, start, end, dummy_run)
    geometry = _unique_geometry()
    if geometry is None:
        return upstream(self, start, end, dummy_run)
    if self._layer_group_idx is not None:
        context_slots = [
            _remapped_context_slots(self, gidx, start, end, geometry)
            for gidx in self._layer_group_idx
        ]
    else:
        context_slots = _remapped_context_slots(self, 0, start, end, geometry)
    self.model.precompute_and_store_context_kv(
        self.hidden_states[start:end],
        self.context_positions[start:end],
        context_slots,
    )


def _install_precompute_context_kv_patch() -> None:
    """Install the context-slot remap on PPU, best effort.

    Importing the speculator pulls vLLM's speculative-decoding stack into the
    process, so a platform that can never hit the remap does not pay for it;
    and the import is guarded by ``Exception`` because an optional module that
    cannot load must not fail this patch module (and with it the plugin) for
    every model — only DSpark loses the remap.
    """
    if not current_platform.is_ppu():
        return
    try:
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
            DFlashSpeculator,
        )
    except Exception:  # noqa: BLE001 - optional module, must not break loading
        _report_once(
            "dflash-unavailable",
            "Context-slot remap not installed: the DFlash speculator module "
            "is unavailable on this build.",
            warn=False,
        )
        return
    if not hasattr(DFlashSpeculator, "_precompute_context_kv"):
        _report_once(
            "dflash-context-kv-missing",
            "Context-slot remap not installed: DFlashSpeculator no longer "
            "defines _precompute_context_kv.",
            warn=False,
        )
        return
    patch(
        "vllm.v1.worker.gpu.spec_decode.dflash.speculator",
        "DFlashSpeculator._precompute_context_kv",
        reason=(
            "A speculator whose draft cache was force-split into kernel blocks "
            "writes its context KV through slots that its input-prepare kernel "
            "fills with manager-block addresses, bypassing the per-layer slot "
            "mappings this patch rewrites. Remap the context slots onto kernel "
            "addressing (PAD slots stay negative) so the context KV lands on "
            "the draft's split cache view."
        ),
        affected_versions=_AFFECTED,
        remove_when=(
            "a force-split token cache has its slot mappings rewritten by "
            "the speculator's own input-prepare kernel."
        ),
    )(_precompute_context_kv)


_install_precompute_context_kv_patch()


@patch(
    "vllm.v1.worker.utils",
    "AttentionGroup.map_kv_cache_to_kernel_blocks",
    reason=(
        "Kimi-K3.1's packed DSA indexer cache holds one state per token and is "
        "written from its block table, but the guard infers that from "
        "tokens_per_state instead of consulting writes_kv_cache_from_block_table, "
        "so the packed cache cannot be split into the 64-token kernel blocks its "
        "kernels address and KV cache binding fails."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "map_kv_cache_to_kernel_blocks asks the backend "
        "(writes_kv_cache_from_block_table) instead of inferring it from "
        "tokens_per_state."
    ),
)
def map_kv_cache_to_kernel_blocks(self, kv_cache):
    # PPU MODIFICATION: begin — upstream infers the compressed-cache contract
    # from ``tokens_per_state``; here the backend declares it, so a cache that
    # stores one state per token can still be split. The shape and stride tests
    # below still decide whether the split is addressable, and an unaddressable
    # cache keeps upstream's error. Every other platform keeps upstream's check.
    if not current_platform.is_ppu():
        return _upstream_map_kv_cache_to_kernel_blocks(self, kv_cache)
    writes_from_block_table = getattr(
        self.backend, "writes_kv_cache_from_block_table", None
    )
    # A backend that spells the contract as a plain bool keeps upstream's
    # check: calling it here would crash KV cache binding for every group
    # on this platform, not just the packed indexer cache.
    declared = (
        bool(writes_from_block_table())
        if callable(writes_from_block_table)
        else None
    )
    backend_name = self.backend.get_name()
    if not declared:
        # Only a cache with no contract attribute at all (FLASHMLA's token
        # cache) may be force-split; a backend that spells the contract as a
        # plain bool opted out of the split path explicitly.
        force_split = (
            _force_split_token_cache(self, kv_cache)
            if writes_from_block_table is None
            else None
        )
        if force_split is not None:
            return force_split
        _report_once(
            f"skip:{backend_name}",
            "Packed kernel-block split skipped for %s "
            "(writes_kv_cache_from_block_table=%r).",
            backend_name,
            declared,
        )
        return _upstream_map_kv_cache_to_kernel_blocks(self, kv_cache)
    assert self.kernel_block_size is not None
    spec = self.kv_cache_spec
    kernel_rows = spec.get_num_kernel_states(self.kernel_block_size)
    if kv_cache.shape[-2] == kernel_rows:
        return kv_cache
    if kv_cache.shape[1] != 1 or kv_cache.stride(0) % (
        kernel_rows * kv_cache.stride(-2)
    ):
        # Warning, not info: upstream's message blames the block size, but an
        # unaddressable layout (strided rows, another KVCacheLayout) is the
        # usual cause, and this line is the only place that says so.
        _report_once(
            f"unsplittable:{backend_name}",
            "Packed %s cache cannot be split into %d-token kernel blocks "
            "(shape=%s, strides=%s); reporting upstream's error, which names "
            "the block size even when the layout is the cause.",
            backend_name,
            self.kernel_block_size,
            tuple(kv_cache.shape),
            kv_cache.stride(),
            warn=True,
        )
        return _upstream_map_kv_cache_to_kernel_blocks(self, kv_cache)
    # Kernel block j of block b is b * kernel_block_stride + j. The view spans
    # the other layers' pages between blocks (never addressing them), which
    # ``view`` cannot express.
    page_stride = kernel_rows * kv_cache.stride(-2)
    self.kernel_block_stride = kv_cache.stride(0) // page_stride
    num_pages = (kv_cache.shape[0] - 1) * self.kernel_block_stride + (
        kv_cache.shape[-2] // kernel_rows
    )
    _report_once(
        f"split:{backend_name}",
        "Split %s cache into %d-token kernel blocks: block_size=%d shape=%s "
        "strides=%s pages=%d kernel_block_stride=%d.",
        backend_name,
        self.kernel_block_size,
        spec.block_size,
        tuple(kv_cache.shape),
        kv_cache.stride(),
        num_pages,
        self.kernel_block_stride,
    )
    split = kv_cache.as_strided(
        (num_pages, *kv_cache.shape[1:-2], kernel_rows, kv_cache.shape[-1]),
        (page_stride, *kv_cache.stride()[1:]),
    )
    # PPU MODIFICATION: end
    return split
