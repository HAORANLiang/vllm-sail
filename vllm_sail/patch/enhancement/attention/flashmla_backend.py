# SPDX-License-Identifier: Apache-2.0
"""PPU patches for ``vllm.v1.attention.backends.mla.flashmla`` (dense FlashMLA).

The fork makes three changes, all PPU-gated; each is applied here by delegation
rather than a body copy:

* ``FlashMLABackend.supports_compute_capability`` accepts capability major 8 on
  PPU (upstream: 9 and 10).
* ``FlashMLAMetadataBuilder.__init__`` sizes the CUDA-graph tile-scheduler
  buffer from a hard-coded ``num_sms = 320`` on PPU ("PPU FIXME (kai):
  temporarily use hard-coded ppu magic number, this is align with ppu mla
  algo"). The upstream ``__init__`` is run first, then the buffer is
  re-allocated with the PPU size.

The fork's third change — the ``[FM_FMHA]`` NVTX range around the decode kernel
in ``FlashMLAImpl.forward_mqa`` — is not patched here: the plugin emits it from
``vllm.v1.attention.ops.flashmla.flash_mla_with_kvcache`` (see
``flashmla_ops``), i.e. around the very call the fork brackets, without wrapping
``forward_mqa`` or handling its untuple ``q``.

A plugin-only fourth patch, ``FlashMLAMetadataBuilder._build_decode``, enables
the FP8 dense path on PPU. Upstream's FP8 branch calls
``get_mla_metadata_dense_fp8`` for int32 tile-scheduler tensors and copies them
into CUDA-graph buffers; PPU builds no ``_flashmla_extension_C`` and the sail
kernel instead fills its metadata lazily from a ``FlashMLASchedMeta``. On PPU FP8
the patch therefore delegates to upstream's non-FP8 branch (which already builds
``scheduler_metadata`` as a sail ``FlashMLASchedMeta``) and attaches a lazy sched
meta to ``scheduler_metadata.tile_scheduler_metadata`` — the field
``forward_mqa``'s FP8 branch forwards to ``flash_mla_with_kvcache_fp8`` — leaving
``num_splits`` ``None`` as the sail kernel requires.
"""

from __future__ import annotations

import torch
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla import flashmla as _flashmla_backend

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.32.0,<0.33.0"
_MODULE = "vllm.v1.attention.backends.mla.flashmla"

#: Hard-coded SM count the fork uses for the PPU MLA scheduler buffers.
_PPU_NUM_SMS = 320

_upstream_supports_compute_capability = (
    _flashmla_backend.FlashMLABackend.supports_compute_capability
)
_upstream_builder_init = _flashmla_backend.FlashMLAMetadataBuilder.__init__
_upstream_build_decode = _flashmla_backend.FlashMLAMetadataBuilder._build_decode


@patch(
    _MODULE,
    "FlashMLABackend.supports_compute_capability",
    reason=(
        "PPU runs FlashMLA on capability major 8. Upstream accepts only majors "
        "9 and 10, so without this PPU could never select the FlashMLA backend."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream accepts PPU's capability for FlashMLA natively.",
)
def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
    if current_platform.is_ppu():
        return capability.major in [8]
    return _upstream_supports_compute_capability(capability)


@patch(
    _MODULE,
    "FlashMLAMetadataBuilder.__init__",
    reason=(
        "PPU FIXME (kai): the fork hard-codes num_sms = 320 for the PPU MLA "
        "tile-scheduler buffers ('temporarily use hard-coded ppu magic number, "
        "this is align with ppu mla algo') instead of querying the device."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "PPU reports a compute-unit count the FlashMLA scheduler can use "
        "directly, and the fork's hard-coded 320 is removed."
    ),
)
def builder_init(
    self,
    kv_cache_spec,
    layer_names,
    vllm_config,
    device: torch.device,
) -> None:
    _upstream_builder_init(self, kv_cache_spec, layer_names, vllm_config, device)
    if current_platform.is_ppu() and self.cg_buf_tile_scheduler_metadata is not None:
        self.cg_buf_tile_scheduler_metadata = torch.zeros(
            (_PPU_NUM_SMS, 8),
            device=self.device,
            dtype=torch.int32,
        )


@patch(
    _MODULE,
    "FlashMLAMetadataBuilder._build_decode",
    reason=(
        "PPU builds no _flashmla_extension_C, so upstream's FP8 dense metadata "
        "path (get_mla_metadata_dense_fp8 -> int32 tensors -> CUDA-graph copies) "
        "cannot run. The sail dense FP8 kernel fills its tile-scheduler metadata "
        "lazily from a FlashMLASchedMeta, the same object the non-FP8 "
        "get_mla_metadata returns. On PPU FP8 this delegates to upstream's "
        "non-FP8 branch to build scheduler_metadata, then attaches a lazy sched "
        "meta (from the patched get_mla_metadata_dense_fp8) to "
        "scheduler_metadata.tile_scheduler_metadata, which forward_mqa's FP8 "
        "branch forwards to flash_mla_with_kvcache_fp8; num_splits stays None as "
        "the sail kernel requires."
    ),
    affected_versions=_AFFECTED,
    remove_when="PPU builds the FP8 dense FlashMLA metadata extension natively.",
)
def _build_decode(
    self,
    block_table_tensor,
    seq_lens_device,
    max_seq_len,
    query_start_loc_cpu,
    query_start_loc_device,
    num_decode_tokens,
    max_query_len,
    dcp_tot_seq_lens_device,
):
    kwargs = dict(
        block_table_tensor=block_table_tensor,
        seq_lens_device=seq_lens_device,
        max_seq_len=max_seq_len,
        query_start_loc_cpu=query_start_loc_cpu,
        query_start_loc_device=query_start_loc_device,
        num_decode_tokens=num_decode_tokens,
        max_query_len=max_query_len,
        dcp_tot_seq_lens_device=dcp_tot_seq_lens_device,
    )
    if not (current_platform.is_ppu() and self.is_fp8_kvcache):
        return _upstream_build_decode(self, **kwargs)

    # Run upstream's non-FP8 branch: get_mla_metadata (sail) already returns a
    # lazy FlashMLASchedMeta and the FP8 int32/CUDA-graph block is skipped.
    # is_fp8_kvcache only gates that block and the sail-ignored flag passed to
    # get_mla_metadata, so flipping it for the delegated call is safe; the
    # branch is entered only when it is True, so restore it to True afterwards.
    self.is_fp8_kvcache = False
    try:
        metadata = _upstream_build_decode(self, **kwargs)
    finally:
        self.is_fp8_kvcache = True

    from vllm.v1.attention.ops.flashmla import get_mla_metadata_dense_fp8

    # The sail get_mla_metadata ignores its args and returns a fresh lazy
    # FlashMLASchedMeta; forward_mqa's FP8 branch reads it back off this field.
    fp8_sched, _ = get_mla_metadata_dense_fp8(seq_lens_device, 1, 1)
    metadata.scheduler_metadata.tile_scheduler_metadata = fp8_sched
    metadata.scheduler_metadata.num_splits = None
    return metadata
