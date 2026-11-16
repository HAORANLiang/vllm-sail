# SPDX-License-Identifier: Apache-2.0
"""PPU patch: shape-aware ``permute_on_load`` for per-channel weight scales.

Upstream's fused mega-attention ``__init__`` registers five load-time
permutations through :func:`permute_on_load` (``vllm/models/deepseek_v41/
nvidia/flash_mla_mega_attn.py``), one of them on ``wo_a.weight_scale`` with
``o_fused_chunk_permutation`` — the per-32-block form of the O layout, whose
index list has ``head_dim // 32`` (128) entries. That is exactly right for an
MXFP8 block-scale checkpoint, where every output row carries one scale per
32-element K block.

An FP8 weight-per-channel checkpoint (``fp8_channelwise_layers``, loaded as
``CompressedTensorsW8A8Fp8`` with ``QuantizationStrategy.CHANNEL``) carries one
FP32 scale per output row instead: ``wo_a.weight_scale`` is ``[rows, 1]`` (or
``[rows]``), so the K axis has a single entry. Feeding the 128-entry
permutation to that axis cannot work — on CUDA the out-of-bounds gather
silently returns a widened tensor and the subsequent ``copy_`` fails with
``output with shape [rows, 1] doesn't match the broadcast shape [rows, 128]``,
which is what aborted every worker before the model finished loading.

The permutation moves 32-element K chunks; a per-row value is constant across K
and therefore unchanged by it, so the correct action is to **skip** the
gather (identity), not to reshape or broadcast the scale. FlashMLA's own
``permute_wv_proj`` documents the same invariant for the runtime WV layout
(``channel_wise ? 1 : hidden/(gran*4)`` scale factors, "a per-channel scale is
unchanged by this move").

Patching the *consumer* module is required, not the source one: upstream does
``from ...fused_layout import permute_on_load``, so it holds a by-value alias in
its own module namespace. Every call site reads that module global, and the
model is constructed after ``register_out_of_tree`` applies the patches, so one
attribute replacement covers all five registrations.

Non-per-channel parameters keep the original behaviour unchanged: the two
canonical geometries (``wq_b.weight`` rows 4096, ``wq_b.weight_scale`` rows
4096, ``wo_a.weight`` K 4096, ``wo_a.weight_scale`` blocks 128) all satisfy
``axis == len(perm)`` and are delegated untouched. Any geometry that is neither
canonical nor a per-channel scale raises: that means caller and permutation
disagree about the layout, and loading the shard would corrupt the weights
silently.
"""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.models.deepseek_v41.nvidia import flash_mla_mega_attn as _upstream_mega

from vllm_sail.patch.utils import patch

_MODULE = "vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn"
_AFFECTED_VERSIONS = ">=0.32.0,<0.33.0"

logger = init_logger(__name__)

_upstream_permute_on_load = _upstream_mega.permute_on_load


def scale_permute_action(axis: int, perm_length: int) -> str:
    """What one ``permute_on_load`` registration should do, by geometry.

    ``axis`` is the size of the parameter's permuted dimension and
    ``perm_length`` the number of entries in the index list. Returns
    ``"apply"`` for the layout the permutation was written for (one entry per
    element) and ``"skip"`` for a per-channel scale (a size-1 axis under a
    multi-entry chunk permutation, which leaves a per-row value unchanged).
    Anything else is a layout mismatch and raises.
    """
    if axis == perm_length:
        return "apply"
    if axis == 1 and perm_length > 1:
        return "skip"
    raise AssertionError(
        f"permute_on_load geometry mismatch: the parameter's permuted axis has "
        f"{axis} entries but the permutation has {perm_length}. Known layouts "
        f"are one-entry-per-element (apply) and a per-channel scale "
        f"(axis == 1, permutation skipped); refusing to guess, because loading "
        f"the shard through a wrong permutation corrupts the weights silently."
    )


@patch(
    _MODULE,
    "permute_on_load",
    reason=(
        "FP8 weight-per-channel checkpoints (fp8_channelwise_layers) store one "
        "scale per output row, so wo_a.weight_scale has a size-1 K axis while "
        "upstream unconditionally registers the 128-entry per-32 chunk "
        "permutation on it; model loading aborts before the first forward. A "
        "per-row scale is invariant under the K-chunk permutation, so skipping "
        "it is exact."
    ),
    affected_versions=_AFFECTED_VERSIONS,
    remove_when=(
        "upstream permutes wo_a.weight_scale only when the scale is an MXFP8 "
        "block layout (or selects the permutation from the quant scheme)."
    ),
)
def permute_on_load(param, perm, dim) -> None:
    """Skip the K-chunk permutation on per-channel scales, else delegate."""
    axis = param.shape[dim]
    action = scale_permute_action(axis, perm.numel())
    if action == "skip":
        logger.info_once(
            "[VLLM_SAIL] permute_on_load: skipping the 32-chunk K permutation "
            "on a per-channel scale (axis %d entries=%d, permutation "
            "entries=%d); a per-row scale is invariant under it.",
            dim,
            axis,
            perm.numel(),
        )
        return
    _upstream_permute_on_load(param, perm, dim)
