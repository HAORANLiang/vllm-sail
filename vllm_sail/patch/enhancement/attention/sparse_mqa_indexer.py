# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: F821
"""Enable the candidate-only indexer with SAIL DeepGEMM and DeepSelect.

Constructors copied from vLLM 8b10bda57ceab3a741d2fb3718af48cef691023e.
Their SM100 checks have no platform hook. Keep the remaining initialization,
forward, candidate expansion, metadata builder and top-k/remapping upstream.
"""

from __future__ import annotations

import functools
import importlib

from vllm.model_executor.layers import sparse_mqa_indexer as _layer
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla import sparse_indexer as _metadata

from vllm_sail.attention.ops import sparse_mqa as _sail
from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.34.0,<0.35.0"
_REASON = "PPU M890 uses SAIL sparse MXFP4 logits and DeepSelect through the upstream interface."
_REMOVE_WHEN = "The sparse indexer exposes platform capability and DeepGEMM provider hooks."
_LAYER = "vllm.model_executor.layers.sparse_mqa_indexer"
_METADATA = "vllm.v1.attention.backends.mla.sparse_indexer"


def _layer_body(fn):
    return bind_body(fn, _layer)


def _metadata_body(fn):
    return bind_body(fn, _metadata)


@patch(
    _LAYER,
    "SparseMQAIndexer.__init__",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
@_layer_body
def sparse_mqa_indexer_init(
    self,
    k_cache,
    topk_tokens: int,
    head_dim: int,
    max_total_seq_len: int,
    topk_indices_buffer: torch.Tensor,
    candidate_blocks: torch.Tensor,
    candidate_block_size: int,
):
    # PPU MODIFICATION: begin
    super(SparseMQAIndexer, self).__init__()
    if current_platform.is_ppu():
        from vllm_sail.attention.ops.sparse_mqa import check_sparse_indexer_support

        check_sparse_indexer_support(get_current_vllm_config())
    elif not (
        # PPU MODIFICATION: end
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and has_deep_gemm_sparse_mqa()
    ):
        raise ValueError(
            "SparseMQAIndexer requires an SM100-class GPU and DeepGEMM >= 2.8."
        )
    if not has_deep_select():
        raise ValueError(
            "SparseMQAIndexer requires the DeepSelect top-k extension "
            "(vllm._deepselect_C)."
        )
    self.k_cache = k_cache
    self.topk_tokens = topk_tokens
    self.head_dim = head_dim
    self.max_total_seq_len = max_total_seq_len
    self.topk_indices_buffer = topk_indices_buffer
    self.candidate_blocks = candidate_blocks
    self.candidate_block_size = candidate_block_size
    self.use_fp4_cache = dsa_indexer_uses_fp4(get_current_vllm_config())


@patch(
    _METADATA,
    "DeepseekV41SparseIndexerMetadataBuilder.__init__",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
@_metadata_body
def sparse_metadata_builder_init(
    self,
    kv_cache_spec: AttentionSpec,
    layer_names: list[str],
    vllm_config: VllmConfig,
    device: torch.device,
    **kwargs,
) -> None:
    # PPU MODIFICATION: begin
    super(DeepseekV41SparseIndexerMetadataBuilder, self).__init__(
        kv_cache_spec, layer_names, vllm_config, device, **kwargs
    )
    # PPU MODIFICATION: end
    hf_config = vllm_config.model_config.hf_text_config
    candidate_block_size = getattr(hf_config, "candidate_block_size", 0)
    candidate_topk_blocks = getattr(hf_config, "candidate_topk_blocks", 0)
    topk_tokens = getattr(hf_config, "index_topk", 0)
    prefix = "attention_config.indexer_sparse_logits"
    if candidate_block_size <= 0 or candidate_topk_blocks <= 0:
        raise ValueError(
            f"{prefix} requires a two-level (candidate block) indexer model "
            "such as DeepSeek-V4.1."
        )
    # PPU MODIFICATION: begin
    if current_platform.is_ppu():
        from vllm_sail.attention.ops.sparse_mqa import check_sparse_indexer_support

        check_sparse_indexer_support(vllm_config)
    elif not (
        # PPU MODIFICATION: end
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
    ):
        raise ValueError(f"{prefix} requires an SM100-class GPU.")
    if not has_deep_gemm_sparse_mqa():
        raise ValueError(
            f"{prefix} requires DeepGEMM >= 2.8 "
            "(fp8_fp4_sparse_mqa_logits not found)."
        )
    if self.dcp_world_size > 1 or self.use_pcp:
        raise NotImplementedError(
            f"{prefix} is not supported with decode/prefill context parallel."
        )
    if not self.supports_varlen:
        raise ValueError(
            f"{prefix} requires the varlen paged MQA logits decode layout."
        )
    # The top-k runs on the kernels' bf16 logits with DeepSelect; there is
    # no fp32 fallback, so the other top-k backends cannot be honored.
    topk_backend = vllm_config.kernel_config.sparse_indexer_topk_backend
    if not has_deep_select():
        raise ValueError(
            f"{prefix} requires the DeepSelect top-k extension "
            "(vllm._deepselect_C) for this GPU."
        )
    if topk_backend not in ("auto", "deep_select"):
        raise ValueError(
            f"{prefix} uses DeepSelect for the top-k; "
            f"kernel_config.sparse_indexer_topk_backend='{topk_backend}' "
            "is not supported with it."
        )
    self.sparse_block_kv = pick_sparse_block_kv(candidate_block_size)
    if kv_cache_spec.block_size % self.sparse_block_kv != 0:
        raise ValueError(
            f"{prefix}: indexer cache pages ({kv_cache_spec.block_size} "
            f"tokens) must be a multiple of {self.sparse_block_kv} tokens."
        )
    self.num_sparse_blocks = candidate_topk_blocks * (
        candidate_block_size // self.sparse_block_kv
    )
    num_sparse_cols = self.num_sparse_blocks * self.sparse_block_kv
    check_deep_select_layout(num_sparse_cols, topk_tokens)
    # Sparse logits are bf16 [rows, num_sparse_cols]; make the prefill
    # chunker (which budgets fp32 [rows, seq_len]) size chunks as if every
    # row were this wide.
    self.min_split_seq_len = (num_sparse_cols * 2 + 3) // 4
    # PPU MODIFICATION: begin
    if current_platform.is_ppu():
        # SAIL returns a logical-width view over 512-column-aligned BF16
        # rows. Budget the physical stride, including for narrow candidates.
        padded_cols = (num_sparse_cols + 511) // 512 * 512
        self.min_split_seq_len = (padded_cols * 2 + 3) // 4
    # PPU MODIFICATION: end

    # Rows are indexed by batch token (decode rows first, then prefill
    # chunks by token range), like ``arange_buffer``.
    max_rows = self.arange_buffer.shape[0]
    int32 = dict(dtype=torch.int32, device=device)
    self.sparse_indices_buffer = torch.zeros(
        (max_rows, self.num_sparse_blocks), **int32
    )
    self.end_buffer = torch.zeros(max_rows, **int32)
    self.col_indices_buffer = torch.zeros((max_rows, topk_tokens), **int32)
    self.zero_row_ks = torch.zeros(max_rows, **int32)
    logger.info_once(
        "DeepSeek V4.1 indexer: candidate consumers score only the candidate "
        "blocks with DeepGEMM sparse MQA logits (%d sparse blocks of %d "
        "tokens per row).",
        self.num_sparse_blocks,
        self.sparse_block_kv,
    )


# Bind the importing modules, including modules loaded before this patch.
# Higher-level scorer functions keep their live upstream module globals.
_REDIRECTED = {
    "vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits": (
        "get_sparse_mqa_logits_metadata",
        "get_paged_sparse_mqa_logits_metadata",
        "fp8_fp4_sparse_mqa_logits",
        "fp8_fp4_paged_sparse_mqa_logits",
    ),
    "vllm.model_executor.layers.sparse_mqa_indexer": (
        "has_deep_gemm_sparse_mqa",
    ),
    "vllm.v1.attention.backends.mla.sparse_indexer": (
        "has_deep_gemm_sparse_mqa",
    ),
}


def _dispatch(original, replacement):
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        if current_platform.is_ppu():
            return replacement(*args, **kwargs)
        return original(*args, **kwargs)

    return wrapped


def _install():
    for module_name, names in _REDIRECTED.items():
        module = importlib.import_module(module_name)
        for name in names:
            patch(
                module_name,
                name,
                reason=_REASON,
                affected_versions=_AFFECTED,
                remove_when=_REMOVE_WHEN,
            )(_dispatch(getattr(module, name), getattr(_sail, name)))


_install()
