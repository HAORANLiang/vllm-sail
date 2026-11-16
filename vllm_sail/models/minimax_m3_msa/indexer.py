# SPDX-License-Identifier: Apache-2.0
"""SAIL MSA indexer (PPU): paged OnlyScore plan + SAIL top-k select.

One plan covers the whole batch (decode and prefill together, exactly like the
upstream ``fmha_sm100`` MSA indexer): ``fmha_sail`` scores every causally
visible KV tile into a per-step ``[heads, max_k_tiles, tokens]`` buffer, then a
single ``sparse_topk_select`` writes the selected tiles into the model-shared
``topk_indices_buffer`` that the sparse attend reads back.

Unlike ``fmha_sm100``, SAIL's ``sparse_topk_select`` takes a *scalar*
``num_valid_pages`` while every query token has its own causal page count. The
init/local blocks are therefore forced per token with a small Triton kernel
before the select, and out-of-range selections are clamped afterwards.

Indexer and attend run between cudagraph segments
(``eager_break_during_capture``), so the plan is rebuilt eagerly in ``build``
and the score buffer is a plain per-step allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerBackend,
    MiniMaxM3IndexerImpl,
    MiniMaxM3IndexerMetadata,
    MiniMaxM3IndexerTritonMetadataBuilder,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    CommonAttentionMetadata,
)

from vllm_sail.models.minimax_m3_msa.adapter import require_sail_msa_ops
from vllm_sail.models.minimax_m3_msa.common import (
    PAGE_SIZE,
    PlanWorkspacePool,
    clamp_topk_to_causal,
    flat_page_table,
    plan_device,
    token_causal_pages,
)
from vllm_sail.models.minimax_m3_msa.gating import (
    MAX_CONTEXT_TILES,
    decide_from_config,
    decide_from_current_config,
)
from vllm_sail.models.minimax_m3_msa.kernels import force_init_local_scores

logger = init_logger(__name__)


class MiniMaxM3IndexerSailMSABackend(MiniMaxM3IndexerBackend):
    """Indexer backend selecting the SAIL (msa_for_sail) builder + impl."""

    @staticmethod
    def get_name() -> str:
        return "MINIMAX_M3_SPARSE_INDEXER_SAIL"

    @staticmethod
    def get_impl_cls() -> type[MiniMaxM3IndexerSailMSAImpl]:
        return MiniMaxM3IndexerSailMSAImpl

    @staticmethod
    def get_builder_cls() -> type[MiniMaxM3IndexerSailMSAMetadataBuilder]:
        return MiniMaxM3IndexerSailMSAMetadataBuilder


@dataclass
class MiniMaxM3IndexerSailMSAMetadata(MiniMaxM3IndexerMetadata):
    """Plan + per-step score/selection inputs for the SAIL indexer."""

    # fmha_sail_plan() tuple for the whole batch; passed unchanged to fmha_sail.
    sail_plan: tuple | None = None
    # Compacted paged page table ([sum_pages] int32) the plan was built for.
    sail_kv_indices: torch.Tensor | None = None
    # [heads, max_k_tiles, num_tokens], fp32 or bf16, filled with -inf padding
    # once per step and overwritten in place by every layer's OnlyScore pass.
    sail_max_score: torch.Tensor | None = None
    sail_max_k_tiles: int = 0
    # Per-token causal page count cdiv(pos + 1, PAGE_SIZE); drives init/local
    # forcing and the post-select clamp.
    sail_num_valid_pages: torch.Tensor | None = None


class MiniMaxM3IndexerSailMSAMetadataBuilder(MiniMaxM3IndexerTritonMetadataBuilder):
    """Builds the SAIL plan eagerly; decode metadata comes from the Triton
    builder (unused by this impl but kept for parity with the base contract)."""

    # The plan, page table and score buffer are rebuilt per step from the
    # batch's lengths (``max_k_tiles`` grows with the context), so a FULL graph
    # would keep replaying capture-time pointers. PIECEWISE capture is safe:
    # ``_run_attention`` is an eager break there and runs these kernels with
    # the fresh metadata.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # The side-cache spec carries the resolved index dtype (bf16 or fp8).
        indexer_fp8 = kv_cache_spec.dtype == torch.float8_e4m3fn
        selection = decide_from_config(
            vllm_config, indexer_kv_dtype="fp8" if indexer_fp8 else "bf16"
        )
        if not selection.enable:
            raise ValueError(
                "MiniMax M3 SAIL MSA indexer selected but its policy is off: "
                f"{selection.reason}"
            )
        self.score_dtype = selection.score_dtype
        self.indexer_fp8 = selection.indexer_fp8
        self._ops = require_sail_msa_ops()
        self._plans = PlanWorkspacePool()

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3IndexerSailMSAMetadata:
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)

        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor

        qsl_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
        qo_lens = (qsl_cpu[1:] - qsl_cpu[:-1]).to(torch.int32)
        kv_lens = seq_lens[:num_reqs].cpu().to(torch.int32)

        device = plan_device(seq_lens)
        workspace = self._plans.get(
            num_tokens=int(qo_lens.sum()),
            max_qo_len=max(int(qo_lens.max()), 1),
            num_qo_heads=self.num_index_heads,
            num_kv_heads=1,
            kv_block_num=0,
            output_maxscore=True,
            use_fp8_kvcache=self.indexer_fp8,
            device=device,
        )
        plan = self._ops.fmha_sail_plan(
            qo_lens,
            kv_lens,
            self.num_index_heads,
            # The index cache has one KV head (MQA proxy); GQA reduction is
            # over the index-query heads.
            num_kv_heads=1,
            page_size=PAGE_SIZE,
            output_maxscore=True,
            causal=True,
            use_fp8_kvcache=self.indexer_fp8,
            # OnlyScore: no output and no split-KV reduction.
            num_kv_splits=1,
            device=device,
            given_workspace=workspace,
        )
        has_mixed_prefill, _, batch_size, first_plan, second_plan = plan
        if batch_size != num_reqs:
            raise RuntimeError(
                "SAIL indexer plan returned an unexpected batch size "
                f"({batch_size}/{num_reqs})"
            )
        # A mixed step yields a per-sub-batch plan each; the library merges the
        # two score buffers back into the one we pass to fmha_sail, so size it
        # for the wider half.
        sub_plans = (first_plan, second_plan) if has_mixed_prefill else (first_plan,)
        max_k_tiles = max(int(p["max_k_tiles"]) for p in sub_plans)
        if not 0 < max_k_tiles <= MAX_CONTEXT_TILES:
            raise RuntimeError(
                f"SAIL indexer plan disabled max_score (max_k_tiles={max_k_tiles})"
            )

        kv_indices = flat_page_table(block_table, seq_lens, num_reqs)
        max_score = torch.full(
            (self.num_index_heads, max_k_tiles, num_tokens),
            float("-inf"),
            dtype=self.score_dtype,
            device=seq_lens.device,
        )
        positions = common_attn_metadata.positions
        assert positions is not None, "SAIL indexer needs per-token positions"
        num_valid_pages = token_causal_pages(positions, num_tokens)

        return MiniMaxM3IndexerSailMSAMetadata(
            seq_lens=base.seq_lens,
            max_seq_len=base.max_seq_len,
            slot_mapping=base.slot_mapping,
            num_actual_tokens=base.num_actual_tokens,
            num_decodes=base.num_decodes,
            num_decode_tokens=base.num_decode_tokens,
            num_prefills=base.num_prefills,
            num_prefill_tokens=base.num_prefill_tokens,
            prefill=base.prefill,
            decode=base.decode,
            sail_plan=plan,
            sail_kv_indices=kv_indices,
            sail_max_score=max_score,
            sail_max_k_tiles=max_k_tiles,
            sail_num_valid_pages=num_valid_pages,
        )


class MiniMaxM3IndexerSailMSAImpl(MiniMaxM3IndexerImpl):
    """SAIL OnlyScore + top-k for the whole batch in two kernel calls."""

    indexer_backend_cls: ClassVar[type[AttentionBackend]] = (
        MiniMaxM3IndexerSailMSABackend
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        selection = decide_from_current_config(indexer_kv_dtype=self.indexer_kv_dtype)
        if not selection.enable:
            raise ValueError(
                "MiniMax M3 SAIL MSA indexer impl built while the policy is "
                f"off: {selection.reason}"
            )
        self.score_dtype = selection.score_dtype
        self._ops = require_sail_msa_ops()
        logger.warning_once(
            "MiniMax M3 indexer: SAIL OnlyScore + top-k "
            "[indexer_kv_dtype=%s, score_dtype=%s]",
            self.indexer_kv_dtype,
            self.score_dtype,
        )

    def forward(
        self,
        index_query: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if kwargs:
            raise ValueError(
                "The SAIL M3 indexer does not implement fused decode "
                f"sparse-table construction (got {sorted(kwargs)})"
            )
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None  # profiling run; caches unbound
        md = attn_metadata[self.index_cache.prefix]
        assert isinstance(md, MiniMaxM3IndexerSailMSAMetadata)

        num_tokens = md.num_actual_tokens
        index_q = index_query[:num_tokens].view(
            -1, self.num_index_heads, self.index_head_dim
        )
        assert md.sail_plan is not None and md.sail_max_score is not None
        assert md.sail_kv_indices is not None and md.sail_num_valid_pages is not None

        buf = self.topk_indices_buffer
        assert buf is not None, "SAIL indexer needs the shared top-k buffer"

        # Index-K cache (pages, 128, D) -> paged MQA (pages, 1, 128, D).
        kv = self.index_cache.kv_cache
        if kv.dtype != self.index_cache.dtype:
            kv = kv.view(self.index_cache.dtype)
        k_pages = kv.view(kv.shape[0], 1, PAGE_SIZE, self.index_head_dim)
        # OnlyScore writes every causally visible tile; the -inf padding filled
        # by the builder stays in place, so no per-layer reset is needed.
        self._ops.fmha_sail(
            index_q,
            k_pages,
            k_pages,  # V placeholder; not read with output_o=False
            md.sail_plan,
            kv_indices=md.sail_kv_indices,
            output_o=False,
            output_maxscore=True,
            max_score=md.sail_max_score,
            sm_scale=self.scale,
            max_score_dtype=md.sail_max_score.dtype,
        )

        # SAIL's select takes a scalar valid-page count, so force each token's
        # init/local tiles here (OnlyScore leaves init blocks unwritten and the
        # local window depends on the per-token causal length).
        force_init_local_scores(
            md.sail_max_score,
            md.sail_num_valid_pages,
            self.init_blocks,
            self.local_blocks,
        )

        out = buf[:num_tokens]
        self._ops.sparse_topk_select(
            md.sail_max_score,
            self.topk_blocks,
            num_valid_pages=md.sail_max_k_tiles,
            output=out,
        )
        # Drop the selections that only won because their tile is -inf padding
        # past this token's causal range, keeping ascending order + -1 tail.
        clamp_topk_to_causal(out, md.sail_num_valid_pages)

        # The attend reads ``buf`` directly; this return is vestigial.
        return None, None
