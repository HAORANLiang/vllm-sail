# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: F821
"""Keep channelwise QKV and unquantized QSA indexer projections separate.

Constructor copied from vLLM 4ba787ec9959e566d3b587a796dff07673324da5.
"""

from __future__ import annotations

# Import model first: qsa and model have an upstream circular dependency.
from vllm.models.qwen4_exp.nvidia import model as _model  # noqa: F401
from vllm.models.qwen4_exp.nvidia import qsa as _qsa

from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch


def _qsa_body(fn):
    return bind_body(fn, _qsa)


@patch(
    "vllm.models.qwen4_exp.nvidia.qsa",
    "Qwen4ExpQSAAttention.__init__",
    reason=(
        "Quantized QKV (PPU channelwise FP8 or compressed-tensors) cannot share "
        "a projection with a BF16 QSA indexer listed in the checkpoint ignore."
    ),
    affected_versions=">=0.33.0,<0.34.0",
    remove_when="QSA fusion accounts for distinct QKV and indexer quantization schemes.",
)
@_qsa_body
def qsa_init(
    self,
    *,
    vllm_config: VllmConfig,
    config: Qwen4ExpTextConfig,
    layer_id: int,
    quant_config: QuantizationConfig | None = None,
    reduce_results: bool = True,
    prefix: str = "",
) -> None:
    nn.Module.__init__(self)
    # PPU MODIFICATION: begin -- checkpoint QKV is channelwise FP8 (or
    # compressed-tensors quantized) while index_qk_proj remains BF16. Keep
    # their schemes and parameters separate.
    separate_indexer = current_platform.is_ppu() and bool(
        getattr(quant_config, "fp8_channelwise_layers", None)
    )
    ignore = getattr(quant_config, "ignore", None)
    if current_platform.is_ppu() and not separate_indexer and ignore:
        from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
            should_ignore_layer,
        )

        # The fused qkv_proj packs q/k/v plus the QSA indexer shard; the
        # checkpoint ignore must cover all four or none of them, otherwise
        # get_scheme_dict rejects the mixed shards. Compare per-shard ignore
        # status (no fused_mapping: these are already shard names).
        qkv_ignored = [
            should_ignore_layer(f"{prefix}.{name}", ignore=ignore)
            for name in ("q_proj", "k_proj", "v_proj")
        ]
        if any(qkv_ignored) != all(qkv_ignored):
            raise ValueError(
                f"QSA checkpoint at {prefix} ignores only some of q/k/v; "
                "mixed-precision QKV shards cannot share a fused projection."
            )
        indexer_ignored = should_ignore_layer(
            f"{prefix}.indexer.index_qk_proj", ignore=ignore
        )
        separate_indexer = indexer_ignored != all(qkv_ignored)
    if separate_indexer:
        import copy

        quant_config = copy.copy(quant_config)
        quant_config.packed_modules_mapping = {
            **quant_config.packed_modules_mapping,
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        }
    # PPU MODIFICATION: end
    cache_config = vllm_config.cache_config
    model_config = vllm_config.model_config
    if cache_config is None:
        raise ValueError("Qwen4Exp QSA requires a paged KV cache")
    if model_config.dtype != torch.bfloat16:
        raise NotImplementedError("Qwen4Exp QSA currently requires BF16")
    if cache_config.cache_dtype not in (
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ):
        raise NotImplementedError(
            "Qwen4Exp QSA requires a BF16 or FP8-e4m3 main KV cache"
        )
    if getattr(quant_config, "kv_cache_scheme", None) is not None:
        raise NotImplementedError("Qwen4Exp QSA does not support KV quantization")
    parallel_config = vllm_config.parallel_config
    if (
        parallel_config.prefill_context_parallel_size > 1
        or parallel_config.decode_context_parallel_size > 1
    ):
        raise NotImplementedError(
            "Qwen4Exp QSA does not support context parallelism"
        )
    if not getattr(config, "is_causal", True):
        raise NotImplementedError("Qwen4Exp QSA requires causal decoder attention")

    self.config = config
    self.hidden_size = int(config.hidden_size)
    tp_size = get_tensor_model_parallel_world_size()
    self.total_num_heads = int(config.num_attention_heads)
    if self.total_num_heads % tp_size:
        raise ValueError("QSA attention heads must be divisible by TP size")
    self.num_heads = self.total_num_heads // tp_size
    # Decode/verify batches have at most 1 + num_spec query tokens per
    # request; use_prefill_config (max_query_len > this) steers the
    # config table. Shorter batches take the decode profile — harmless,
    # the difference is tile-shape tuning, not correctness.
    self._max_decode_query_len = 1 + vllm_config.num_speculative_tokens
    self.total_num_kv_heads = int(config.num_key_value_heads)
    if self.total_num_kv_heads >= tp_size:
        if self.total_num_kv_heads % tp_size:
            raise ValueError("QSA KV heads must be divisible by TP size")
    elif tp_size % self.total_num_kv_heads:
        raise ValueError("TP size must be divisible by replicated QSA KV heads")
    self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
    self.head_dim = int(config.head_dim or self.hidden_size // self.total_num_heads)
    self.q_size = self.num_heads * self.head_dim
    self.kv_size = self.num_kv_heads * self.head_dim
    self.scaling = self.head_dim**-0.5
    self.dual_chunk_attention_config = getattr(
        config, "dual_chunk_attention_config", None
    )
    if self.dual_chunk_attention_config is not None:
        raise NotImplementedError("Qwen4Exp QSA does not support dual-chunk RoPE")
    # Qwen4Exp full-attention checkpoints always pack a sigmoid output
    # gate next to Q, even when an inherited config default says otherwise.
    self.attn_output_gate = True

    self.qkv_proj = QKVParallelLinear(
        self.hidden_size,
        self.head_dim,
        self.total_num_heads * (1 + self.attn_output_gate),
        self.total_num_kv_heads,
        bias=False,
        quant_config=model.without_modelopt_fp4(quant_config),
        prefix=f"{prefix}.qkv_proj",
    )
    self.o_proj = RowParallelLinear(
        self.total_num_heads * self.head_dim,
        self.hidden_size,
        bias=False,
        reduce_results=reduce_results,
        quant_config=quant_config,
        prefix=f"{prefix}.o_proj",
    )
    self.rotary_emb = get_rope(
        head_size=self.head_dim,
        max_position=config.max_position_embeddings,
        rope_parameters=config.rope_parameters,
        mrope_positions_factor=mrope_positions_factor(vllm_config),
    )
    self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
    self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    mm_config = model_config.multimodal_config
    text_only = mm_config is None or mm_config.language_model_only
    mrope_section = getattr(self.rotary_emb, "mrope_section", None)
    supports_mrope = bool(
        type(self.rotary_emb) is MRotaryEmbedding
        and mrope_section
        and len(mrope_section) == 3
        and sum(mrope_section) == self.rotary_emb.rotary_dim // 2
        and getattr(self.rotary_emb, "mrope_interleaved", False)
    )
    supports_dtype = getattr(self.rotary_emb, "dtype", None) in (
        torch.float16,
        torch.bfloat16,
    )
    self.use_fused_qk_norm_rope_gate = (
        self.attn_output_gate
        and getattr(self.rotary_emb, "is_neox_style", False)
        and current_platform.is_cuda()
        and supports_dtype
        and (text_only or supports_mrope)
    )

    self.layer_name = f"{prefix}.attn"
    self.attn_type = AttentionType.DECODER
    self.kv_cache_dtype = qsa_kv_cache_dtype(cache_config, prefix)
    self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
        self.kv_cache_dtype, model_config
    )
    if self.kv_cache_torch_dtype not in (torch.bfloat16, torch.uint8):
        raise NotImplementedError(
            "Qwen4Exp QSA requires BF16 or FP8-e4m3 (uint8) cache storage"
        )
    self.kv_sharing_target_layer_name = None
    self.kv_cache = torch.tensor([])
    set_default_quant_scales(self, register_buffer=True)

    self.attn_backend = Qwen4ExpQSAFlashAttentionBackend
    self.impl = Qwen4ExpQSAFlashAttentionImpl(
        self.num_heads,
        self.head_dim,
        self.scaling,
        self.num_kv_heads,
        None,
        None,
        self.kv_cache_dtype,
        None,
        AttentionType.DECODER,
        None,
    )
    self.indexer = QSAIndexer(
        vllm_config=vllm_config,
        config=config,
        layer_id=layer_id,
        rotary_emb=self.rotary_emb,
        quant_config=quant_config,
        prefix=f"{prefix}.indexer",
    )
    # One launch does the indexer prepare, the main QK-norm/RoPE/gate and
    # the main K/V cache write (see QSAIndexer.forward); otherwise all of
    # them take the separate kernels. The fused writer receives an fp8
    # pointer, which Triton cannot lower below sm_89.
    self.use_fused_qsa_prepare = (
        self.use_fused_qk_norm_rope_gate
        and self.indexer.use_fused_pre_indexer
        and (
            self.kv_cache_torch_dtype != torch.uint8
            or current_platform.has_device_capability(89)
        )
    )
    self.fuse_indexer_projection = vllm_config.lora_config is None
    # PPU MODIFICATION: begin
    self.fuse_indexer_projection &= not separate_indexer
    # PPU MODIFICATION: end
    if self.fuse_indexer_projection:
        self.index_qk_size = self.indexer.index_qk_proj.output_size
        self.qkv_proj = Qwen4ExpQSAQKVIndexerLinear(
            self.qkv_proj,
            self.index_qk_size,
            model.without_modelopt_fp4(quant_config),
        )
        del self.indexer.index_qk_proj

    max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    # PACKED selection buffer: the trailing column holds each row's
    # valid-entry count (written by the expand kernel) — never a token
    # index; the sparse attention kernel reads it as its loop bound.
    # MTP skip_topk steps reuse rows frozen from step 0; the count is
    # a row column, so compaction/reuse keep it paired with the content.
    self.register_buffer(
        "topk_indices_buffer",
        torch.empty(
            max_tokens,
            self.indexer.packed_output_width,
            dtype=torch.int32,
        ),
        persistent=False,
    )

    static_context = vllm_config.compilation_config.static_forward_context
    if self.layer_name in static_context:
        raise ValueError(f"Duplicate layer name: {self.layer_name}")
    static_context[self.layer_name] = self
