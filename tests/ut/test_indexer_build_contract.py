# SPDX-License-Identifier: Apache-2.0
"""Keep the indexer build's prefill-chunk delegation the plugin patches against.

Upstream 0.33 replaced TP row sharding of the sparse indexer with ShardQ, so
``row_shard_sizes`` and its helpers are gone. What the plugin still depends on
is the shape of upstream's ``build``: the compressed slot mapping is written
into the builder buffer from the shared block table, prefill chunking goes
through ``self._split_indexer_prefill_chunks(...)`` (which the plugin replaces
with a logits-dtype-aware body and which therefore must keep receiving
``request_offset`` and ``fused_mqa_topk_eligible``), and each chunk is built
with the positions upstream passes for ShardQ.
"""

from types import SimpleNamespace as NS

import pytest


class Values(list):
    def __getitem__(self, index):
        value = super().__getitem__(index)
        return Values(value) if isinstance(index, slice) else value

    def __floordiv__(self, divisor):
        return Values(value // divisor for value in self)

    def max(self):
        return NS(item=lambda: max(self))


@pytest.fixture
def build_case():
    def make(compress_ratio):
        compressed_calls, split_calls, chunk_calls = [], [], []
        table, slot_mapping, compressed_slots = object(), object(), object()
        config = NS(attention_config=NS(indexer_tp_shard_min_rows=37))
        common = NS(
            positions=None,
            num_reqs=1,
            num_actual_tokens=8,
            query_start_loc=object(),
            query_start_loc_cpu=Values([0, 8]),
            seq_lens=Values([17]),
            seq_lens_cpu_upper_bound=Values([17]),
            slot_mapping=slot_mapping,
            block_table_tensor=table,
            dcp_local_seq_lens=None,
            max_seq_len=17,
        )
        builder = NS(
            compress_ratio=compress_ratio,
            use_pcp=False,
            pcp_world_size=1,
            dcp_world_size=1,
            dcp_rank=0,
            cp_kv_cache_interleave_size=1,
            compressed_cp_interleave_size=1,
            kv_cache_spec=NS(num_states=256),
            compressed_slot_mapping_buffer=compressed_slots,
            decode_threshold=1,
            use_flattening=False,
            use_shardq=False,
            supports_varlen=False,
            max_prefill_buffer_size=128,
            indexer_uses_fp4=True,
            fused_mqa_topk_eligible=False,
            vllm_config=config,
            tp_size=8,
            _prefill_split_seq_lens=lambda value: value,
            _split_indexer_prefill_chunks=lambda *args, **kwargs: split_calls.append(
                (args, kwargs)
            )
            or [(slice(0, 1), slice(0, 8))],
        )

        def compressed(*args, **kwargs):
            compressed_calls.append((args, kwargs))
            return compressed_slots

        def chunk(*args, **kwargs):
            chunk_calls.append((args, kwargs))
            return NS(block_table=args[7])

        namespace = {
            "get_compressed_slot_mapping": compressed,
            "torch": NS(
                diff=lambda value: Values([value[1] - value[0]]),
                bfloat16="bf16",
                float32="f32",
            ),
            "envs": NS(VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=1),
            "split_decodes_and_prefills": lambda *args, **kwargs: (0, 1, 0, 8),
            "build_prefill_chunk_metadata": chunk,
            "DeepseekV32IndexerMetadata": lambda **kwargs: NS(**kwargs),
            "DeepseekV32IndexerPrefillMetadata": lambda chunks, **kwargs: NS(
                chunks=chunks, **kwargs
            ),
        }
        return builder, common, namespace, compressed_calls, split_calls, chunk_calls

    return make


def check_result(load, build_case, compress_ratio):
    (
        builder,
        common,
        namespace,
        compressed_calls,
        split_calls,
        chunk_calls,
    ) = build_case(compress_ratio)
    assert not hasattr(builder, "kernel_block_size")
    result = load(namespace)(builder, 0, common)
    assert result.block_table is common.block_table_tensor
    assert result.prefill.chunks[0].block_table is common.block_table_tensor
    assert result.prefill.shardq_gather_indices is None
    assert result.prefill.max_prefill_seq_len == 17

    # The plugin's splitter body takes these two keywords; upstream must keep
    # passing them or the patched chunker silently receives its own defaults.
    ((args, kwargs),) = split_calls
    assert args[2:] == (128, 1024 * 1024)
    assert kwargs["request_offset"] == 0
    assert kwargs["fused_mqa_topk_eligible"] is False

    ((chunk_args, chunk_kwargs),) = chunk_calls
    assert chunk_args[0:2] == (0, 1)
    assert chunk_args[7] is common.block_table_tensor
    assert chunk_kwargs["query_slice"] == slice(0, 8)
    assert chunk_kwargs["skip_kv_gather"] is False
    assert chunk_kwargs["positions"] is None
    assert chunk_kwargs["dcp_rank"] == 0
    assert chunk_kwargs["dcp_world_size"] == 1
    assert chunk_kwargs["cp_kv_cache_interleave_size"] == 1

    assert len(compressed_calls) == int(compress_ratio > 1)
    if compressed_calls:
        args, kwargs = compressed_calls[0]
        assert args[4] is common.block_table_tensor
        assert (
            result.slot_mapping
            is kwargs["out"]
            is builder.compressed_slot_mapping_buffer
        )
    else:
        assert result.slot_mapping is common.slot_mapping


@pytest.mark.upstream_source
@pytest.mark.parametrize("compress_ratio", [1, 8])
def test_actual_upstream_producer_keeps_the_delegated_contract(
    upstream_source_root, build_case, compress_ratio
):
    import ast

    from tests.support.source import definition

    path = upstream_source_root / "vllm/v1/attention/backends/mla/indexer.py"
    node = definition(path, "DeepseekV32IndexerMetadataBuilder.build")
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )

    def load(namespace):
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        return namespace["build"]

    check_result(load, build_case, compress_ratio)
