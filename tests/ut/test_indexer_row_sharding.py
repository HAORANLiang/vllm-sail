# SPDX-License-Identifier: Apache-2.0
"""Pin the prefill-chunk geometry the selected upstream indexer build relies on.

vLLM 0.34 removed TP prefill row-sharding (``balanced_prefill_row_shard``,
``tp_prefill_row_sharding_supported``, the ``row_shard_sizes`` field and
``self.tp_size``) in favor of ShardQ plus fused_mqa_topk chunk planning. The
plugin never consumed ``row_shard_sizes``, so there is no row-shard behavior to
preserve. This contract instead pins the live replacement: the real chunker's
budget geometry, the build's block_table / slot_mapping identity chain, and the
ShardQ gather gate, all against the selected upstream source.
"""

import ast
from dataclasses import dataclass
from types import SimpleNamespace as NS

import pytest

from tests.support.source import definition, function

INDEXER = "vllm/v1/attention/backends/mla/indexer.py"
FUSED = "vllm/model_executor/kernels/attention/dsa/fused_mqa_topk.py"
FUSED_NAMES = ("FUSED_MQA_TOPK_MAX_KEYS", "FUSED_MQA_TOPK_MAX_ROWS")


class Scalar(int):
    """0-dim tensor double: the chunker reads ``seq_lens[i].item()``."""

    def item(self):
        return int(self)


class Values(list):
    def __getitem__(self, index):
        value = super().__getitem__(index)
        return Values(value) if isinstance(index, slice) else Scalar(value)

    def __floordiv__(self, divisor):
        return Values(value // divisor for value in self)

    def max(self):
        return NS(item=lambda: max(self))


def _load_dataclass(upstream_source_root, name, namespace):
    node = definition(upstream_source_root / INDEXER, name)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), INDEXER, "exec"), namespace)
    return namespace[name]


@pytest.fixture
def chunker(upstream_source_root):
    """The real staticmethod chunker bound to the real constants and helper."""
    namespace = {"envs": NS(VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=1)}
    for node in ast.parse((upstream_source_root / FUSED).read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in FUSED_NAMES
            for target in node.targets
        ):
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), FUSED, "exec"),
                namespace,
            )
    assert set(FUSED_NAMES) <= namespace.keys()
    namespace["dense_logits_rows"] = function(
        upstream_source_root / INDEXER, "dense_logits_rows", namespace
    )
    return function(
        upstream_source_root / INDEXER,
        "DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks",
        namespace,
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "seq_lens,query_lens,workspace,logits_bytes,eligible,offset,expected",
    [
        # Logits budget M*N splits two requests that together exceed it.
        (
            [4, 4],
            [2, 2],
            128,
            64,
            False,
            0,
            [(slice(0, 1), slice(0, 2)), (slice(1, 2), slice(0, 2))],
        ),
        # Fused eligibility bypasses the logits budget while keys fit.
        (
            [4, 4],
            [2, 2],
            128,
            64,
            True,
            0,
            [(slice(0, 2), slice(0, 4))],
        ),
        # A single over-budget request is sub-chunked on the query dimension.
        (
            [8],
            [10],
            128,
            64,
            False,
            0,
            [(slice(0, 1), slice(q, q + 2)) for q in range(0, 10, 2)],
        ),
        # request_offset renumbers request slices without touching query slices.
        (
            [4, 4],
            [2, 2],
            128,
            64,
            False,
            3,
            [(slice(3, 4), slice(0, 2)), (slice(4, 5), slice(0, 2))],
        ),
    ],
)
def test_chunker_budget_geometry(
    chunker, seq_lens, query_lens, workspace, logits_bytes, eligible, offset, expected
):
    chunks = chunker(
        Values(seq_lens),
        Values(query_lens),
        workspace,
        logits_bytes,
        request_offset=offset,
        fused_mqa_topk_eligible=eligible,
    )
    assert chunks == expected


@pytest.fixture
def build_case(upstream_source_root, chunker):
    def make(compress_ratio, *, use_shardq=False, split=None):
        compressed_calls, chunk_calls, shardq_calls = [], [], []
        table, slot_mapping, compressed_slots = object(), object(), object()

        def compressed(*args, **kwargs):
            compressed_calls.append((args, kwargs))
            return compressed_slots

        def chunk_metadata(
            start_idx,
            end_idx,
            query_start_loc,
            query_start_loc_cpu,
            uncompressed_seq_lens,
            compressed_seq_lens,
            compressed_seq_lens_cpu,
            block_table,
            compress_ratio,
            *,
            query_slice=None,
            skip_kv_gather=False,
            dcp_rank=0,
            dcp_world_size=1,
            cp_kv_cache_interleave_size=1,
            pcp_plan=None,
            positions=None,
        ):
            chunk_calls.append((start_idx, end_idx, compress_ratio, query_slice))
            return NS(
                block_table=block_table,
                req_slice=slice(start_idx, end_idx),
                query_slice=query_slice,
                skip_kv_gather=skip_kv_gather,
            )

        def build_shardq_split(common, config):
            shardq_calls.append((common, config))
            return split

        builder = NS(
            compress_ratio=compress_ratio,
            use_pcp=False,
            use_shardq=use_shardq,
            fused_mqa_topk_eligible=False,
            pcp_world_size=1,
            dcp_world_size=1,
            dcp_rank=0,
            cp_kv_cache_interleave_size=1,
            compressed_cp_interleave_size=1,
            kv_cache_spec=NS(num_states=256),
            compressed_slot_mapping_buffer=compressed_slots,
            decode_threshold=1,
            use_flattening=False,
            supports_varlen=False,
            max_prefill_buffer_size=128,
            indexer_uses_fp4=True,
            vllm_config=NS(),
            _prefill_split_seq_lens=lambda value: value,
            _split_indexer_prefill_chunks=chunker,
        )
        common = NS(
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
        namespace = {
            "dataclass": dataclass,
            "get_compressed_slot_mapping": compressed,
            "build_prefill_chunk_metadata": chunk_metadata,
            "build_shardq_split": build_shardq_split,
            "torch": NS(
                diff=lambda value: Values([value[1] - value[0]]),
                bfloat16="bf16",
                float32="f32",
            ),
            "envs": NS(VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=1),
            "split_decodes_and_prefills": lambda *args, **kwargs: (0, 1, 0, 8),
            "DeepseekV32IndexerMetadata": lambda **kwargs: NS(**kwargs),
            "DeepseekV32IndexerPrefillMetadata": _load_dataclass(
                upstream_source_root, "DeepseekV32IndexerPrefillMetadata", {"dataclass": dataclass}
            ),
        }
        build = function(
            upstream_source_root / INDEXER,
            "DeepseekV32IndexerMetadataBuilder.build",
            namespace,
        )
        return NS(
            build=build,
            builder=builder,
            common=common,
            table=table,
            compressed_slots=compressed_slots,
            compressed_calls=compressed_calls,
            chunk_calls=chunk_calls,
            shardq_calls=shardq_calls,
        )

    return make


def check_build(case, compress_ratio):
    result = case.build(case.builder, 0, case.common)
    assert result.block_table is case.table
    assert result.prefill.chunks[0].block_table is case.table
    assert result.prefill.chunks[0].req_slice == slice(0, 1)
    assert result.prefill.chunks[0].query_slice == slice(0, 8)
    assert result.prefill.chunks[0].skip_kv_gather is False
    assert result.prefill.max_prefill_seq_len == 17
    assert result.prefill.shardq_gather_indices is None
    assert case.chunk_calls == [(0, 1, compress_ratio, slice(0, 8))]
    assert len(case.compressed_calls) == int(compress_ratio > 1)
    if case.compressed_calls:
        args, kwargs = case.compressed_calls[0]
        assert args[4] is case.table
        assert (
            result.slot_mapping is kwargs["out"] is case.builder.compressed_slot_mapping_buffer
        )
    else:
        assert result.slot_mapping is case.common.slot_mapping


@pytest.mark.upstream_source
@pytest.mark.parametrize("compress_ratio", [1, 8])
def test_build_preserves_identity_and_chunk_geometry(build_case, compress_ratio):
    check_build(build_case(compress_ratio), compress_ratio)


@pytest.mark.upstream_source
def test_build_shardq_gate_without_split_keeps_dense_chunks(build_case):
    case = build_case(1, use_shardq=True, split=None)
    check_build(case, 1)
    assert len(case.shardq_calls) == 1
    assert case.shardq_calls[0][0] is case.common


@pytest.mark.upstream_source
def test_prefill_metadata_field_contract(upstream_source_root):
    cls = _load_dataclass(
        upstream_source_root,
        "DeepseekV32IndexerPrefillMetadata",
        {"dataclass": dataclass},
    )
    empty = cls([])
    assert empty.chunks == []
    assert empty.max_prefill_seq_len == -1
    assert empty.shardq_gather_indices is None