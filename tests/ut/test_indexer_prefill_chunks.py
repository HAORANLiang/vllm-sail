# SPDX-License-Identifier: Apache-2.0
"""Plan indexer prefill chunks for fused MQA+top-k and dtype-sized logits."""

from types import SimpleNamespace as NS

import pytest

from tests.support.source import function

PATCH = "vllm_sail/patch/enhancement/attention/mla_indexer.py"
UPSTREAM = "vllm/v1/attention/backends/mla/indexer.py"


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


def namespace(builder=None):
    sizes = {"fp32": 4, "bf16": 2}

    def empty(*args, **kwargs):
        return NS(element_size=lambda: sizes[kwargs["dtype"]])

    return {
        "torch": NS(empty=empty, float32="fp32", bfloat16="bf16"),
        "FUSED_MQA_TOPK_MAX_KEYS": 8,
        "FUSED_MQA_TOPK_MAX_ROWS": 4,
        "dense_logits_rows": lambda seq_len, max_bytes: max(
            1, max_bytes // (4 * max(1, seq_len))
        ),
        "_current_builder": [builder] if builder is not None else [],
    }


def split(
    path, name, seq_lens, query_lens, workspace, logits_bytes, builder=None, **kwargs
):
    fn = function(path, name, namespace(builder))
    return fn(
        [Scalar(value) for value in seq_lens],
        [Scalar(value) for value in query_lens],
        workspace,
        logits_bytes,
        **kwargs,
    )


@pytest.mark.parametrize(
    "fp4,expected",
    [
        (True, [(slice(0, 1), slice(0, 10))]),
        (False, [(slice(0, 1), slice(0, 5)), (slice(0, 1), slice(5, 10))]),
    ],
    ids=["bf16", "fp32"],
)
def test_dense_logits_budget_follows_the_logits_dtype(fp4, expected):
    """A bf16 logits cache (fp4 indexer) plans against half the fp32 bytes."""
    assert (
        split(
            PATCH,
            "_split_indexer_prefill_chunks_body",
            [100],
            [10],
            1000,
            2000,
            builder=NS(indexer_uses_fp4=fp4),
        )
        == expected
    )


def test_fused_plan_packs_within_the_row_and_key_caps():
    assert split(
        PATCH,
        "_split_indexer_prefill_chunks_body",
        [3, 3, 3],
        [2, 2, 2],
        100,
        8,
        fused_mqa_topk_eligible=True,
    ) == [(slice(0, 2), slice(0, 4)), (slice(2, 3), slice(0, 2))]


@pytest.mark.parametrize("fp4,rows", [(False, 5), (True, 10)], ids=["fp32", "bf16"])
def test_fused_plan_row_splits_a_request_past_the_key_cap(fp4, rows):
    """Past FUSED_MQA_TOPK_MAX_KEYS the dense budget applies, in logits bytes."""
    chunks = split(
        PATCH,
        "_split_indexer_prefill_chunks_body",
        [9],
        [10],
        100,
        200,
        fused_mqa_topk_eligible=True,
        builder=NS(indexer_uses_fp4=fp4),
    )
    expected = [(slice(0, 1), slice(0, rows))]
    if rows < 10:
        expected.append((slice(0, 1), slice(rows, 10)))
    assert chunks == expected


@pytest.mark.upstream_source
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize(
    "seq_lens,query_lens,workspace,logits_bytes",
    [([3, 3, 3], [2, 2, 2], 100, 8), ([100], [10], 1000, 2000)],
)
def test_copied_planner_matches_upstream_on_the_fp32_path(
    upstream_source_root, fused, seq_lens, query_lens, workspace, logits_bytes
):
    kwargs = {"fused_mqa_topk_eligible": fused}
    local = split(
        PATCH,
        "_split_indexer_prefill_chunks_body",
        seq_lens,
        query_lens,
        workspace,
        logits_bytes,
        **kwargs,
    )
    upstream = split(
        upstream_source_root / UPSTREAM,
        "DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks",
        seq_lens,
        query_lens,
        workspace,
        logits_bytes,
        **kwargs,
    )
    assert local == upstream
