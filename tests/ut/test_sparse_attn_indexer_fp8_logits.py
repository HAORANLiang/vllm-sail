# SPDX-License-Identifier: Apache-2.0
"""PPU logits-dtype pin dispatch for the sparse attention indexer.

The pin installed by ``vllm_sail.patch.enhancement.attention.
sparse_attn_indexer`` owns the module-level ``fp8_fp4_mqa_logits`` /
``fp8_fp4_paged_mqa_logits`` names on PPU (the int8-indexer patch is
inactive on sm_89). It must split its two callers by Q dtype: packed-FP4
queries (int8 values, int32 scales) go to the MXFP4 kernels with BF16
weights, legacy FP8 queries (float8_e4m3fn, per-token scales folded into
FP32 weights) go to the SAIL FP8 kernels. These tests pin that dispatch
with recording stubs, so no device or DeepGEMM wheel is required.
"""

import pytest

pytestmark = pytest.mark.requires_runtime

torch = pytest.importorskip("torch", reason="requires torch")
plugin = pytest.importorskip(
    "vllm_sail.patch.enhancement.attention.sparse_attn_indexer",
    reason="requires vllm",
)


class _State:
    pass


class _StubPlatform:
    def __init__(self, ppu: bool):
        self._ppu = ppu

    def is_ppu(self):
        return self._ppu


class _Recorder:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


@pytest.fixture
def ppu(monkeypatch):
    """Force the PPU branch and record every kernel the pin can reach."""
    state = _State()
    state.upstream_dense = _Recorder("upstream-dense")
    state.upstream_paged = _Recorder("upstream-paged")
    state.fp8_dense = _Recorder("fp8-dense")
    state.fp8_paged = _Recorder("fp8-paged")
    state.fp4_dense = _Recorder("fp4-dense")
    state.fp4_paged = _Recorder("fp4-paged")

    monkeypatch.setattr(plugin, "current_platform", _StubPlatform(ppu=True))
    monkeypatch.setattr(plugin, "_upstream_fp8_fp4_mqa_logits", state.upstream_dense)
    monkeypatch.setattr(
        plugin, "_upstream_fp8_fp4_paged_mqa_logits", state.upstream_paged
    )
    monkeypatch.setattr(plugin._ppu_deep_gemm, "fp8_mqa_logits", state.fp8_dense)
    monkeypatch.setattr(
        plugin._ppu_deep_gemm, "fp8_paged_mqa_logits", state.fp8_paged
    )
    monkeypatch.setattr(plugin._ppu_deep_gemm, "fp8_fp4_mqa_logits", state.fp4_dense)
    monkeypatch.setattr(
        plugin._ppu_deep_gemm, "fp8_fp4_paged_mqa_logits", state.fp4_paged
    )
    return state


def test_fp8_dense_q_scores_through_the_sail_fp8_kernel(ppu):
    """Legacy FP8 Q keeps FP32 weights and never reaches the MXFP4 kernel."""
    q = torch.zeros((2, 4, 8), dtype=torch.float8_e4m3fn)
    weights = torch.zeros((2, 4), dtype=torch.float32)
    out = plugin.fp8_fp4_mqa_logits_ppu(
        (q, None),
        ("k_fp8", "k_scales"),
        weights,
        "ks",
        "ke",
        32,
        logits_dtype=torch.float32,
    )
    assert out == "fp8-dense"
    args, kwargs = ppu.fp8_dense.calls[-1]
    assert args[0] is q
    assert args[1] == ("k_fp8", "k_scales")
    assert args[2] is weights
    assert kwargs["clean_logits"] is False
    assert kwargs["max_seqlen_k"] == 32
    assert not ppu.fp4_dense.calls
    assert not ppu.upstream_dense.calls


def test_fp8_dense_q_without_a_window_keeps_global_columns(ppu):
    """A missing max_seqlen_k maps to the legacy 0 (global columns) form."""
    q = torch.zeros((2, 4, 8), dtype=torch.float8_e4m3fn)
    plugin.fp8_fp4_mqa_logits_ppu(
        (q, None), ("k", "s"), torch.zeros((2, 4)), "ks", "ke"
    )
    _, kwargs = ppu.fp8_dense.calls[-1]
    assert kwargs["max_seqlen_k"] == 0


def test_fp4_dense_q_keeps_the_mxfp4_kernel_with_bf16_weights(ppu):
    """Packed-FP4 Q (int8) still casts weights and pins BF16 logits."""
    q = torch.zeros((2, 4, 8), dtype=torch.int8)
    weights = torch.zeros((2, 4), dtype=torch.float32)
    out = plugin.fp8_fp4_mqa_logits_ppu(
        (q, "q_sf"), ("k", "s"), weights, "ks", "ke", 32
    )
    assert out == "fp4-dense"
    args, kwargs = ppu.fp4_dense.calls[-1]
    assert args[2].dtype is torch.bfloat16
    assert kwargs["clean_logits"] is False
    assert kwargs["logits_dtype"] is torch.bfloat16
    assert not ppu.fp8_dense.calls


def test_fp8_paged_q_scores_through_the_sail_fp8_paged_kernel(ppu):
    """Legacy FP8 paged queries keep FP32 weights and drop request indices."""
    q = torch.zeros((3, 1, 4, 8), dtype=torch.float8_e4m3fn)
    weights = torch.zeros((3, 4), dtype=torch.float32)
    out = plugin.fp8_fp4_paged_mqa_logits_ppu(
        (q, None),
        "kv_cache",
        weights,
        torch.zeros((3, 1), dtype=torch.int32),
        "block_table",
        "schedule",
        128,
        logits_dtype=torch.float32,
        indices=torch.zeros(3, dtype=torch.int32),
    )
    assert out == "fp8-paged"
    args, kwargs = ppu.fp8_paged.calls[-1]
    assert args[0] is q
    assert args[1] == "kv_cache"
    assert args[2] is weights
    assert args[6] == 128
    assert kwargs["clean_logits"] is False
    assert "indices" not in kwargs
    assert not ppu.fp4_paged.calls
    assert not ppu.upstream_paged.calls


def test_fp8_paged_q_forwards_the_capacity_bound_unchanged(ppu):
    """The wrapper adds no default; the kernel owns the bound validation."""
    q = torch.zeros((1, 1, 4, 8), dtype=torch.float8_e4m3fn)
    plugin.fp8_fp4_paged_mqa_logits_ppu(
        (q, None), "kv_cache", torch.zeros((1, 4)), "lens", "table", "schedule"
    )
    args, _ = ppu.fp8_paged.calls[-1]
    assert args[6] is None


def test_fp4_paged_q_keeps_mxfp4_and_requires_indices(ppu):
    """MXFP4 paged queries keep the BF16 cast and the indices guard."""
    q = torch.zeros((2, 1, 4, 8), dtype=torch.int8)
    with pytest.raises(ValueError, match="indices"):
        plugin.fp8_fp4_paged_mqa_logits_ppu(
            (q, "q_sf"),
            "kv_cache",
            torch.zeros((2, 4), dtype=torch.float32),
            "lens",
            "table",
            "schedule",
            64,
        )
    out = plugin.fp8_fp4_paged_mqa_logits_ppu(
        (q, "q_sf"),
        "kv_cache",
        torch.zeros((2, 4), dtype=torch.float32),
        "lens",
        "table",
        "schedule",
        64,
        indices=torch.zeros(2, dtype=torch.int32),
    )
    assert out == "fp4-paged"
    args, _ = ppu.fp4_paged.calls[-1]
    assert args[2].dtype is torch.bfloat16
    assert not ppu.fp8_paged.calls


def test_non_ppu_platform_delegates_both_entries_upstream(ppu):
    """Off PPU the pin is a pure pass-through to the captured originals."""
    plugin.current_platform._ppu = False
    dense = plugin.fp8_fp4_mqa_logits_ppu(
        ("q", None), ("k", "s"), "w", "ks", "ke", 8, logits_dtype=torch.bfloat16
    )
    paged = plugin.fp8_fp4_paged_mqa_logits_ppu(
        ("q", None), "cache", "w", "lens", "table", "sched", 8, indices="idx"
    )
    assert dense == "upstream-dense"
    assert paged == "upstream-paged"
    assert ppu.upstream_paged.calls[-1][1]["indices"] == "idx"
    assert not ppu.fp8_dense.calls and not ppu.fp8_paged.calls
