# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the 810E Kimi K3.1 DSA indexer patch.

Covers the pieces that a mis-specified quantizer or scoring wrapper would get
wrong silently: the shared platform gate, the software e4m3 encoder and the
int8 quantizer against torch's reference casts, and both ``fp8_fp4_*``
replacements (delegation, MXFP4 rejection, row expansion, refused varlen
indices).

Importing the ``models`` package installs every model patch, which is what
``tests/ut/test_model_patches.py`` expects to find in the registry; the module
globals this file patches (``_original_fp8_*``, ``_sail_deep_gemm``) are read at
call time, so ``monkeypatch`` covers them.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

pytestmark = pytest.mark.requires_runtime

torch = pytest.importorskip("torch", reason="requires torch")
triton = pytest.importorskip("triton", reason="requires triton")
tl = pytest.importorskip("triton.language", reason="requires triton")

patch_module = pytest.importorskip(
    "vllm_sail.patch.enhancement.models.kimi_k3_dsa_indexer",
    reason="requires vLLM",
)

# Globals of *this* module: a ``triton.jit`` body resolves callees here.
_encode_e4m3fn = patch_module._encode_e4m3fn
_quant_to_int8 = patch_module._quant_to_int8

_NEEDS_GPU = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU to JIT the probe kernels"
)


@triton.jit
def _encode_probe(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, _encode_e4m3fn(tl.load(x_ptr + offs)))


@triton.jit
def _int8_probe(y_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, _quant_to_int8(tl.load(y_ptr + offs)))


# ---------------------------------------------------------------------------
# Gate: one verdict for both patches
# ---------------------------------------------------------------------------


def test_gate_drives_every_switch() -> None:
    from vllm_sail.patch.enhancement import dsa_indexer_gate as gate

    assert patch_module._SW_FP8_CAST is gate.uses_software_e4m3()
    assert patch_module._USE_INT8_INDEXER is gate.int8_indexer_active()
    assert patch_module._FP8_MAX == (224.0 if patch_module._USE_FNUZ else 448.0)
    assert patch_module._SW_ENCODER_ELIGIBLE is not (
        patch_module._SW_FP8_CAST and patch_module._USE_FNUZ
    )


def test_attention_pin_follows_the_shared_gate() -> None:
    """The logits-dtype pin must not re-derive the platform predicate."""
    from vllm_sail.patch.enhancement.attention import (
        sparse_attn_indexer as attention_patch,
    )

    source = inspect.getsource(attention_patch._install_logits_dtype_pin)
    assert "int8_indexer_active()" in source
    assert "supports_fp8()" not in source


# ---------------------------------------------------------------------------
# Quantizers
# ---------------------------------------------------------------------------


@_NEEDS_GPU
def test_software_e4m3_encoder_matches_the_hardware_cast() -> None:
    values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        448.0,
        -448.0,
        448.5,
        -448.5,
        2.0**-9,  # smallest normal
        2.0**-10,  # smallest subnormal
        2.0**-12,  # rounds to zero
        1e-9,
        1.0625,  # exact tie at three mantissa bits
        1.1875,
        1.375,
        0.9,
        3.1415927,
        -2.7182817,
        240.0,
        255.0,
    ]
    x = torch.tensor(values, dtype=torch.float32, device="cuda")
    expected = x.to(torch.float8_e4m3fn).view(torch.uint8)
    encoded = torch.empty_like(expected)
    _encode_probe[(1,)](x, encoded, BLOCK=triton.next_power_of_2(x.numel()))
    assert encoded.tolist() == expected.tolist()


@_NEEDS_GPU
def test_int8_quantizer_rounds_to_nearest_even() -> None:
    values = [
        -127.5,
        -127.4,
        -126.5,
        -2.5,
        -1.5,
        -0.5,
        0.0,
        0.5,
        1.5,
        2.5,
        126.6,
        127.4,
        200.0,
        -200.0,
    ]
    y = torch.tensor(values, dtype=torch.float32, device="cuda")
    expected = torch.round(y.clamp(-127.0, 127.0)).to(torch.int8)
    quantized = torch.empty_like(expected)
    _int8_probe[(1,)](y, quantized, BLOCK=triton.next_power_of_2(y.numel()))
    assert quantized.tolist() == expected.tolist()


def test_encoder_is_refused_below_capability_with_fnuz(monkeypatch) -> None:
    monkeypatch.setattr(patch_module, "_SW_ENCODER_ELIGIBLE", False)
    with pytest.raises(NotImplementedError):
        patch_module._require_supported_encoder()
    monkeypatch.setattr(patch_module, "_SW_ENCODER_ELIGIBLE", True)
    assert patch_module._require_supported_encoder() is None


# ---------------------------------------------------------------------------
# Dense scoring helper
# ---------------------------------------------------------------------------


def test_fp8_q_delegates_to_the_captured_original(monkeypatch) -> None:
    calls = {}

    def original(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return "fp8-logits"

    monkeypatch.setattr(patch_module, "_original_fp8_mqa", original)
    q = torch.zeros((2, 4, 8), dtype=torch.float8_e4m3fn)
    out = patch_module.fp8_fp4_mqa_logits(
        (q, None), ("k", "s"), "w", "ks", "ke", 16, logits_dtype=torch.bfloat16
    )
    assert out == "fp8-logits"
    assert calls["args"][5] == 16
    assert calls["kwargs"]["logits_dtype"] is torch.bfloat16


def test_int8_dense_scores_through_the_sail_kernel(monkeypatch) -> None:
    calls = {}

    def fake(q, kv, weights, ks, ke, clean_logits, *, max_seqlen_k=0):
        calls.update(q=q, kv=kv, weights=weights, max_seqlen_k=max_seqlen_k)
        calls["clean_logits"] = clean_logits
        return "int8-logits"

    monkeypatch.setattr(patch_module._sail_deep_gemm, "int8_mqa_logits", fake)
    q = torch.zeros((2, 4, 8), dtype=torch.int8)
    k = torch.zeros((5, 8), dtype=torch.uint8)
    out = patch_module.fp8_fp4_mqa_logits(
        (q, None), (k, torch.zeros(5)), "w", "ks", "ke", 32
    )
    assert out == "int8-logits"
    assert calls["kv"][0].dtype is torch.int8
    assert calls["max_seqlen_k"] == 32
    assert calls["clean_logits"] is False


@pytest.mark.parametrize("max_seqlen_k", [None])
def test_int8_dense_refuses_an_unset_window(max_seqlen_k) -> None:
    q = torch.zeros((2, 4, 8), dtype=torch.int8)
    k = torch.zeros((5, 8), dtype=torch.uint8)
    with pytest.raises(ValueError, match="max_seqlen_k"):
        patch_module.fp8_fp4_mqa_logits(
            (q, None), (k, torch.zeros(5)), "w", "ks", "ke", max_seqlen_k
        )


def test_mxfp4_q_is_rejected() -> None:
    q = torch.zeros((2, 4, 8), dtype=torch.int8)
    # Packed uint8 Q viewed as int8 plus int32 block scales is an MXFP4 cache.
    q_scale = torch.zeros((2, 4), dtype=torch.int32)
    k = torch.zeros((5, 8), dtype=torch.int8)
    with pytest.raises(NotImplementedError, match="MXFP4"):
        patch_module.fp8_fp4_mqa_logits(
            (q, q_scale), (k, torch.zeros(5)), torch.zeros((2, 4)), None, None, 8
        )
    with pytest.raises(NotImplementedError, match="MXFP4"):
        patch_module.fp8_fp4_paged_mqa_logits(
            (q, q_scale),
            None,
            torch.zeros((2, 4)),
            torch.zeros((2, 1), dtype=torch.int32),
            torch.zeros((2, 1), dtype=torch.int32),
            None,
            max_model_len=64,
        )


# ---------------------------------------------------------------------------
# Paged scoring helper
# ---------------------------------------------------------------------------


def _capture_paged(monkeypatch):
    calls = {}

    def fake(
        flat_q,
        kv_cache,
        weights,
        contexts,
        tables,
        schedule_metadata,
        max_model_len,
        clean_logits,
    ):
        calls.update(
            q=flat_q,
            contexts=contexts,
            tables=tables,
            max_model_len=max_model_len,
            clean_logits=clean_logits,
        )
        return "int8-paged"

    monkeypatch.setattr(patch_module._sail_deep_gemm, "int8_paged_mqa_logits", fake)
    return calls


def test_paged_rows_expand_per_query(monkeypatch) -> None:
    calls = _capture_paged(monkeypatch)
    q = torch.zeros((2, 3, 4, 8), dtype=torch.int8)
    seq_lens = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    block_table = torch.arange(12, dtype=torch.int32).reshape(2, 6)
    out = patch_module.fp8_fp4_paged_mqa_logits(
        (q, None),
        "cache",
        torch.zeros((6, 4)),
        seq_lens,
        block_table,
        "schedule",
        max_model_len=128,
    )
    assert out == "int8-paged"
    assert calls["q"].shape == (6, 1, 4, 8)
    assert calls["contexts"].shape == (6, 1)
    assert calls["contexts"].is_contiguous()
    assert calls["contexts"].flatten().tolist() == [0, 1, 2, 3, 4, 5]
    assert calls["tables"].shape[0] == 6
    assert torch.equal(calls["tables"][0], block_table[0])
    assert torch.equal(calls["tables"][2], block_table[0])
    assert torch.equal(calls["tables"][3], block_table[1])
    assert calls["max_model_len"] == 128
    assert calls["clean_logits"] is False


def test_paged_single_query_rows_stay_request_major(monkeypatch) -> None:
    calls = _capture_paged(monkeypatch)
    q = torch.zeros((2, 4, 8), dtype=torch.int8)
    seq_lens = torch.tensor([7, 9], dtype=torch.int32)
    block_table = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    patch_module.fp8_fp4_paged_mqa_logits(
        (q, None),
        None,
        torch.zeros((2, 4)),
        seq_lens,
        block_table,
        None,
        max_model_len=64,
    )
    assert calls["q"].shape == (2, 1, 4, 8)
    assert calls["contexts"].flatten().tolist() == [7, 9]
    assert torch.equal(calls["tables"], block_table)


def test_paged_refuses_a_varlen_decode_schedule() -> None:
    q = torch.zeros((2, 3, 4, 8), dtype=torch.int8)
    with pytest.raises(NotImplementedError, match="request index"):
        patch_module.fp8_fp4_paged_mqa_logits(
            (q, None),
            None,
            torch.zeros((6, 4)),
            torch.zeros((2, 3), dtype=torch.int32),
            torch.zeros((2, 4), dtype=torch.int32),
            None,
            max_model_len=64,
            indices=torch.zeros(6, dtype=torch.int32),
        )


def test_paged_refuses_mismatched_rows() -> None:
    q = torch.zeros((2, 3, 4, 8), dtype=torch.int8)
    with pytest.raises(ValueError, match="context_lens"):
        patch_module.fp8_fp4_paged_mqa_logits(
            (q, None),
            None,
            torch.zeros((6, 4)),
            torch.zeros(4, dtype=torch.int32),
            torch.zeros((2, 4), dtype=torch.int32),
            None,
            max_model_len=64,
        )
    with pytest.raises(ValueError, match="block_tables"):
        patch_module.fp8_fp4_paged_mqa_logits(
            (q, None),
            None,
            torch.zeros((6, 4)),
            torch.zeros((2, 3), dtype=torch.int32),
            torch.zeros((5, 4), dtype=torch.int32),
            None,
            max_model_len=64,
        )


def test_paged_fp8_q_delegates_to_the_captured_original(monkeypatch) -> None:
    calls = {}

    def original(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return "fp8-paged"

    monkeypatch.setattr(patch_module, "_original_fp8_paged", original)
    q = torch.zeros((2, 2, 4, 8), dtype=torch.float8_e4m3fn)
    out = patch_module.fp8_fp4_paged_mqa_logits(
        (q, None),
        "cache",
        "w",
        "lens",
        "table",
        "schedule",
        128,
        logits_dtype=torch.bfloat16,
        indices="idx",
    )
    assert out == "fp8-paged"
    assert calls["kwargs"]["indices"] == "idx"
    assert calls["kwargs"]["logits_dtype"] is torch.bfloat16


# ---------------------------------------------------------------------------
# Installation guards
# ---------------------------------------------------------------------------


def test_install_skips_an_already_patched_scoring_name(monkeypatch) -> None:
    from vllm_sail.patch.utils import PATCH_MARKER, PATCH_REGISTRY

    def marked(*args, **kwargs):
        return None

    setattr(marked, PATCH_MARKER, {"test": "already-owned"})
    monkeypatch.setattr(patch_module._sparse, "fp8_fp4_mqa_logits", marked)
    before = len(PATCH_REGISTRY)
    patch_module._install_int8_scoring()
    assert len(PATCH_REGISTRY) == before


def test_int8_scoring_targets_are_registered_when_active() -> None:
    """The conditional int8 targets must still reach ``PATCH_REGISTRY``."""
    from vllm_sail.patch.enhancement.dsa_indexer_gate import int8_indexer_active
    from vllm_sail.patch.utils import PATCH_REGISTRY

    if not int8_indexer_active():
        pytest.skip("the int8 DSA indexer path is inactive on this target")
    targets = {record.target for record in PATCH_REGISTRY}
    assert {
        "vllm.model_executor.layers.sparse_attn_indexer.fp8_fp4_mqa_logits",
        "vllm.model_executor.layers.sparse_attn_indexer.fp8_fp4_paged_mqa_logits",
    } <= targets


def test_consumer_rebinding_is_idempotent_and_complete() -> None:
    patch_module._rebind_kimi_consumer()
    patch_module._rebind_kimi_consumer()
    consumer = importlib.import_module(patch_module._DSAMODEL_MODULE)
    for name in (
        "indexer_k_norm_quant_insert",
        "indexer_q_quant_fold_weights",
        "dsa_fused_prologue",
    ):
        assert getattr(consumer, name) is getattr(patch_module._dsa_indexer, name)
