# SPDX-License-Identifier: Apache-2.0
"""Run the indexer dispatch body without torch, vLLM or a device.

Device tests cover the imported module, cache bytes and real consumers.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from tests.support.source import function


@pytest.mark.parametrize(
    "dtype,use_fp4_cache,available",
    [("bfloat16", True, True), ("bfloat16", True, False), ("float32", False, True)],
)
def test_prefill_topk_loads_plugin_extension_lazily(
    monkeypatch, dtype, use_fp4_cache, available
):
    plugin_topk, upstream_topk = Mock(), Mock()
    torch = ModuleType("torch")
    torch.int8, torch.int32 = "int8", "int32"
    torch.bfloat16, torch.float32 = "bfloat16", "float32"
    torch.float8_e4m3fn = "float8_e4m3fn"
    torch.ops = SimpleNamespace(
        _ppu_C=SimpleNamespace(),
        _C=SimpleNamespace(top_k_per_row_prefill=upstream_topk),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    path = Path(__file__).parents[2] / "vllm_sail/_custom_ops.py"
    spec = importlib.util.spec_from_file_location("_test_indexer_custom_ops", path)
    ppu_ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ppu_ops)
    imports = []

    def load(name):
        imports.append(name)
        assert name == "vllm_sail._C"
        if not available:
            raise ModuleNotFoundError(name)
        torch.ops._ppu_C.top_k_per_row_prefill_bf16 = plugin_topk
        return ModuleType(name)

    monkeypatch.setattr(ppu_ops.importlib, "import_module", load)
    layer = "model.layers.0.self_attn.indexer.k_cache"
    chunk = SimpleNamespace(
        token_start=0,
        token_end=3,
        total_seq_lens=64,
        skip_kv_gather=True,
        cu_seqlen_ks=object(),
        cu_seqlen_ke=object(),
    )
    metadata = SimpleNamespace(
        slot_mapping=SimpleNamespace(shape=(3,)),
        num_decodes=0,
        num_prefills=1,
        num_decode_tokens=0,
        prefill=SimpleNamespace(chunks=[chunk]),
    )
    logits = MagicMock(dtype=dtype, shape=(3, 64))
    logits.stride.side_effect = lambda dim: (64, 1)[dim]
    compute_logits = Mock(return_value=logits)
    namespace = {
        "torch": torch,
        "get_forward_context": lambda: SimpleNamespace(attn_metadata={layer: metadata}),
        "_resolve_layer_name": lambda name: name,
        "DeepseekV32IndexerMetadata": SimpleNamespace,
        "current_workspace_manager": lambda: SimpleNamespace(
            get_simultaneous=lambda *args: (MagicMock(), MagicMock())
        ),
        "_gather_workspace_shapes": lambda *args: (object(), object()),
        "is_deep_gemm_supported": lambda: True,
        "fp8_fp4_mqa_logits": compute_logits,
        "fp8_mqa_logits": compute_logits,
        "ppu_ops": ppu_ops,
    }
    indexer = function(
        "vllm_sail/attention/ops/mla_sparse.py", "ppu_sparse_attn_indexer", namespace
    )
    q = MagicMock(dtype="float8_e4m3fn")
    topk = MagicMock()
    args = (
        SimpleNamespace(shape=(3, 1)),
        layer,
        object(),
        q,
        MagicMock() if use_fp4_cache else None,
        None,
        MagicMock(),
        128,
        "ue8m0",
        2,
        128,
        64,
        64,
        topk,
        True,
        use_fp4_cache,
    )
    if not available:
        with pytest.raises(RuntimeError, match="top_k_per_row_prefill_bf16.*not built"):
            indexer(*args)
        plugin_topk.assert_not_called()
        upstream_topk.assert_not_called()
        return

    assert indexer(*args) is topk
    assert indexer(*args) is topk
    expected = (
        logits,
        chunk.cu_seqlen_ks,
        chunk.cu_seqlen_ke,
        topk.__getitem__.return_value,
        3,
        64,
        1,
        2,
    )
    if use_fp4_cache:
        assert imports == ["vllm_sail._C"]
        plugin_topk.assert_called_with(*expected)
        assert plugin_topk.call_count == 2
        upstream_topk.assert_not_called()
    else:
        assert imports == []
        upstream_topk.assert_called_with(*expected)
        plugin_topk.assert_not_called()


@pytest.mark.parametrize("q_dtype", ["int8", "float8_e4m3fn"])
@pytest.mark.parametrize("skip_insert", [False, True])
def test_writer_dispatch_and_speculative_padding(q_dtype, skip_insert):
    layer = "model.layers.0.self_attn.indexer.k_cache"
    slots = SimpleNamespace(shape=(3,))
    metadata = SimpleNamespace(
        slot_mapping=slots,
        num_decodes=0,
        num_prefills=0,
        num_decode_tokens=0,
    )
    int8_writer, fp8_writer = Mock(), Mock()
    namespace = {
        "torch": SimpleNamespace(int8="int8"),
        "get_forward_context": lambda: SimpleNamespace(attn_metadata={layer: metadata}),
        "_resolve_layer_name": lambda name: name,
        "DeepseekV32IndexerMetadata": SimpleNamespace,
        "indexer_k_quant_and_cache_int8": int8_writer,
        "ops": SimpleNamespace(indexer_k_quant_and_cache=fp8_writer),
    }
    indexer = function(
        "vllm_sail/attention/ops/mla_sparse.py", "ppu_sparse_attn_indexer", namespace
    )
    k, topk = MagicMock(), MagicMock()
    cache, weights = object(), object()
    result = indexer(
        SimpleNamespace(shape=(5, 1)),
        layer,
        cache,
        SimpleNamespace(dtype=q_dtype),
        None,
        k,
        weights,
        128,
        "ue8m0",
        2,
        128,
        32,
        32,
        topk,
        skip_insert,
    )

    assert result is topk
    k.__getitem__.assert_called_once_with(slice(None, 3))
    topk.__setitem__.assert_called_once_with(slice(None, 5), -1)
    if skip_insert:
        int8_writer.assert_not_called()
        fp8_writer.assert_not_called()
    elif q_dtype == "int8":
        int8_writer.assert_called_once_with(
            k.__getitem__.return_value, cache, slots, 128
        )
        fp8_writer.assert_not_called()
    else:
        fp8_writer.assert_called_once_with(
            k.__getitem__.return_value, cache, slots, 128, "ue8m0"
        )
        int8_writer.assert_not_called()
