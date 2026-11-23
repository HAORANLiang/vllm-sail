# SPDX-License-Identifier: Apache-2.0
"""DP attention padding must not reach dense projection or MegaGate uninitialized."""

import math
from types import SimpleNamespace

import pytest

from tests.support.modules import load_patch


class Output:
    def __init__(self, rows, start=0):
        self.rows = rows
        self.start = start

    def __getitem__(self, key):
        assert isinstance(key, slice) and key.stop is None and key.step is None
        return Output(self.rows, self.start + key.start)

    def zero_(self):
        self.rows[self.start :] = [0.0] * (len(self.rows) - self.start)


@pytest.mark.parametrize(
    "decode,prefill,capacity",
    [(0, 5, 8), (3, 0, 8), (2, 3, 8), (0, 0, 8), (8, 0, 8), (21, 0, 24)],
)
@pytest.mark.parametrize("ppu", [True, False])
@pytest.mark.parametrize("metadata_present", [True, False])
def test_attention_initializes_only_padding_after_upstream(
    modules, decode, prefill, capacity, ppu, metadata_present
):
    calls = []
    active = decode + prefill
    swa = SimpleNamespace(num_decode_tokens=decode, num_prefill_tokens=prefill)
    metadata = {"layer.swa": swa} if metadata_present else None

    class Upstream:
        def forward_mqa(self, q, kv, positions, output):
            calls.append((q, kv, positions, output))
            # Real attention writes the actual queries; the allocation's unused
            # tail deliberately contains NaNs to make the regression deterministic.
            output.rows[:active] = [float(i + 1) for i in range(active)]
            if metadata is None:
                output.zero_()  # upstream's metadata-free profiling path

    modules("torch")
    modules(
        "vllm.forward_context",
        get_forward_context=lambda: SimpleNamespace(attn_metadata=metadata),
    )
    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: ppu))
    modules(
        "vllm.model_executor.layers.quantization.utils.mxfp8_utils",
        dequant_mxfp8_to_bf16=None,
    )
    modules("vllm.models.deepseek_v4.nvidia.ops.o_proj", deep_gemm_fp8_o_proj=None)
    modules(
        "vllm.models.deepseek_v41.nvidia.flashmla",
        DeepseekV4FlashMLAAttention=Upstream,
    )
    modules("vllm.models.deepseek_v41.sparse_mla", DeepseekV4FlashMLABackend=object)
    modules("vllm_sail.models.deepseek_v4.flashmla", DeepseekV4FlashMLAAttention=object)
    modules("vllm_sail.models.deepseek_v41.config", int8_w8a8_enabled=lambda: False)
    module = load_patch("vllm_sail/models/deepseek_v41/flashmla.py")
    attn = module.DeepseekV41FlashMLAAttention()
    attn.swa_cache_layer = SimpleNamespace(prefix="layer.swa")
    output = Output([float("nan")] * capacity)
    q, kv, positions = object(), object(), object()
    attn.forward_mqa(q, kv, positions, output)
    assert calls == [(q, kv, positions, output)]
    if not metadata_present:
        assert output.rows == [0.0] * capacity
    else:
        assert output.rows[:active] == [float(i + 1) for i in range(active)]
        if ppu:
            assert output.rows[active:] == [0.0] * (capacity - active)
        else:
            assert all(math.isnan(value) for value in output.rows[active:])
