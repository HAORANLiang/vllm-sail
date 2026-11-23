# SPDX-License-License: Apache-2.0
"""DP padding must not reach the mega path's projection or MegaGate uninitialized.

PR#355 covered the non-mega FlashMLA path, whose ``forward_mqa`` receives a
plain tensor output. The fused mega-attention path receives a
``QuantizedActivation`` (data + scale pair) and upstream writes only the
decode rows (``0 .. num_decode_tokens``) plus the prefill rows (up to
``num_decode_tokens + num_prefill_tokens``). This test pins the PPU tail
zeroing added to ``PPUDeepseekV4MegaAttnAttention.forward_mqa``: the exact
rows upstream wrote stay untouched, and only the DP padding tail (data and
scale) is cleared on PPU. Non-PPU platforms keep the uninitialized tail, and
the metadata-free warmup path is unaffected because upstream already zeroes
the whole buffer there.
"""

import math
from types import SimpleNamespace

import pytest

from tests.support.modules import load_patch


class RowSlice:
    """A ``data``/``scale``-style view supporting ``[n:].zero_()``."""

    def __init__(self, rows, start=0):
        self.rows = rows
        self.start = start

    def __getitem__(self, key):
        assert isinstance(key, slice) and key.stop is None and key.step is None
        return RowSlice(self.rows, self.start + key.start)

    def zero_(self):
        self.rows[self.start :] = [0.0] * (len(self.rows) - self.start)


class QuantizedOutput:
    """Minimal ``QuantizedActivation`` stand-in (data + scale token rows)."""

    def __init__(self, data_rows, scale_rows):
        self.data = RowSlice(data_rows)
        self.scale = RowSlice(scale_rows)


@pytest.mark.parametrize(
    "decode,prefill,capacity",
    [(0, 5, 8), (3, 0, 8), (2, 3, 8), (0, 0, 8), (8, 0, 8), (21, 0, 24)],
)
@pytest.mark.parametrize("ppu", [True, False])
@pytest.mark.parametrize("metadata_present", [True, False])
def test_mega_attention_initializes_only_padding_after_upstream(
    modules, decode, prefill, capacity, ppu, metadata_present
):
    calls = []
    active = decode + prefill
    swa = SimpleNamespace(num_decode_tokens=decode, num_prefill_tokens=prefill)
    metadata = {"layer.swa": swa} if metadata_present else None

    class UpstreamMega:
        def forward_mqa(self, q, kv, positions, output):
            calls.append((q, kv, positions, output))
            # Real kernels write only the active rows; the allocation's unused
            # tail deliberately contains NaNs to make the regression
            # deterministic for both the data and the scale half.
            output.data.rows[:active] = [float(i + 1) for i in range(active)]
            output.scale.rows[:active] = [float(-(i + 1)) for i in range(active)]
            if metadata is None:
                output.data.zero_()  # upstream's metadata-free warmup path
                output.scale.zero_()

    # Mirror the gate-test mock surface: the module under test only needs its
    # imports satisfied; no kernel runs in this test.
    modules("torch")
    modules(
        "vllm.logger",
        init_logger=lambda name: SimpleNamespace(info_once=lambda *a, **k: None),
    )
    modules(
        "vllm.model_executor.layers.fusion.quant_activation",
        QuantizedActivation=type("QuantizedActivation", (), {}),
    )
    modules(
        "vllm.model_executor.layers.quantization.utils.mxfp8_utils",
        dequant_mxfp8_to_bf16=lambda weight, scale: weight,
    )
    modules(
        "vllm.model_executor.layers.quantization.utils.quant_utils",
        kFp8DynamicTokenSym="kFp8DynamicTokenSym",
    )
    modules(
        "vllm.models.deepseek_v41.common.ops",
        combine_topk_swa_indices=lambda *a, **k: None,
        dequantize_and_gather_k_cache=lambda *a, **k: None,
    )
    modules("vllm.models.deepseek_v41.common.ops.fused_layout", WV_GROUP_SIZE=8)
    modules(
        "vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn",
        _HEAD_DIM_V=512,
        _QUANT_GROUP=32,
        _ROPE_DIM=64,
        _token_slice=lambda out, start, end: out,
        alloc_mega_attn_output=lambda num_tokens, n_wv_group, device: None,
        DeepseekV4MegaAttnAttention=UpstreamMega,
    )
    modules(
        "vllm.models.deepseek_v41.sparse_mla",
        DeepseekV4FlashMLABackend=type("DeepseekV4FlashMLABackend", (), {}),
    )
    modules("vllm.utils.math_utils", round_up=lambda x, y: -(-x // y) * y)
    modules("vllm.v1.worker.workspace", current_workspace_manager=lambda: None)
    modules(
        "vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: ppu)
    )
    modules(
        "vllm.forward_context",
        get_forward_context=lambda: SimpleNamespace(attn_metadata=metadata),
    )
    modules(
        "vllm_sail.envs",
        VLLM_SAIL_FUSED_ATTN_PER_TOKEN_SCALE=False,
        VLLM_SAIL_PPU_KV_NVFP4=False,
        VLLM_SAIL_PPU_KV_MXFP8=False,
    )

    module = load_patch("vllm_sail/models/deepseek_v41/flash_mla_mega_attn.py")
    attn = module.PPUDeepseekV4MegaAttnAttention()
    attn.swa_cache_layer = SimpleNamespace(prefix="layer.swa")
    output = QuantizedOutput(
        [float("nan")] * capacity, [float("nan")] * capacity
    )
    q, kv, positions = object(), object(), object()
    attn.forward_mqa(q, kv, positions, output)

    # Upstream forward always runs exactly once, unconditionally.
    assert calls == [(q, kv, positions, output)]
    if not metadata_present:
        # Warmup path: upstream zeroed the whole buffer; the tail fix must
        # not disturb it (zeroing zeros is a no-op for the active part only).
        assert output.data.rows == [0.0] * capacity
        assert output.scale.rows == [0.0] * capacity
    else:
        assert output.data.rows[:active] == [float(i + 1) for i in range(active)]
        assert output.scale.rows[:active] == [float(-(i + 1)) for i in range(active)]
        if ppu:
            # The regression: only the DP padding tail is cleared.
            assert output.data.rows[active:] == [0.0] * (capacity - active)
            assert output.scale.rows[active:] == [0.0] * (capacity - active)
        else:
            # Non-PPU keeps upstream semantics: the tail stays untouched.
            assert all(math.isnan(v) for v in output.data.rows[active:])
            assert all(math.isnan(v) for v in output.scale.rows[active:])
