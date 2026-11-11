# SPDX-License-Identifier: Apache-2.0
"""Communication regressions with CPU-only dependencies."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    path = (
        ROOT
        / "vllm_sail/model_executor/layers/fused_moe/prepare_finalize"
        / f"{name}.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("ranks", [8, 16])
@pytest.mark.parametrize("draft_first", [False, True])
def test_dspark_buffer_allocation_is_independent_of_model_order(ranks, draft_first):
    normalize = load("deepep_buffer").dspark_buffer_args
    config = types.SimpleNamespace(
        speculative_config=types.SimpleNamespace(use_dspark=lambda: True),
        model_config=types.SimpleNamespace(
            hf_text_config=types.SimpleNamespace(
                n_routed_experts=384, dspark_n_routed_experts=128
            )
        ),
    )
    allocations = []
    for experts in [128, 384] if draft_first else [384, 128]:
        args = dict(
            num_ep_ranks=ranks,
            num_global_experts=experts,
            num_local_experts=experts // ranks,
            max_num_tokens_per_dp_rank=256,
            token_hidden_size=5120,
        )
        allocations.append(normalize(args, config))
        assert args["num_global_experts"] == experts
    assert allocations[0] == allocations[1]
    assert allocations[0]["num_global_experts"] == 384
    config.speculative_config = None
    assert normalize(args, config) is args


@pytest.mark.parametrize(
    "quant_dtype,defer", [("mxfp4", False), ("mxfp4", True), ("fp8", False)]
)
@pytest.mark.parametrize("tokens", [0, 3])
def test_ht_quantizes_mxfp4_only_after_dispatch(
    monkeypatch, quant_dtype, defer, tokens
):
    calls = []
    x = types.SimpleNamespace(shape=(tokens, 5120))
    packed = types.SimpleNamespace(shape=(tokens, 2560))
    scales = object()
    meta, ids, weights = object(), object(), object()

    class Base:
        def prepare_async(self, *args, defer_input_quant=False):
            calls.append(("dispatch", args[0], defer_input_quant))

            def receive():
                calls.append("receive")
                return x, None, meta, ids, weights

            return receive

    def quantize(input_, scale, **kwargs):
        assert input_ is x
        calls.append(("quantize", kwargs))
        return packed, scales

    name = "vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht"
    monkeypatch.setitem(
        sys.modules, name, types.SimpleNamespace(DeepEPHTPrepareAndFinalize=Base)
    )
    name = "vllm.model_executor.layers.fused_moe.utils"
    monkeypatch.setitem(
        sys.modules, name, types.SimpleNamespace(moe_kernel_quantize_input=quantize)
    )
    module = load("deepep_ht")
    config = types.SimpleNamespace(
        quant_dtype=quant_dtype,
        a1_scale=None,
        per_act_token_quant=False,
        block_shape=[1, 32],
        is_scale_swizzled=False,
        mx_alignment=32,
    )
    receiver = module.PPUDeepEPHTPrepareAndFinalize().prepare_async(
        x, weights, ids, 384, None, False, config, defer
    )
    needs_quant = quant_dtype == "mxfp4" and not defer
    assert calls == [("dispatch", x, defer or needs_quant)]
    result = receiver()
    assert result[2:] == (meta, ids, weights)
    assert result[0] is (packed if needs_quant else x)
    assert result[1] is (scales if needs_quant else None)
    assert calls[1] == "receive"
    assert len(calls) == (3 if needs_quant else 2)
