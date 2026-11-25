# SPDX-License-Identifier: Apache-2.0
"""Idle replay rows must stay masked when graph capacity exceeds live rows."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from tests.support.modules import load_patch


class Tensor:
    device = "fake"

    def __init__(self, values):
        self.data = np.asarray(values)
        self.shape = self.data.shape

    def __getitem__(self, key):
        return Tensor(self.data[key])

    def __invert__(self):
        return Tensor(~self.data)

    def fill_(self, value):
        self.data.fill(value)


@pytest.fixture
def harness(modules, patch_utils_module):
    state = NS(ppu=True, context=None, prepared=0, original_calls=0, native_calls=[])

    def index_select(source, dim, indices, *, out):
        out.data[:] = np.take(source.data, indices.data, axis=dim)

    modules(
        "torch",
        bool=np.bool_,
        int64=np.int64,
        float32=np.float32,
        empty_like=lambda x, dtype: Tensor(np.empty(x.shape, dtype=dtype)),
        empty=lambda shape, dtype, device: Tensor(np.empty(shape, dtype=dtype)),
        index_select=index_select,
    )
    modules("vllm.platforms", current_platform=NS(is_ppu=lambda: state.ppu))
    modules(
        "vllm.forward_context",
        is_forward_context_available=lambda: state.context is not None,
        get_forward_context=lambda: state.context,
    )

    class ModelState:
        pcp_size = 1

        def _prepare_replay_batch(self, input_batch, *args, **kwargs):
            state.prepared += 1
            return "prepared"

    modules(
        "vllm.models.deepseek_v41.nvidia.model_state",
        DeepseekV41ModelState=ModelState,
    )

    def gate_original(*args, **kwargs):
        state.original_calls += 1
        return "original"

    def native(x, weight, topk, *, mask, out, **kwargs):
        state.native_calls.append((mask, kwargs))
        # Model the native invalid-score check: padded NaNs are ignored, while
        # a NaN belonging to a real request must still surface as a failure.
        assert np.isfinite(x.data[mask.data]).all(), "invalid real row"
        indices, weights = out
        indices.data[:] = np.where(mask.data[:, None], 2, -1)
        weights.data[:] = np.where(mask.data[:, None], 1.0, 0.0)

    dg = modules(
        "vllm.utils.deep_gemm",
        bf16_mega_gate=gate_original,
        _lazy_init=lambda: None,
        _bf16_mega_gate_impl=native,
    )
    consumer = modules(
        "vllm.models.deepseek_v4.nvidia.model", bf16_mega_gate=gate_original
    )
    module = load_patch("vllm_sail/patch/bugfix/deepseek_v41_replay_padding.py")
    module.install()
    model = ModelState()
    model._kept_rows = Tensor(np.zeros(8, dtype=np.int64))
    context = NS(is_padding=None, batch_descriptor=NS(num_tokens=4))
    batch = NS(rows=Tensor([1, 4]), forward_context=context)
    model.decoder_replay_layers = NS(replay_batch=batch)
    return state, model, batch, dg, consumer, gate_original


def test_replay_mask_tracks_selected_rows_and_reuses_captured_storage(harness):
    state, model, batch, *_ = harness
    parent = NS(is_padding=Tensor([False, True, False, False, False, True]))
    assert model._prepare_replay_batch(parent) == "prepared"
    captured_mask = batch.forward_context.is_padding
    assert captured_mask.data.tolist() == [True, False, True, True]
    allocation = model._sail_replay_padding

    batch.rows = Tensor([0, 1, 2])
    model._prepare_replay_batch(parent)
    assert model._sail_replay_padding is allocation
    assert np.shares_memory(captured_mask.data, batch.forward_context.is_padding.data)
    assert captured_mask.data.tolist() == [False, True, False, True]
    assert state.prepared == 2


@pytest.mark.parametrize("ppu,pcp", [(False, 1), (True, 2)])
def test_other_platform_and_existing_pcp_mask_are_preserved(harness, ppu, pcp):
    state, model, batch, *_ = harness
    state.ppu, model.pcp_size = ppu, pcp
    existing = Tensor([True, False, True, True])
    batch.forward_context.is_padding = existing
    assert model._prepare_replay_batch(NS()) == "prepared"
    assert batch.forward_context.is_padding is existing
    assert not hasattr(model, "_sail_replay_padding")


def test_no_replay_batch_delegates_without_allocating(harness):
    _, model, *_ = harness
    model.decoder_replay_layers.replay_batch = None
    assert model._prepare_replay_batch(NS()) == "prepared"
    assert not hasattr(model, "_sail_replay_padding")


def test_native_gate_masks_nan_padding_but_checks_real_rows(harness):
    state, _, _, dg, consumer, _ = harness
    state.context = NS(is_padding=Tensor([False, True, True]))
    x = Tensor([[1.0], [float("nan")], [float("inf")]])
    weights, indices = consumer.bf16_mega_gate(x, "weight", 2, ep_rank=3)
    assert indices.data.tolist() == [[2, 2], [-1, -1], [-1, -1]]
    assert weights.data.tolist() == [[1, 1], [0, 0], [0, 0]]
    assert state.native_calls[0][0].data.tolist() == [True, False, False]
    assert state.native_calls[0][1]["ep_rank"] == 3
    assert state.original_calls == 0
    assert consumer.bf16_mega_gate is dg.bf16_mega_gate

    state.context.is_padding.data[1] = False
    with pytest.raises(AssertionError, match="invalid real row"):
        dg.bf16_mega_gate(x, "weight", 2)


@pytest.mark.parametrize("ppu,context", [(False, True), (True, False), (True, None)])
def test_unaffected_gate_paths_delegate_to_original(harness, ppu, context):
    state, _, _, dg, _, _ = harness
    state.ppu = ppu
    state.context = NS(is_padding=Tensor([True])) if context else None
    if context is None:
        state.context = NS(is_padding=None)
    assert dg.bf16_mega_gate(Tensor([[1.0]]), "weight", 1) == "original"
    assert state.original_calls == 1
    assert not state.native_calls


def test_patch_preserves_original_bindings(harness, patch_utils_module):
    _, model, _, dg, consumer, original = harness
    prepare = getattr(model._prepare_replay_batch, patch_utils_module.PATCH_MARKER)
    prepare_target = (
        "vllm.models.deepseek_v41.nvidia.model_state."
        "DeepseekV41ModelState._prepare_replay_batch"
    )
    assert prepare[prepare_target].__name__ == "_prepare_replay_batch"
    originals = getattr(dg.bf16_mega_gate, patch_utils_module.PATCH_MARKER)
    assert originals["vllm.utils.deep_gemm.bf16_mega_gate"] is original
    assert originals["vllm.models.deepseek_v4.nvidia.model.bf16_mega_gate"] is original
    assert consumer.bf16_mega_gate is dg.bf16_mega_gate


def test_reinstallation_is_rejected_without_replacing_active_patches(harness):
    _, model, _, dg, consumer, _ = harness
    prepare, gate = model._prepare_replay_batch.__func__, dg.bf16_mega_gate
    module = load_patch("vllm_sail/patch/bugfix/deepseek_v41_replay_padding.py")
    with pytest.raises(RuntimeError, match="already patched"):
        module.install()
    assert model._prepare_replay_batch.__func__ is prepare
    assert dg.bf16_mega_gate is gate
    assert consumer.bf16_mega_gate is gate
