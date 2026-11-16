# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""Native GDN MTP accuracy against unfused Triton and eager PyTorch.

Retains vLLM v0.31.0 test_fused_gdn_post_conv.py's head/TP baseline and extends
its dtype, layout, padding, numerical-boundary and multi-round coverage. Every
round checks output, per-token state writes and untouched storage. The eager
oracle has separate worked-example CPU tests; none substitutes for a PPU run.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from ._gdn_mtp_utils import (
    apply_signal_profile,
    check_case,
    check_round,
    initial_states,
    make_inputs,
)

pytestmark = pytest.mark.ppu


@pytest.fixture(scope="module")
def runtime():
    # Delay runtime imports so a CPU torch environment can collect the complete
    # matrix. Missing native kernels on an actual PPU are failures, not skips.
    if not torch.cuda.is_available():
        pytest.skip("requires a real PPU")
    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.platforms import current_platform
    from vllm.third_party.flash_linear_attention.ops import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn

    from vllm_sail.attention.gdn_decode import has_fused_gdn_kernel
    from vllm_sail.attention.pla_decode import get_sail_cuda_pla_k_last

    if not current_platform.is_ppu():
        pytest.skip("requires the PPU platform")
    assert has_fused_gdn_kernel(), "Build/install the native GDN MTP kernel"
    return SimpleNamespace(
        ops=ops,
        update=fused_sigmoid_gating_delta_rule_update,
        rmsnorm=rmsnorm_fn,
        layer=QwenGatedDeltaNetAttention,
        pla_handle=get_sail_cuda_pla_k_last,
    )


@pytest.fixture(autouse=True)
def independent_triton_reference(runtime, monkeypatch):
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "triton")
    assert runtime.pla_handle() is None, "reference must not dispatch to PLA"


@pytest.mark.parametrize(
    "head_ratio,tp_size,lengths,state_dtype,norm_dtype",
    [
        pytest.param(8, 16, (4, 4), torch.bfloat16, torch.bfloat16, id="tp16-bf16"),
        pytest.param(8, 4, (4, 4), torch.float32, torch.float32, id="tp4-fp32"),
        pytest.param(8, 16, (4, 2, 0), torch.bfloat16, torch.float32, id="tp16-ragged"),
        pytest.param(8, 4, (4, 2, 0), torch.float32, torch.bfloat16, id="tp4-ragged"),
        pytest.param(8, 16, (9,), torch.float32, torch.bfloat16, id="tp16-max"),
        pytest.param(8, 4, (9,), torch.bfloat16, torch.float32, id="tp4-max"),
        pytest.param(1, 1, (4, 4), torch.float32, torch.bfloat16, id="ratio1-tp1"),
        pytest.param(2, 1, (4, 2, 0), torch.bfloat16, torch.bfloat16, id="ratio2-tp1"),
        pytest.param(2, 4, (8,), torch.float32, torch.float32, id="ratio2-tp4"),
        pytest.param(3, 2, (4, 2, 0), torch.float32, torch.float32, id="ratio3-tp2"),
        pytest.param(4, 4, (8,), torch.float32, torch.bfloat16, id="ratio4-tp4"),
    ],
)
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@pytest.mark.parametrize("seed", [0, 17])
@torch.inference_mode()
def test_upstream_head_and_tp_baseline(
    runtime,
    record_property,
    head_ratio,
    tp_size,
    lengths,
    state_dtype,
    norm_dtype,
    activation,
    seed,
):
    inp = make_inputs(
        lengths,
        head_ratio=head_ratio,
        key_heads=16 // tp_size,
        state_dtype=state_dtype,
        norm_dtype=norm_dtype,
        activation=activation,
        seed=seed,
        permuted=False,
    )
    check_case(inp, runtime, record_property)


@pytest.mark.parametrize("head_ratio", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("norm_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias_dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@torch.inference_mode()
def test_all_head_and_dtype_combinations(
    runtime,
    record_property,
    head_ratio,
    state_dtype,
    norm_dtype,
    bias_dtype,
    activation,
):
    inp = make_inputs(
        (8, 5, 1, 0),
        head_ratio=head_ratio,
        state_dtype=state_dtype,
        norm_dtype=norm_dtype,
        bias_dtype=bias_dtype,
        activation=activation,
        seed=29,
    )
    check_case(inp, runtime, record_property)


@pytest.mark.parametrize("length", range(1, 10))
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@torch.inference_mode()
def test_every_supported_mtp_length(
    runtime,
    record_property,
    length,
    state_dtype,
    activation,
):
    inp = make_inputs(
        (length, max(1, length - 1), 0),
        key_heads=2,
        head_ratio=3,
        state_dtype=state_dtype,
        norm_dtype=torch.bfloat16,
        bias_dtype=torch.float16,
        activation=activation,
        seed=length,
    )
    check_case(inp, runtime, record_property, eager=True)


@pytest.mark.parametrize(
    "layout", ["contiguous", "packed", "padded_rows", "gapped_state"]
)
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@torch.inference_mode()
def test_supported_strides_and_state_slot_alignment(
    runtime,
    record_property,
    layout,
    state_dtype,
    activation,
):
    inp = make_inputs(
        (8, 3, 1),
        layout=layout,
        state_dtype=state_dtype,
        activation=activation,
        seed=37,
    )
    check_case(inp, runtime, record_property, eager=True)


@pytest.mark.parametrize(
    "profile",
    [
        "zero_q",
        "zero_k",
        "tiny_qk",
        "large_qk",
        "zero_gate",
        "softplus_threshold",
        "strong_decay",
        "sigmoid_saturation",
    ],
)
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@torch.inference_mode()
def test_numerical_boundaries(
    runtime, record_property, profile, state_dtype, activation
):
    inp = make_inputs(
        (8, 2, 0),
        head_ratio=4,
        state_dtype=state_dtype,
        activation=activation,
        seed=41,
    )
    apply_signal_profile(inp, profile)
    check_case(inp, runtime, record_property, eager=True)


@pytest.mark.parametrize("eps", [0.0, 1e-6, 1e-3])
@pytest.mark.parametrize("scale", [128**-0.5, 0.5])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@torch.inference_mode()
def test_norm_epsilon_and_query_scale(
    runtime,
    record_property,
    eps,
    scale,
    state_dtype,
    activation,
):
    inp = make_inputs(
        (8, 3),
        state_dtype=state_dtype,
        activation=activation,
        eps=eps,
        scale=scale,
        seed=43,
    )
    check_case(inp, runtime, record_property, eager=True)


@pytest.mark.parametrize("padding", ["front", "middle", "end", "null", "negative"])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@torch.inference_mode()
def test_padding_and_null_source_isolation(
    runtime,
    record_property,
    padding,
    state_dtype,
    activation,
):
    lengths = {"front": (0, 4, 2), "middle": (4, 0, 2), "end": (4, 2, 0)}.get(
        padding, (4, 3, 2)
    )
    inp = make_inputs(lengths, state_dtype=state_dtype, activation=activation, seed=47)
    if padding in {"null", "negative"}:
        inp.kwargs["state_indices"][1].fill_(0 if padding == "null" else -1)
    check_case(inp, runtime, record_property, eager=True)


@pytest.mark.parametrize("destination", [0, -1])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@torch.inference_mode()
def test_null_candidate_destinations_keep_other_slots_intact(
    runtime,
    record_property,
    destination,
    state_dtype,
    activation,
):
    inp = make_inputs((8, 3), state_dtype=state_dtype, activation=activation, seed=53)
    inp.kwargs["state_indices"][:, 1::2] = destination
    check_case(inp, runtime, record_property, eager=True)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@pytest.mark.parametrize("seed", [0, 61])
@torch.inference_mode()
def test_ten_rounds_of_rewind_with_changing_request_lengths(
    runtime,
    record_property,
    state_dtype,
    activation,
    seed,
):
    schedules = ((8, 5, 1, 0), (3, 2, 7, 0), (1, 8, 4, 0), (6, 1, 8, 0))
    inp = make_inputs(
        schedules[0],
        width=8,
        head_ratio=3,
        key_heads=2,
        state_dtype=state_dtype,
        activation=activation,
        seed=seed,
    )
    states = initial_states(inp, eager=True)
    indices = inp.kwargs["state_indices"]
    previous = schedules[0]
    for step in range(10):
        inp = make_inputs(
            schedules[step % len(schedules)],
            width=8,
            head_ratio=3,
            key_heads=2,
            state_dtype=state_dtype,
            activation=activation,
            seed=seed + step,
            indices=indices,
        )
        accepted = [
            1 + (step + request) % length if length else 1
            for request, length in enumerate(previous)
        ]
        inp.kwargs["num_accepted_tokens"].copy_(torch.tensor(accepted, device="cuda"))
        check_round(inp, states, runtime, record_property, step=step)
        previous = inp.lengths


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@torch.inference_mode()
def test_layer_post_conv_entrypoint_matches_both_references(
    runtime,
    record_property,
    state_dtype,
    activation,
):
    inp = make_inputs(
        (8, 3, 0),
        state_dtype=state_dtype,
        activation=activation,
        seed=67,
    )
    check_case(inp, runtime, record_property, eager=True, layer_entry=True)


@pytest.mark.parametrize(
    "lengths,width,message",
    [((0,), 1, "at least one token"), ((8,), 10, "1 <= S <= 9")],
)
def test_empty_input_and_excess_mtp_width_are_rejected(
    runtime, lengths, width, message
):
    inp = make_inputs(lengths, width=width)
    with pytest.raises(RuntimeError, match=message):
        runtime.ops.fused_gdn_decode_post_conv_mtp(**inp.kwargs, state=inp.state)
