# SPDX-License-Identifier: Apache-2.0
"""Separate reachable target activation peaks from persistent runner capacity."""

from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch


@pytest.fixture
def harness(modules, patch_utils_module):
    state = NS(ppu=True, dspark=True, calls=[])
    modules("vllm.platforms", current_platform=NS(is_ppu=lambda: state.ppu))

    class Runner:
        max_num_tokens = 32768

        def _dummy_run(self, num_tokens, *args, **kwargs):
            state.calls.append((num_tokens, args, kwargs, self.max_num_tokens))
            return "profile-output"

    original = Runner._dummy_run
    modules("vllm.v1.worker.gpu.model_runner", GPUModelRunner=Runner)
    module = load_patch("vllm_sail/patch/bugfix/dspark_profile_token_budget.py")
    module.install()
    runner = Runner()
    runner.vllm_config = NS(
        speculative_config=NS(use_dspark=lambda: state.dspark),
        scheduler_config=NS(max_num_scheduled_tokens=4096),
    )
    return state, runner, module, original


def test_profiles_reachable_target_but_preserves_draft_buffer_capacity(harness):
    state, runner, _, _ = harness
    assert (
        runner._dummy_run(32768, "arg", is_profile=True, skip_attn=True)
        == "profile-output"
    )
    assert state.calls == [
        (4096, ("arg",), {"is_profile": True, "skip_attn": True}, 32768)
    ]
    assert runner.max_num_tokens == 32768
    assert runner.vllm_config.scheduler_config.max_num_scheduled_tokens == 4096


@pytest.mark.parametrize("tokens", [1, 768, 4096])
def test_never_increases_a_smaller_profile_batch(harness, tokens):
    state, runner, _, _ = harness
    runner._dummy_run(tokens, is_profile=True)
    assert state.calls[0][0] == tokens


@pytest.mark.parametrize("is_profile", [False, True])
@pytest.mark.parametrize(
    "ppu,dspark,budget",
    [
        (False, True, 4096),
        (True, False, 4096),
        (True, True, None),
        (True, True, 0),
        (True, True, -1),
        (True, True, 32768),
    ],
)
def test_unaffected_platforms_methods_and_budgets_delegate(
    harness, is_profile, ppu, dspark, budget
):
    state, runner, _, _ = harness
    state.ppu, state.dspark = ppu, dspark
    runner.vllm_config.scheduler_config.max_num_scheduled_tokens = budget
    runner._dummy_run(32768, is_profile=is_profile, randomize_inputs=True)
    assert state.calls == [
        (32768, (), {"is_profile": is_profile, "randomize_inputs": True}, 32768)
    ]


def test_graph_warmup_is_not_capped(harness):
    state, runner, _, _ = harness
    runner._dummy_run(8192, uniform_decode=True)
    assert state.calls[0][0] == 8192


def test_no_speculation_delegates(harness):
    state, runner, _, _ = harness
    runner.vllm_config.speculative_config = None
    runner._dummy_run(32768, is_profile=True)
    assert state.calls[0][0] == 32768


def test_scheduler_without_a_separate_token_budget_delegates(harness):
    state, runner, _, _ = harness
    del runner.vllm_config.scheduler_config.max_num_scheduled_tokens
    runner._dummy_run(32768, is_profile=True)
    assert state.calls[0][0] == 32768


def test_patch_records_original_and_rejects_reinstallation(harness, patch_utils_module):
    _, runner, module, original = harness
    marker = getattr(runner._dummy_run, patch_utils_module.PATCH_MARKER)
    assert (
        marker["vllm.v1.worker.gpu.model_runner.GPUModelRunner._dummy_run"] is original
    )
    with pytest.raises(RuntimeError, match="already patched"):
        module.install()
