# SPDX-License-Identifier: Apache-2.0
"""Upstream fixes with explicit installation and dependency-free imports."""


def install() -> None:
    from vllm_sail.patch.bugfix import (
        async_llm_shutdown,
        deepseek_v41_replay_padding,
        dp_idle_mtp_dummy_batch,
        dspark_profile_token_budget,
        glm5_next_int8_mla_dequant,
        qwen4_exp_ple_quant_scheme,
    )

    async_llm_shutdown.install()
    dp_idle_mtp_dummy_batch.install()
    deepseek_v41_replay_padding.install()
    dspark_profile_token_budget.install()
    glm5_next_int8_mla_dequant.install()
    qwen4_exp_ple_quant_scheme.install()
