# SPDX-License-Identifier: Apache-2.0
"""Profile DSpark target activations at the reachable scheduler token budget."""

import functools
import logging

from vllm_sail.patch.utils import patch

_MODULE = "vllm.v1.worker.gpu.model_runner"


def install() -> None:
    from vllm.platforms import current_platform
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    original = GPUModelRunner._dummy_run

    @functools.wraps(original)
    def dummy_run(self, num_tokens, *args, is_profile=False, **kwargs):
        spec = self.vllm_config.speculative_config
        budget = getattr(
            self.vllm_config.scheduler_config, "max_num_scheduled_tokens", None
        )
        if (
            is_profile
            and current_platform.is_ppu()
            and spec is not None
            and spec.use_dspark()
            and budget is not None
            and 0 < budget < num_tokens
        ):
            # The scheduler already counts target verification tokens in this
            # budget. Keep runner capacity intact: the original dummy run still
            # profiles the separate draft pass and its extra query slots.
            logging.getLogger("vllm.vllm_sail.profile_token_budget").info(
                "DSpark memory profile target tokens: %d -> %d; "
                "runner capacity remains %d.",
                num_tokens, budget, self.max_num_tokens,
            )
            num_tokens = budget
        return original(self, num_tokens, *args, is_profile=is_profile, **kwargs)

    patch(
        _MODULE,
        "GPUModelRunner._dummy_run",
        reason=(
            "PPU DSpark can reserve a large MegaMoE/runner capacity while "
            "max_num_scheduled_tokens limits target batches. Profiling the "
            "unreachable capacity overestimates activation memory and can "
            "leave a negative KV cache budget."
        ),
        affected_versions=">=0.33.0,<0.34.0",
        remove_when=(
            "The V2 GPU model runner profiles target activations using the "
            "scheduler token limit while preserving draft and buffer capacity."
        ),
    )(dummy_run)
