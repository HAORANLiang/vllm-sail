# SPDX-License-Identifier: Apache-2.0
"""Lazy PPU KDA SDK access; never probe the NVIDIA native-op stub."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType, SimpleNamespace

from vllm.platforms import current_platform

from vllm_sail import envs

_REPLAY_MODULE = "pla.decode.kda_replayssm_vllm"


def get_pla_kda_kernel(mode):
    if not current_platform.is_ppu() or not envs.VLLM_SAIL_USE_PLA:
        return None
    module, name = {
        "decode": ("pla.decode.kda", "fused_kda_decode_mega_forward"),
        "prefill": ("pla.prefill.flashkdapro", "flashkda_fwd"),
    }[mode]
    try:
        return getattr(import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            f"PPU KDA requires {module}.{name}; install a compatible PLA SDK "
            "or set VLLM_SAIL_USE_PLA=0 to use Triton."
        ) from exc


def get_pla_kda_replay() -> ModuleType | None:
    """PLA's vLLM KDA replay namespace, or None to keep Triton RecoverSSM.

    Unlike prefill and decode, replay is optional: an SDK built without it
    still serves RecoverSSM through upstream's Triton verify and commit.
    """
    if not current_platform.is_ppu() or not envs.VLLM_SAIL_USE_PLA:
        return None
    try:
        return import_module(_REPLAY_MODULE)
    except ImportError:
        return None


class PLAKDAReplayCommitContext:
    """Upstream's RecoverSSM commit-context interface over PLA replay commit."""

    # The specialized-decode deferred path reads ``backend`` to route commit;
    # a PLA context owns its own plan and never defers.
    backend = "pla"

    def __init__(self, context, spec_query_len: int) -> None:
        self._context = context
        self._spec_query_len = spec_query_len

    @classmethod
    def create(
        cls,
        layers,
        *,
        spec_query_len: int,
        max_num_reqs: int,
        backend: str = "triton",
    ) -> PLAKDAReplayCommitContext:
        # Signature parity with KDARecoverSSMCommitContext.create, so the two
        # commit contexts are drop-in for one another. The conv-state layout and
        # null block id are environment/module facts, not caller inputs, so
        # derive them here -- upstream's create reads is_conv_state_dim_first()
        # the same way. PLA ships one kernel set, so ``backend`` (upstream's
        # cute/triton plan selector) is accepted for parity and ignored.
        from vllm.model_executor.layers.mamba.mamba_utils import (
            is_conv_state_dim_first,
        )
        from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

        conv_state_layout = "DS" if is_conv_state_dim_first() else "SD"
        # PLA reads exactly ``(conv, state)`` and ``replay_records`` from each
        # layer; upstream layers carry more cache entries and other records.
        views = [
            SimpleNamespace(
                kv_cache=tuple(layer.kv_cache[:2]),
                replay_records=layer._pla_replay_records,
            )
            for layer in layers
        ]
        return cls(
            import_module(_REPLAY_MODULE).KDAReplayCommitContext.create(
                views,
                spec_query_len=spec_query_len,
                max_num_reqs=max_num_reqs,
                conv_state_layout=conv_state_layout,
                null_block_id=NULL_BLOCK_ID,
            ),
            spec_query_len,
        )

    def commit(
        self,
        num_accepted_tokens,
        state_indices,
        query_start_loc,
        request_indices=None,
        block_table=None,
        num_computed_tokens=None,
        mamba_block_size: int | None = None,
    ) -> None:
        """Fold the sampled tokens in; ``num_accepted_tokens`` keeps the bonus."""
        batch = state_indices.shape[0]
        if batch == 0:
            return
        # Upstream's mapping may be longer than the real batch; PLA requires
        # exactly one request row per committed query.
        if request_indices is not None:
            request_indices = request_indices[:batch]
        # PLA requires the verify projection's token count and skips query
        # intervals beyond it. Commit only sees device tensors, and a value
        # stashed at verify time would be stale under CUDA-graph replay, so
        # pass the batch * spec_query_len upper bound instead: every committed
        # request holds at most spec_query_len rows, so the bound never skips
        # a real interval.
        self._context.commit(
            num_accepted_tokens,
            state_indices,
            query_start_loc,
            request_indices=request_indices,
            block_table=block_table,
            num_computed_tokens=num_computed_tokens,
            mamba_block_size=mamba_block_size,
            token_capacity=batch * self._spec_query_len,
        )
