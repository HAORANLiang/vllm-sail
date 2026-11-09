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

    def __init__(self, context) -> None:
        self._context = context

    @classmethod
    def create(
        cls,
        layers,
        *,
        spec_query_len: int,
        max_num_reqs: int,
        conv_state_layout: str,
        null_block_id: int,
    ) -> PLAKDAReplayCommitContext:
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
                null_block_id=null_block_id,
            )
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
        self._context.commit(
            num_accepted_tokens,
            state_indices,
            query_start_loc,
            request_indices=request_indices,
            block_table=block_table,
            num_computed_tokens=num_computed_tokens,
            mamba_block_size=mamba_block_size,
        )
