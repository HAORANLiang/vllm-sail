# SPDX-License-Identifier: Apache-2.0
"""PPU change to ``vllm.model_executor.models.llama4``.

Ports the fork's +1 hunk: ``Llama4MoE.custom_routing_function`` runs inside
the compiled model region on PPU and must itself be ``torch.compile``-d, or
the router becomes a graph break. Delegating: the replacement calls a
module-level ``torch.compile`` of the captured upstream function.
"""

from __future__ import annotations

import torch
from vllm.model_executor.models.llama4 import Llama4MoE

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.33.0,<0.34.0"

# Captured before patching; torch.compile is lazy, so this is cheap at
# import time and mirrors the fork's decorator placement.
_upstream_custom_routing_function = Llama4MoE.custom_routing_function
_compiled_custom_routing_function = torch.compile(_upstream_custom_routing_function)


@patch(
    "vllm.model_executor.models.llama4",
    "Llama4MoE.custom_routing_function",
    reason=(
        "The fork wraps the Llama4 custom router in torch.compile so it "
        "fuses into the compiled model graph on PPU; without it the router "
        "is a graph break. Delegates by calling the compiled upstream "
        "function."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream applies torch.compile to Llama4MoE.custom_routing_function "
        "itself."
    ),
)
def custom_routing_function(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _compiled_custom_routing_function(
        hidden_states, gating_output, topk, renormalize
    )
