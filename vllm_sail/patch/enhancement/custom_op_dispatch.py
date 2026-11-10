# SPDX-License-Identifier: Apache-2.0
"""Route ``CustomOp`` dispatch to ``forward_ppu`` when an op provides one.

Upstream ``CustomOp.dispatch_forward`` ends in a platform ladder that falls
through to ``forward_cuda``. PPU declares ``PlatformEnum.CUDA``, so it already
lands on ``forward_cuda`` — which is the right default and exactly what
``forward_ppu`` does in the in-tree fork.

What is missing is the ability for an individual op to *override* the CUDA
implementation on PPU. Five ops in the fork do:
``model_executor/layers/mhc.py`` (4 ops) and
``model_executor/layers/sparse_attn_indexer.py`` (1 op).

**Why a wrapper instead of a body copy.** The fork inserts an
``elif current_platform.is_ppu(): return self.forward_ppu`` branch into
``dispatch_forward``, which means carrying a verbatim copy of a ~35-line upstream
method that also handles ``_enforce_enable``, the enabled/disabled custom-op
bookkeeping and ``maybe_compile``. Copying it would silently diverge whenever
upstream touches any of that. Instead we delegate to the original and only
redirect the one case we care about: upstream chose ``forward_cuda`` *and* this
op class actually overrides ``forward_ppu``. Upstream can add platforms, change
the enable logic, or reorder the ladder without breaking us.
"""

from __future__ import annotations

from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.33.0,<0.34.0"

# Captured before patching so the replacement can delegate to it.
_upstream_dispatch_forward = CustomOp.dispatch_forward


@patch(
    "vllm.model_executor.custom_op",
    "CustomOp.forward_ppu",
    allow_missing=True,
    reason=(
        "Provides the default PPU implementation for every CustomOp: PPU ops are "
        "assumed CUDA-compatible, so the default delegates to forward_cuda. Ops "
        "with a genuinely different PPU kernel override this method."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream adds a forward_ppu hook, or PPU stops needing per-op overrides.",
)
def forward_ppu(self, *args, **kwargs):
    # By default, PPU ops are compatible with the CUDA implementation.
    return self.forward_cuda(*args, **kwargs)


@patch(
    "vllm.model_executor.custom_op",
    "CustomOp.dispatch_forward",
    reason=(
        "Upstream's platform ladder has no PPU branch, so an op that overrides "
        "forward_ppu never gets called. This wrapper delegates to upstream and "
        "only redirects when upstream selected forward_cuda AND the op class "
        "actually overrides forward_ppu."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream's dispatch ladder gains a PPU (or vendor-hook) branch.",
)
def dispatch_forward(self, compile_native: bool):
    method = _upstream_dispatch_forward(self, compile_native)

    if not current_platform.is_ppu():
        return method

    # Only intervene where upstream landed on the CUDA implementation. If it
    # chose forward_native (op disabled), forward_hip, forward_cpu, ... then it
    # had a reason and PPU must not second-guess it.
    if method != self.forward_cuda:
        return method

    # `type(self).forward_ppu is not CustomOp.forward_ppu` distinguishes a real
    # override from the inherited default. Calling the default would just bounce
    # back to forward_cuda, so skipping it keeps the returned callable identical
    # to upstream's for the common case — no extra frame in the hot path.
    if getattr(type(self), "forward_ppu", None) is not CustomOp.forward_ppu:
        return self.forward_ppu

    return method
