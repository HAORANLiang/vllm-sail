# SPDX-License-Identifier: Apache-2.0
"""TEMPLATE — module-attribute patch. Copy into bugfix/, enhancement/ or
performance/, then replace every TODO. This file is excluded from lint and is
never imported.

Use this form for a module-level function, class, or Triton kernel. The
replacement object is installed **unchanged**, which is what makes Triton objects
patchable — and why ``@patch`` must be the outermost decorator.
"""

from __future__ import annotations

# TODO: capture the upstream implementation here if you intend to delegate to it
# (preferred over copying its body). Import the module, not the symbol, so the
# capture happens before the patch replaces it.
#
#   from vllm import some_module
#   _upstream_target_function = some_module.target_function
from vllm_sail.patch.utils import patch


@patch(
    # TODO: module that owns the target.
    "vllm.some_module",
    # TODO: attribute name. Omit if it equals the replacement's __name__.
    "target_function",
    # TODO: why this patch exists. Be specific about what PPU needs and what
    # upstream does instead. One or two sentences.
    reason="TODO",
    # TODO: vLLM versions this applies to, e.g. ">=0.33.0,<0.34.0".
    affected_versions="TODO",
    # TODO: a VERIFIABLE removal condition — an upstream PR number, a named API,
    # or a capability. "When upstream fixes it" is not acceptable.
    remove_when="TODO",
)
def target_function(arg: int) -> int:
    # If delegating, call the captured upstream implementation and adjust:
    #
    #   from vllm.platforms import current_platform   # lazy: inside the body
    #   result = _upstream_target_function(arg)
    #   return _ppu_adjust(result) if current_platform.is_ppu() else result
    #
    # If you must copy the upstream body, copy it VERBATIM — same signature,
    # decorators, comments and unchanged lines — and mark only your changes:
    #
    #   # PPU MODIFICATION: begin
    #   ...
    #   # PPU MODIFICATION: end
    raise NotImplementedError("TODO")


# --- Triton kernel variant -------------------------------------------------
# @patch("vllm.some_module", "target_kernel", reason=..., affected_versions=...,
#        remove_when=...)
# @triton.heuristics({...})
# @triton.autotune(configs=[...], key=[...])
# @triton.jit
# def target_kernel(...):
#     ...
