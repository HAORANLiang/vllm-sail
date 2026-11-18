# SPDX-License-Identifier: Apache-2.0
"""Keep the MiniMax M3 ops package importable without quack on PPU.

``nvidia/ops/__init__.py`` eagerly imports the CuTeDSL index-decode-score
kernel, which needs ``quack`` (not installed, SM100-only). Even the plain
Triton ``norm_mxfp8`` path dies because importing any submodule runs that
package ``__init__``. Preload the kernel module once under a temporary quack
stub, then rebind its ``make_fake_tensor`` to an explicit error: on PPU the
SAIL MSA gate never selects this kernel, and any accidental call now fails
loudly instead of at import time.
"""

from __future__ import annotations

import importlib
import sys
import types

from vllm.platforms import current_platform

from vllm_sail.patch.utils import PATCH_REGISTRY, PatchRecord

_AFFECTED = ">=0.34.0,<0.35.0"
_TARGET = "vllm.models.minimax_m3.nvidia.ops.index_decode_score.make_fake_tensor"
_REASON = (
    "The MiniMax M3 ops package eagerly imports a CuTeDSL kernel that needs "
    "quack; without it even the Triton norm path fails at import time on PPU."
)
_REMOVE_WHEN = (
    "Upstream makes the CuTeDSL export lazy or capability-gated, or PPU "
    "gains a quack/CuTeDSL stack."
)

_OPS = "vllm.models.minimax_m3.nvidia.ops"
_KERNEL = f"{_OPS}.index_decode_score"


def _unsupported_make_fake_tensor(*args, **kwargs):
    raise NotImplementedError(
        "MiniMax M3 CuTeDSL index-decode-score kernel is not supported on PPU"
    )


def install() -> None:
    if not current_platform.is_ppu():
        return
    if _KERNEL in sys.modules:
        return
    # Temporary stand-in so the kernel module's top-level quack import passes.
    quack_stub = types.ModuleType("quack")
    compile_utils_stub = types.ModuleType("quack.compile_utils")
    compile_utils_stub.make_fake_tensor = _unsupported_make_fake_tensor
    quack_stub.compile_utils = compile_utils_stub
    sys.modules["quack"] = quack_stub
    sys.modules["quack.compile_utils"] = compile_utils_stub
    try:
        kernel = importlib.import_module(_KERNEL)
    finally:
        sys.modules.pop("quack", None)
        sys.modules.pop("quack.compile_utils", None)
    # The module bound the stub by value; make any real use fail loudly.
    kernel.make_fake_tensor = _unsupported_make_fake_tensor
    PATCH_REGISTRY.append(
        PatchRecord(
            target=_TARGET,
            reason=_REASON,
            affected_versions=_AFFECTED,
            remove_when=_REMOVE_WHEN,
            kind="module",
            was_missing=False,
            original_source=None,
        )
    )