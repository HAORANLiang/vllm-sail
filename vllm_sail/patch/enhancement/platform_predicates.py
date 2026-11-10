# SPDX-License-Identifier: Apache-2.0
"""Add ``Platform.is_ppu()`` to vLLM.

This is the keystone patch of the whole plugin. The in-tree PPU fork added a
``PlatformEnum.PPU`` member and an ``is_ppu()`` predicate to
``vllm/platforms/interface.py``; a plugin cannot add an enum member, but it can
add the predicate.

Adding it — rather than requiring PPU code to ask something else — means all 101
``current_platform.is_ppu()`` call sites in the fork port over verbatim, and
non-PPU platforms keep answering ``False``.

``PPUPlatform`` declares ``_enum = PlatformEnum.CUDA`` (see
``vllm_sail/platform.py``), so PPU cannot be identified from ``_enum``. The
discriminator is ``device_name == "ppu"``, which ``PPUPlatform`` sets as a class
attribute and no other platform uses.
"""

from __future__ import annotations

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.33.0,<0.34.0"


@patch(
    "vllm.platforms.interface",
    "Platform.is_ppu",
    allow_missing=True,
    reason=(
        "vLLM has no PlatformEnum.PPU member and no is_ppu() predicate. PPU code "
        "throughout this plugin (and the ~101 call sites inherited from the "
        "in-tree PPU fork) branches on current_platform.is_ppu(), so the "
        "predicate must exist on the base Platform class for every platform, "
        "returning False for non-PPU platforms."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "vLLM gains a first-class way for an out-of-tree platform to declare a "
        "vendor identity that is queryable without an enum member (e.g. a "
        "`Platform.vendor` field), or upstream accepts PlatformEnum.PPU."
    ),
)
def is_ppu(self) -> bool:
    """Whether the current platform is a PPU accelerator.

    Keyed on ``device_name`` rather than ``_enum`` because ``PPUPlatform``
    intentionally declares ``PlatformEnum.CUDA`` for CUDA-path compatibility.
    """
    return getattr(self, "device_name", None) == "ppu"


@patch(
    "vllm.platforms.interface",
    "Platform.is_sleep_mode_available",
    reason=(
        "Upstream gates sleep mode on `self._enum in (CUDA, ROCM, XPU)`. PPU "
        "declares PlatformEnum.CUDA so it already passes, but the in-tree fork "
        "listed PPU explicitly and we keep the behaviour pinned here so that an "
        "upstream change to the gate (e.g. to a capability probe) surfaces as a "
        "patch-drift failure rather than silently disabling sleep mode on PPU."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream replaces the enum allow-list with a capability query that PPU "
        "answers correctly by inheritance."
    ),
)
def is_sleep_mode_available(self) -> bool:
    from vllm.platforms.interface import PlatformEnum

    return self._enum in (
        PlatformEnum.CUDA,
        PlatformEnum.ROCM,
        PlatformEnum.XPU,
    ) or self.is_ppu()
