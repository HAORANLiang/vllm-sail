# SPDX-License-Identifier: Apache-2.0
"""Point vLLM's DeepGEMM helper bindings at the PPU DeepGEMM wrapper.

Three quantization modules import their DeepGEMM helpers from
``vllm.utils.deep_gemm`` at module scope. The in-tree fork turns each of those
into an ``if current_platform.is_ppu():`` import switch, selecting
``vllm.utils.ppu_deep_gemm`` instead:

* ``quantization/utils/fp8_utils.py`` — ``get_tma_aligned_size``,
  ``is_deep_gemm_e8m0_used``, ``transform_sf_into_required_layout``
* ``quantization/fp8.py`` — ``is_deep_gemm_supported``
* ``quantization/input_quant_fp8.py`` — ``DeepGemmQuantScaleFMT``,
  ``is_deep_gemm_e8m0_used``, ``is_deep_gemm_supported``

PPU's DeepGEMM has different scale-factor layout requirements and TMA
alignment, so using CUDA's helpers produces wrong scale layouts rather than a
clean error.

A plugin cannot rewrite an already-executed module-scope import, so instead we
rebind the names on each importing module itself. They are function / class
objects, and the PPU implementations are necessarily different objects from
CUDA's, so they go through :func:`~vllm_sail.patch.utils.patch` (``patch_value``
is for data constants and raises on a differing existing value).

The rebinding is unconditional at install time — this plugin only ever loads on a
PPU host, and it is the PPU platform plugin that triggers installation.
"""

from __future__ import annotations

from vllm.logger import init_logger

from vllm_sail.patch.utils import patch

logger = init_logger(__name__)

_AFFECTED = ">=0.33.0,<0.34.0"
_REASON = (
    "PPU's DeepGEMM requires different TMA alignment and scale-factor layout "
    "transforms than CUDA's. {module} binds these helpers from "
    "vllm.utils.deep_gemm at module scope; on PPU they must come from the PPU "
    "DeepGEMM wrapper instead, or FP8 GEMMs get wrong scale layouts or a "
    "CUDA-only support probe."
)
_REMOVE_WHEN = (
    "vllm.utils.deep_gemm dispatches on current_platform internally, or upstream "
    "accepts a DeepGEMM backend registry."
)

#: Module -> names that module binds from a DeepGEMM wrapper at import time.
#: Kept explicit so that an upstream addition to one of those import lists
#: shows up as a missing-attribute error here rather than as a
#: silently-CUDA-flavoured helper on PPU.
_REDIRECTED = {
    "vllm.model_executor.layers.quantization.utils.fp8_utils": (
        "get_tma_aligned_size",
        "is_deep_gemm_e8m0_used",
        "transform_sf_into_required_layout",
    ),
    "vllm.model_executor.layers.quantization.fp8": (
        "is_deep_gemm_supported",
    ),
    "vllm.model_executor.layers.quantization.input_quant_fp8": (
        "DeepGemmQuantScaleFMT",
        "is_deep_gemm_e8m0_used",
        "is_deep_gemm_supported",
    ),
}


def _install() -> None:
    import vllm_sail.utils.deep_gemm as ppu_deep_gemm

    needed = sorted({name for names in _REDIRECTED.values() for name in names})
    missing = [name for name in needed if not hasattr(ppu_deep_gemm, name)]
    if missing:
        raise AttributeError(
            f"vllm_sail.utils.deep_gemm is missing {', '.join(missing)}, which "
            "vLLM's quantization modules import from a DeepGEMM wrapper. The "
            "PPU DeepGEMM wrapper must stay API-compatible with "
            "vllm.utils.deep_gemm."
        )

    for module_name, names in _REDIRECTED.items():
        for name in names:
            patch(
                module_name,
                name,
                reason=_REASON.format(module=module_name.rsplit(".", 1)[-1]),
                affected_versions=_AFFECTED,
                remove_when=_REMOVE_WHEN,
            )(getattr(ppu_deep_gemm, name))

    logger.debug(
        "Redirected DeepGEMM helpers in %s to vllm_sail.utils.deep_gemm",
        ", ".join(sorted(_REDIRECTED)),
    )


_install()
