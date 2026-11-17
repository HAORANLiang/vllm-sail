# SPDX-License-Identifier: Apache-2.0
"""vLLM SAIL: inference on T-Head PPU accelerators.

PPU is CUDA-compatible, so this plugin deliberately reuses upstream vLLM's CUDA
code paths rather than vendoring its own attention backends, model runner or
sampler. See ``docs/developer_guide/architecture.md`` for the full rationale.

Two callables are exported as vLLM entry points:

* :func:`register`              -- ``vllm.platform_plugins``
* :func:`register_out_of_tree`  -- ``vllm.general_plugins``

``register_out_of_tree`` owns the complete general-plugin lifecycle,
including model registration. :func:`register_model` remains public for
backward compatibility and focused tests, but is not a separate entry point.
"""

from vllm_sail.version import __version__, __version_tuple__

__all__ = [
    "__version__",
    "__version_tuple__",
    "collect_env",
    "register",
    "register_model",
    "register_out_of_tree",
]

# ---------------------------------------------------------------------------
# Idempotency guards.
#
# vLLM loads `vllm.general_plugins` in every process -- API server, engine core
# subprocess, and each worker -- and may call a hook more than once within one
# process. Our `@patch` decorator raises RuntimeError on double application by
# design, so the hooks below must be individually idempotent rather than relying
# on module-import caching. (vLLM-MetaX omits this guard; vllm-ascend documents
# the same subtlety in its `_ensure_global_patch`.)
# ---------------------------------------------------------------------------
_SHIMS_INSTALLED = False
_PATCHES_APPLIED = False
_REGISTRIES_APPLIED = False
_MODELS_REGISTERED = False


def register() -> str:
    """``vllm.platform_plugins`` hook: return the PPU platform class path.

    Called when ``vllm.platforms.current_platform`` is first resolved. This is
    the *earliest* plugin hook vLLM offers, which is why the import shims that
    must beat upstream's own imports are installed here rather than in
    :func:`register_out_of_tree`.

    Keep this function cheap and free of torch-heavy imports: it runs during
    vLLM's platform resolution, before the platform object exists.
    """
    global _SHIMS_INSTALLED
    if not _SHIMS_INSTALLED:
        # vllm.platforms.cuda unconditionally imports this extension even when
        # vLLM_TARGET_DEVICE=empty deliberately omitted it. PPU registers the
        # actual ops later from PPUPlatform.import_kernels().
        from vllm_sail.native.bootstrap import install as _install_native_shim

        _install_native_shim()

        # Must run before anything imports `vllm.vllm_flash_attn`, whose
        # __init__ raises ImportError when vLLM's own FA extensions are absent
        # (PPU ships its own flash_attn / flash_attn_3 wheels instead).
        from vllm_sail.attention.flash_attn_shim import install as _install_fa_shim

        _install_fa_shim()
        _SHIMS_INSTALLED = True

    return "vllm_sail.platform.PPUPlatform"


def register_out_of_tree() -> None:
    """Install the complete ``vllm.general_plugins`` lifecycle.

    Order matters: the compatibility check runs first so that a version
    mismatch produces a clear error instead of an obscure patch failure. Model
    registration belongs to this same hook so an exact-name plugin allowlist
    cannot enable the PPU platform while filtering out its model overrides.
    """
    global _PATCHES_APPLIED, _REGISTRIES_APPLIED

    from vllm_sail.compat import check_vllm_compatibility

    check_vllm_compatibility()

    if not _PATCHES_APPLIED:
        import vllm_sail.patch

        vllm_sail.patch.install()
        _PATCHES_APPLIED = True

    if not _REGISTRIES_APPLIED:
        import vllm_sail.ops  # noqa: F401
        import vllm_sail.registry

        # Call the explicit lifecycle API. Import-only registration is unsafe:
        # tuned-config patches import a registry child while patch.install() is
        # still running, so the package may already be cached at this point.
        vllm_sail.registry.register()

        # Layer NVTX follows its config flag, independent of the outer profiling env.
        from vllm_sail.profiling.layerwise import install as _install_layerwise

        _install_layerwise()

        # NVTX instrumentation is opt-in: this installs nothing unless
        # VLLM_SAIL_NVTX_PROFILE is set and the optional deps import.
        from vllm_sail.profiling import install as _install_profiling

        _install_profiling()
        _REGISTRIES_APPLIED = True

    register_model()


def register_model() -> None:
    """``vllm.general_plugins`` hook: register PPU model implementations."""
    global _MODELS_REGISTERED
    if _MODELS_REGISTERED:
        return

    from vllm_sail.models import register_model as _register_model

    _register_model()
    _MODELS_REGISTERED = True


def collect_env() -> None:
    """Console entry point for the PPU environment report."""
    from vllm_sail.collect_env import main

    main()
