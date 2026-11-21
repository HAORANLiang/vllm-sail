# SPDX-License-Identifier: Apache-2.0
"""Check the registered platform guard before NVIDIA-only initialization."""

from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch


@pytest.mark.parametrize("ppu", [False, True])
def test_registered_finalize_retains_ordinary_ppu_path(modules, ppu):
    calls = []
    result = object()

    class Model:
        def _init_fused_moe_finalize(self, config):
            if ppu:
                raise ModuleNotFoundError("NVIDIA-only dependency", name="quack")
            calls.append((self, config))
            return result

    modules(
        "vllm.models.minimax_m3.nvidia.model",
        MiniMaxM3Model=Model,
        MiniMaxM3MLP=type("MLP", (), {}),
        _supports_qkv_fused_epilogue=lambda attn: False,
    )
    modules(
        "vllm.platforms", current_platform=NS(is_ppu=lambda: ppu, is_cuda=lambda: True)
    )
    load_patch("vllm_sail/patch/enhancement/models/minimax_m3.py")
    instance, config = Model(), object()
    instance.fuse_final_norm_allreduce = True
    actual = instance._init_fused_moe_finalize(config)
    assert actual is (None if ppu else result)
    assert calls == ([] if ppu else [(instance, config)])
    assert instance.fuse_final_norm_allreduce is True


@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("supported", [False, True])
def test_registered_qkv_capability_delegates_only_off_ppu(
    modules, patch_utils_module, ppu, supported
):
    calls = []

    def upstream(attn):
        if ppu:
            raise ModuleNotFoundError("NVIDIA MXFP8 provider")
        calls.append(attn)
        return supported

    provider = modules(
        "vllm.models.minimax_m3.nvidia.model",
        MiniMaxM3Model=type("Model", (), {"_init_fused_moe_finalize": lambda: None}),
        MiniMaxM3MLP=type("MLP", (), {}),
        _supports_qkv_fused_epilogue=upstream,
    )
    modules("vllm.platforms", current_platform=NS(is_ppu=lambda: ppu))
    load_patch("vllm_sail/patch/enhancement/models/minimax_m3.py")
    attn = object()
    assert provider._supports_qkv_fused_epilogue(attn) is (False if ppu else supported)
    assert calls == ([] if ppu else [attn])
    record = next(
        r
        for r in reversed(patch_utils_module.PATCH_REGISTRY)
        if r.target.endswith("._supports_qkv_fused_epilogue")
    )
    assert (
        getattr(provider._supports_qkv_fused_epilogue, patch_utils_module.PATCH_MARKER)[
            record.target
        ]
        is upstream
    )
    with pytest.raises(RuntimeError, match="already patched"):
        load_patch("vllm_sail/patch/enhancement/models/minimax_m3.py")
