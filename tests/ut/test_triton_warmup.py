# SPDX-License-Identifier: Apache-2.0
"""Exercise the installed preflight patch without torch, vLLM or a device."""

from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch


@pytest.fixture
def warmup(modules):
    calls, warnings = [], []
    dtype = object()
    platform = NS(is_ppu=lambda: True, has_device_capability=lambda _: False)
    modules("torch", float8_e4m3fn=dtype)
    modules("vllm.platforms", current_platform=platform)
    modules(
        "vllm.logger", init_logger=lambda _: NS(warning=lambda *a: warnings.append(a))
    )

    class Owner:
        def compile_many(self, keys):
            calls.extend(keys)

        def warmup_inputs(self, key):
            return key

    modules(
        "vllm.model_executor.warmup.jit_warmup_triton_helper",
        VllmTritonJitKernel=Owner,
    )
    patch = load_patch("vllm_sail/patch/enhancement/triton_warmup.py")
    patch.install()
    return NS(
        owner=Owner(),
        cls=Owner,
        patch=patch,
        platform=platform,
        fp8=dtype,
        calls=calls,
        warnings=warnings,
    )


def test_filters_only_unsupported_input_keys(warmup):
    w = warmup
    fp8 = {"x": NS(dtype=w.fp8)}
    direct_dtype = {"dtype": w.fp8}
    supported = {"x": NS(dtype=object())}
    w.owner.compile_many(iter([fp8, supported, direct_dtype]))
    assert w.calls == [supported]
    assert len(w.warnings) == 1
    assert w.warnings[0][1] == 2


def test_all_unsupported_keys_never_reach_original(warmup):
    warmup.owner.compile_many([{"x": NS(dtype=warmup.fp8)}])
    assert warmup.calls == []


@pytest.mark.parametrize("ppu,capable", [(False, False), (True, True)])
def test_other_platforms_bypass_input_inspection(warmup, ppu, capable):
    w = warmup
    w.platform.is_ppu = lambda: ppu
    w.platform.has_device_capability = lambda _: capable
    key = object()  # Has no inputs.values; inspecting it would fail.
    w.owner.compile_many([key])
    assert w.calls == [key]
    assert w.warnings == []


def test_unknown_input_metadata_error_propagates(warmup):
    with pytest.raises(AttributeError):
        warmup.owner.compile_many([object()])


def test_install_is_idempotent(warmup):
    original = warmup.cls.compile_many
    warmup.patch.install()
    assert warmup.cls.compile_many is original
