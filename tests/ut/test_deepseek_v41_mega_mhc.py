# SPDX-License-Identifier: Apache-2.0
"""Exercise optional MegaMHC admission without torch or vLLM."""

from types import SimpleNamespace

import pytest

from tests.ut import test_ppu_kernel_capabilities as harness

modules = harness.modules


class Tensor:
    def __init__(self, shape, dtype="bf16", *, aligned=True, contiguous=True):
        self.shape, self.dtype, self.ndim = shape, dtype, len(shape)
        self.device, self.is_cuda = "cuda:0", True
        self.aligned, self.contiguous = aligned, contiguous

    def data_ptr(self):
        return 32 if self.aligned else 33

    def is_contiguous(self):
        return self.contiguous


@pytest.fixture
def admission(modules):
    state = SimpleNamespace(enabled=True, ppu=True, cap=True)
    modules("torch", bfloat16="bf16", float32="fp32")
    modules(
        "vllm.logger",
        init_logger=lambda name: SimpleNamespace(warning_once=lambda *a: None),
    )
    modules(
        "vllm.platforms",
        current_platform=SimpleNamespace(
            is_ppu=lambda: state.ppu,
            has_device_capability=lambda cap: state.cap,
        ),
    )
    settings = modules("vllm_sail.envs", VLLM_SAIL_DEEPSEEK_V41_MEGA_MHC=True)
    modules("vllm_sail.utils.deep_gemm", is_deep_gemm_supported=lambda: True)
    modules("deep_gemm", mega_mhc=lambda **kwargs: None)
    module = harness.load_patch("vllm_sail/models/deepseek_v41/mega_mhc.py")
    args = [
        Tensor((7, 5120)),
        Tensor((7, 4, 5120)),
        Tensor((7, 4), "fp32"),
        Tensor((5120,)),
    ]
    return module, state, settings, args


def test_shifted_boundary_accepts_supported_bf16_contract(admission):
    module, _, _, args = admission
    assert module.can_use(*args, False)


@pytest.mark.parametrize(
    "reason",
    ["disabled", "old_ppu", "cuda", "aux", "no_pre", "stride", "alignment", "hidden"],
)
def test_shifted_boundary_falls_back_before_launch(admission, reason):
    module, state, settings, args = admission
    if reason == "disabled":
        settings.VLLM_SAIL_DEEPSEEK_V41_MEGA_MHC = False
    elif reason == "old_ppu":
        state.cap = False
    elif reason == "cuda":
        state.ppu = False
    elif reason == "no_pre":
        args[2] = None
    elif reason == "stride":
        args[0].contiguous = False
    elif reason == "alignment":
        args[0].aligned = False
    elif reason == "hidden":
        args[0] = Tensor((7, 4096))
    assert not module.can_use(*args, reason == "aux")
