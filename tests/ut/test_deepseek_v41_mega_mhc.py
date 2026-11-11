# SPDX-License-Identifier: Apache-2.0
"""Exercise optional MegaMHC admission without torch or vLLM."""

from types import SimpleNamespace

import pytest

from tests.support.modules import load_patch


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
    module = load_patch("vllm_sail/models/deepseek_v41/mega_mhc.py")
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


@pytest.mark.parametrize("preloaded", [False, True])
def test_patch_dispatch_preserves_cuda_and_rebinds_preloaded_consumer(
    modules, preloaded
):
    from vllm_sail.patch.utils import PATCH_MARKER

    provider_name = "vllm.models.deepseek_v41.nvidia.ops.mega_mhc"
    calls = []
    originals = {}

    def original(name):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name

        originals[name] = call
        return call

    provider = modules(
        provider_name,
        **{
            name: original(name)
            for name in ("is_mega_mhc_supported", "can_use_mega_mhc", "mega_mhc")
        },
    )
    consumer = (
        modules(
            "vllm.models.deepseek_v41.nvidia.ops.mhc",
            can_use_mega_mhc=provider.can_use_mega_mhc,
        )
        if preloaded
        else None
    )
    state = SimpleNamespace(ppu=True)
    modules(
        "vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: state.ppu)
    )
    sail = modules(
        "vllm_sail.models.deepseek_v41.mega_mhc",
        is_supported=lambda *args: "ppu-supported",
        can_use=lambda *args: "ppu-admitted",
        mega_mhc_kernel=lambda: lambda *args, **kwargs: (args, kwargs),
    )
    load_patch("vllm_sail/patch/enhancement/models/deepseek_v41_mhc.py")
    for name, before in originals.items():
        assert (
            getattr(getattr(provider, name), PATCH_MARKER)[f"{provider_name}.{name}"]
            is before
        )
    if consumer:
        assert consumer.can_use_mega_mhc is provider.can_use_mega_mhc
    args = tuple(object() for _ in range(5))
    assert provider.is_mega_mhc_supported(5120, 4) == "ppu-supported"
    assert provider.can_use_mega_mhc(*args) == "ppu-admitted"
    assert provider.mega_mhc(x=args[0]) == ((), {"x": args[0]})
    assert calls == []
    state.ppu = False
    assert provider.is_mega_mhc_supported(5120, 4) == "is_mega_mhc_supported"
    assert provider.can_use_mega_mhc(*args) == "can_use_mega_mhc"
    assert provider.mega_mhc(x=args[0]) == "mega_mhc"
    assert calls[-1] == ("mega_mhc", (), {"x": args[0]})
    state.ppu = True

    def failed_launch(**kwargs):
        raise RuntimeError("launch failed")

    sail.mega_mhc_kernel = lambda: failed_launch
    with pytest.raises(RuntimeError, match="launch failed"):
        provider.mega_mhc(x=args[0])
    assert len(calls) == 3
    sail.mega_mhc_kernel = lambda: None
    with pytest.raises(RuntimeError, match="unavailable after capability selection"):
        provider.mega_mhc(x=args[0])
    with pytest.raises(RuntimeError, match="already patched"):
        load_patch("vllm_sail/patch/enhancement/models/deepseek_v41_mhc.py")


@pytest.mark.upstream_source
def test_upstream_shifted_boundary_forwards_previous_mix_and_output_storage(
    upstream_source_root,
):
    from tests.support.source import function

    class Buffer:
        def __init__(self, shape):
            self.shape = shape

        def new_empty(self, *shape, dtype):
            return Buffer(shape)

        def unsqueeze(self, dim):
            return self

        def squeeze(self, dim):
            return self

    launches = []
    run = function(
        upstream_source_root / "vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py",
        "mhc_shifted_post_pre_deep_gemm",
        {
            "torch": SimpleNamespace(
                empty_like=lambda t: Buffer(t.shape), float32="fp32", bfloat16="bf16"
            ),
            "mega_mhc": lambda **kwargs: launches.append(kwargs),
        },
    )
    x, residual, previous = Buffer((7, 5120)), Buffer((7, 4, 5120)), Buffer((7, 4))
    result = run(
        x,
        residual,
        previous,
        Buffer((7, 4, 1)),
        Buffer((7, 4, 4)),
        object(),
        object(),
        object(),
        1e-6,
        1e-6,
        2.0,
        1e-6,
        20,
        object(),
        1e-6,
    )
    assert len(launches) == 1
    launch = launches[0]
    assert launch["shifted_prev_mix"] is previous
    assert launch["x"] is x and launch["residual"] is residual
    assert result == tuple(
        launch[name]
        for name in (
            "new_residual",
            "new_post_mix",
            "new_comb_res_mix",
            "y_bf16",
            "new_prev_mix",
        )
    )
