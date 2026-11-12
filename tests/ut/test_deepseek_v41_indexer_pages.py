# SPDX-License-Identifier: Apache-2.0
"""Exercise the installed V4.1 indexer patch with strict upstream doubles."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from tests.support.modules import load_patch
from tests.support.source import function


@pytest.mark.parametrize("ppu", [True, False])
@pytest.mark.parametrize("ratio", [1, 2])
def test_indexer_publishes_geometry_and_physical_pages(modules, ppu, ratio):
    @dataclass(frozen=True)
    class Spec:
        block_size: int = 256
        tokens_per_state: int = ratio
        page_size_padded: int | None = 33792
        indexer_n_head: int | None = None
        indexer_q_head_dim: int | None = None

    class Cache:
        def get_kv_cache_spec(self, config):
            return Spec()

    class Indexer:
        def __init__(self, cache):
            self.k_cache = cache
            self.n_head = 32
            self.head_dim = 128

    class Backend:
        @staticmethod
        def get_supported_kernel_block_sizes(kv_cache_spec=None):
            return [128]

    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: ppu))
    modules(
        "vllm.models.deepseek_v41.attention",
        DeepseekV4IndexerCache=Cache,
        DeepseekV4Indexer=Indexer,
    )
    modules("vllm.v1.attention.backends.mla.indexer", DeepseekV41IndexerBackend=Backend)

    class Runner:
        def capture_model(self, *, profile_only=False):
            return profile_only

    modules("vllm.v1.worker.gpu.model_runner", GPUModelRunner=Runner)
    module = load_patch("vllm_sail/patch/enhancement/models/deepseek_v41_indexer.py")
    assert Runner.capture_model is module.capture_model
    cache = Cache()
    cache.compress_ratio = ratio
    Indexer(cache)
    spec = cache.get_kv_cache_spec(object())
    if ppu:
        assert spec.block_size == 64 * ratio
        assert spec.page_size_padded is None
        assert (spec.indexer_n_head, spec.indexer_q_head_dim) == (32, 128)
        assert Backend.get_supported_kernel_block_sizes(spec) == [64, 128]
    else:
        assert spec == Spec()
        assert Backend.get_supported_kernel_block_sizes(spec) == [128]


@pytest.mark.parametrize("sizes", [None, [], [1, 64, 72]])
def test_indexer_projection_warmup_covers_capture_sizes(modules, sizes):
    calls = []

    class Tensor:
        def __getitem__(self, item):
            return item.stop

    def zeros(shape, *, dtype, device):
        calls.append(("allocate", shape, dtype, device))
        return Tensor()

    class Indexer:
        wq_b = SimpleNamespace(
            input_size=1280, params_dtype="bf16", weight=SimpleNamespace(device="ppu")
        )

        def _wq_b_proj(self, inputs):
            calls.append(("project", inputs))

    modules("torch", zeros=zeros)
    warmup = function(
        "vllm_sail/patch/enhancement/models/deepseek_v41_indexer.py",
        "warmup_indexer_projections",
        {"attention": SimpleNamespace(DeepseekV4Indexer=Indexer)},
    )
    runner = SimpleNamespace(
        model=SimpleNamespace(modules=lambda: [object(), Indexer()]),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=sizes)
        ),
    )
    warmup(runner)
    assert calls == (
        [("allocate", (72, 1280), "bf16", "ppu")]
        + [("project", size) for size in sizes]
        if sizes
        else []
    )


@pytest.mark.parametrize("ppu", [True, False])
@pytest.mark.parametrize("profile_only", [True, False])
def test_indexer_warmup_precedes_profile_and_real_capture(ppu, profile_only):
    calls = []
    runner = object()

    def capture(actual, *, profile_only):
        assert actual is runner
        calls.append(("capture", profile_only))
        return 123

    forward = function(
        "vllm_sail/patch/enhancement/models/deepseek_v41_indexer.py",
        "capture_model",
        {
            "current_platform": SimpleNamespace(is_ppu=lambda: ppu),
            "warmup_indexer_projections": lambda actual: calls.append(("warm", actual)),
            "_capture_model": capture,
        },
    )
    assert forward(runner, profile_only=profile_only) == 123
    assert calls == ([("warm", runner)] if ppu else []) + [("capture", profile_only)]
