# SPDX-License-Identifier: Apache-2.0
"""Exercise V4.1 cache sizing and upstream kernel selection without torch."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from tests.support.source import function
from tests.ut import test_ppu_kernel_capabilities as harness

modules = harness.modules
PATH = "vllm_sail/patch/enhancement/models/deepseek_v41_indexer.py"


@dataclass(frozen=True)
class Spec:
    block_size: int
    num_kv_heads: int
    head_size: int
    dtype: object
    tokens_per_state: int
    alignment: int
    block_stride_alignment: int | None = None
    page_size_padded: int | None = None
    indexer_n_head: int | None = None
    indexer_q_head_dim: int | None = None

    def __post_init__(self):
        size = self.block_size // self.tokens_per_state * self.head_size
        if self.page_size_padded is None:
            object.__setattr__(
                self,
                "page_size_padded",
                -(-size // self.alignment) * self.alignment,
            )


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu,capability", [(True, 89), (False, 90), (False, 100)])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("manager_block", [128, 256])
def test_indexer_cache_pages(
    modules, upstream_source_root, ppu, capability, ratio, manager_block
):
    platform = SimpleNamespace(
        is_ppu=lambda: ppu,
        is_device_capability_family=lambda value: capability == value,
    )
    modules("vllm.platforms", current_platform=platform)
    indexer_source = upstream_source_root / "vllm/v1/attention/backends/mla/indexer.py"
    kernel_sizes = function(
        indexer_source,
        "DeepseekV41IndexerBackend.get_supported_kernel_block_sizes",
        {"current_platform": platform},
    )
    backend = type(
        "Backend", (), {"get_supported_kernel_block_sizes": staticmethod(kernel_sizes)}
    )
    modules("vllm.v1.attention.backends.mla.indexer", DeepseekV41IndexerBackend=backend)
    cache_spec = function(
        upstream_source_root / "vllm/models/deepseek_v41/attention.py",
        "DeepseekV4IndexerCache.get_kv_cache_spec",
        {
            "MLAAttentionSpec": Spec,
            "_use_v41_mxfp8_kv_record": lambda: capability == 100,
        },
    )
    cache_cls = type("Cache", (), {"get_kv_cache_spec": cache_spec})
    modules(
        "vllm.models.deepseek_v41.attention",
        DeepseekV4IndexerCache=cache_cls,
        DeepseekV4Indexer=type("Indexer", (), {"__init__": lambda self: None}),
    )
    cache = cache_cls()
    cache.cache_config = SimpleNamespace(block_size=manager_block)
    cache.compress_ratio, cache.head_dim, cache.dtype = ratio, 132, "uint8"
    cache.sparse_logits = False
    config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="fp8_ds_mla"))
    before = cache.get_kv_cache_spec(config)
    harness.load_patch(PATH)
    spec = cache.get_kv_cache_spec(config)
    assert spec.indexer_n_head is None and spec.indexer_q_head_dim is None
    select = function(
        upstream_source_root / "vllm/v1/worker/utils.py",
        "select_common_block_size",
        {"MultipleOf": type("MultipleOf", (), {})},
    )
    selected = select(spec.block_size, [backend])
    assert backend.get_supported_kernel_block_sizes(kv_cache_spec=spec) == (
        [64, 128] if ppu else [64 if capability == 90 else 128]
    )
    if ppu:
        assert spec.block_size == selected == 64 * ratio
        assert spec.block_size // spec.tokens_per_state == 64
        # Recompute alignment after changing the block size; never retain the
        # old 128/256-row page padding. No virtual splitting is needed.
        assert spec.page_size_padded == 8640
    else:
        assert spec == before
        assert selected == (64 if capability == 90 else 128)
    assert cache.cache_config.block_size == manager_block


@pytest.mark.parametrize("ppu", [True, False])
@pytest.mark.parametrize("n_head,head_dim", [(32, 128), (8, 64)])
@pytest.mark.parametrize("ratio", [1, 2])
def test_indexer_init_publishes_query_geometry_only_on_ppu(
    modules, ppu, n_head, head_dim, ratio
):
    calls = []
    before = Spec(256, 1, 132, "uint8", ratio, 576)

    class Cache:
        def get_kv_cache_spec(self, vllm_config):
            calls.append(("spec", self, vllm_config))
            return before

    class Indexer:
        def __init__(self, config, token, *, k_cache, owns_k):
            calls.append(("init", self, config, token, k_cache, owns_k))
            # The wrapper must delegate before accessing the geometry or cache.
            assert not hasattr(self, "k_cache")
            self.n_head = config.index_n_heads
            self.head_dim = config.index_head_dim
            self.k_cache = k_cache

    original_init = Indexer.__init__
    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: ppu))
    modules(
        "vllm.models.deepseek_v41.attention",
        DeepseekV4Indexer=Indexer,
        DeepseekV4IndexerCache=Cache,
    )
    modules(
        "vllm.v1.attention.backends.mla.indexer",
        DeepseekV41IndexerBackend=type(
            "Backend",
            (),
            {"get_supported_kernel_block_sizes": staticmethod(lambda **kw: [128])},
        ),
    )
    patched = harness.load_patch(PATH)
    cache = Cache()
    cache.compress_ratio = ratio
    cache.head_dim = 132  # Packed K record width is not the query head dimension.
    cache.n_head, cache.q_head_dim = 4, 16
    cache_state = vars(cache).copy()
    config = SimpleNamespace(index_n_heads=n_head, index_head_dim=head_dim)
    token, vllm_config = object(), object()
    indexer = Indexer(config, token, k_cache=cache, owns_k=False)
    spec = cache.get_kv_cache_spec(vllm_config)
    assert calls == [
        ("init", indexer, config, token, cache, False),
        ("spec", cache, vllm_config),
    ]
    assert indexer.k_cache is cache
    if ppu:
        assert vars(cache) == {
            **cache_state,
            "n_head": n_head,
            "q_head_dim": head_dim,
        }
        assert spec == replace(
            before,
            block_size=64 * ratio,
            page_size_padded=None,
            indexer_n_head=n_head,
            indexer_q_head_dim=head_dim,
        )
    else:
        assert vars(cache) == cache_state
        assert spec is before
    assert before.block_size == 256
    assert before.indexer_n_head is None and before.indexer_q_head_dim is None
    marker = modules("vllm_sail.patch.utils").PATCH_MARKER
    assert (
        getattr(Indexer.__init__, marker)[
            "vllm.models.deepseek_v41.attention.DeepseekV4Indexer.__init__"
        ]
        is original_init
    )
    assert Indexer.__init__ is patched.indexer_init
    with pytest.raises(
        RuntimeError, match="DeepseekV4Indexer.__init__.*already patched"
    ):
        harness.load_patch(PATH)
    assert Indexer.__init__ is patched.indexer_init
