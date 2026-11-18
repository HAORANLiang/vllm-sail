# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the packed kernel-block split patch.

The patch relaxes ``AttentionGroup.map_kv_cache_to_kernel_blocks`` for a
backend that declares ``writes_kv_cache_from_block_table``: a packed cache
storing one state per token may still be split when its strides make the
kernel blocks addressable. These tests pin the arithmetic against upstream's
verbatim body (``kernel_block_stride`` / ``num_pages`` / ``as_strided``) and
the three delegate paths: non-PPU platforms, a non-callable contract, and an
unaddressable cache.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.requires_runtime

torch = pytest.importorskip("torch", reason="requires torch")
pytest.importorskip("vllm", reason="requires vLLM")

from vllm.platforms import current_platform

import vllm_sail  # noqa: F401  (installs the plugin hooks)

from vllm_sail.patch.enhancement.attention import packed_token_cache_split as plugin


class _FakeSpec:
    def __init__(self, kernel_rows: int, block_size: int, tokens_per_state: int):
        self._kernel_rows = kernel_rows
        self.block_size = block_size
        self.tokens_per_state = tokens_per_state

    def get_num_kernel_states(self, kernel_block_size: int) -> int:
        assert kernel_block_size == 64
        return self._kernel_rows


class _FakeBackend:
    def __init__(self, contract):
        self._contract = contract

    def get_name(self) -> str:
        return "FakeKimiIndexer"

    def writes_kv_cache_from_block_table(self) -> bool:
        return self._contract


class _FakeGroup:
    def __init__(self, spec, contract, kernel_block_size=64):
        self.kv_cache_spec = spec
        self.backend = _FakeBackend(contract)
        self.kernel_block_size = kernel_block_size
        self.kernel_block_stride = 0


def _packed_cache(
    manager_blocks: int,
    kernel_blocks_per_manager_block: int,
    kernel_rows: int,
    head_dim: int = 8,
) -> torch.Tensor:
    """A [B, 1, k*kernel_rows, D] cache whose manager-block stride is a whole
    number of kernel pages, like the real packed layout (which additionally
    leaves other layers' pages between blocks; the arithmetic only needs the
    divisibility the split checks)."""
    k = kernel_blocks_per_manager_block
    return torch.empty(
        manager_blocks, 1, k * kernel_rows, head_dim, dtype=torch.bfloat16
    )


@pytest.fixture
def ppu_platform(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(type(current_platform), "is_ppu", lambda self: True)
    return None


def test_split_view_matches_the_arithmetic(ppu_platform) -> None:
    spec = _FakeSpec(kernel_rows=64, block_size=256, tokens_per_state=1)
    group = _FakeGroup(spec, contract=True)
    cache = _packed_cache(
        manager_blocks=3, kernel_blocks_per_manager_block=4, kernel_rows=64
    )
    view = plugin.map_kv_cache_to_kernel_blocks(group, cache)
    # Continuous cache: page_stride=512, kernel_block_stride=4, num_pages=12.
    assert view.shape == (12, 1, 64, 8)
    assert view.stride() == (512, 2048, 8, 1)
    assert group.kernel_block_stride == 4
    # Kernel block j of block b is page b*kernel_block_stride + j: dim0 of the
    # view is the flat page index, so page 2*4+3=11 holds block (b=2, j=3) at
    # 11*512 elements, plus the in-page (row, col) offset. data_ptr is bytes.
    assert view.data_ptr() == cache.data_ptr()
    assert (
        view[11, 0, 7, 3].data_ptr()
        == cache.data_ptr() + (11 * 512 + 7 * 8 + 3) * 2
    )


def test_short_cache_is_returned_untouched(ppu_platform) -> None:
    spec = _FakeSpec(kernel_rows=64, block_size=64, tokens_per_state=1)
    group = _FakeGroup(spec, contract=True)
    cache = torch.empty(4, 1, 64, 8, dtype=torch.bfloat16)
    assert plugin.map_kv_cache_to_kernel_blocks(group, cache) is cache


def test_non_ppu_keeps_upstreams_guard(monkeypatch) -> None:
    monkeypatch.setattr(type(current_platform), "is_ppu", lambda self: False)
    spec = _FakeSpec(kernel_rows=64, block_size=256, tokens_per_state=1)
    group = _FakeGroup(spec, contract=True)
    cache = _packed_cache(3, 4, 64)
    with pytest.raises(ValueError, match="cannot be split into"):
        plugin.map_kv_cache_to_kernel_blocks(group, cache)


def test_non_callable_contract_keeps_upstreams_guard(ppu_platform) -> None:
    class _BoolBackend(_FakeBackend):
        writes_kv_cache_from_block_table = True  # a plain bool, not a method

    spec = _FakeSpec(kernel_rows=64, block_size=256, tokens_per_state=1)
    group = _FakeGroup(spec, contract=True)
    group.backend = _BoolBackend(True)
    cache = _packed_cache(3, 4, 64)
    with pytest.raises(ValueError, match="cannot be split into"):
        plugin.map_kv_cache_to_kernel_blocks(group, cache)


def test_unsplittable_cache_warns_and_keeps_upstreams_error(
    ppu_platform, caplog
) -> None:
    spec = _FakeSpec(kernel_rows=64, block_size=256, tokens_per_state=1)
    group = _FakeGroup(spec, contract=True)
    # stride(0) is not a multiple of the page stride: the layout is not the
    # addressable one, so the split must refuse instead of mis-addressing.
    cache = torch.empty(3, 2, 256, 8, dtype=torch.bfloat16)
    with caplog.at_level("WARNING", logger=plugin.logger.name):
        with pytest.raises(ValueError, match="cannot be split into"):
            plugin.map_kv_cache_to_kernel_blocks(group, cache)
    assert any("cannot be split into" in r.getMessage() for r in caplog.records)
