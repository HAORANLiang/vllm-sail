# SPDX-License-Identifier: Apache-2.0
"""``VLLM_SAIL_KPOOL_DECODE_LOGITS`` API contract for the PPU kpool indexer.

The kpool decode path now selects the paged int8 logits kernel unless the
``non_paged`` dense fallback is explicitly requested. These tests pin that
choice API together with the dense-pool guard the fallback still relies on.
CPU-only by design: no torch, no vLLM, no accelerator, in line with the
``tests/ut`` merge gate.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_sail import envs

KPOOL_SOURCE = Path(__file__).parents[2] / "vllm_sail/attention/ops/mla_sparse_kpool.py"
SAIL_NAME = "VLLM_SAIL_KPOOL_DECODE_LOGITS"
PPU_NAME = "VLLM_PPU_KPOOL_DECODE_LOGITS"
BACKENDS = ("non_paged", "paged")
DEFAULT_BACKEND = "paged"
DENSE_POOL_LIMIT = 1 << 21


@pytest.fixture(autouse=True)
def clean_backend_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (SAIL_NAME, PPU_NAME, "VLLM_SAIL_KPOOL_MAX_DENSE_POOLS"):
        monkeypatch.delenv(name, raising=False)


def test_default_is_paged_without_any_environment() -> None:
    assert envs.VLLM_SAIL_KPOOL_DECODE_LOGITS == DEFAULT_BACKEND
    assert envs.VLLM_PPU_KPOOL_DECODE_LOGITS == DEFAULT_BACKEND
    assert envs.snapshot()[SAIL_NAME] == DEFAULT_BACKEND


def test_empty_environment_falls_back_to_paged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SAIL_NAME, "")
    assert envs.VLLM_SAIL_KPOOL_DECODE_LOGITS == DEFAULT_BACKEND


@pytest.mark.parametrize("prefix", ["VLLM_SAIL", "VLLM_PPU"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_whitelist_accepts_both_backends(
    monkeypatch: pytest.MonkeyPatch, prefix: str, backend: str
) -> None:
    monkeypatch.setenv(f"{prefix}_KPOOL_DECODE_LOGITS", f" {backend.upper()} ")
    assert envs.VLLM_SAIL_KPOOL_DECODE_LOGITS == backend
    assert envs.VLLM_PPU_KPOOL_DECODE_LOGITS == backend


@pytest.mark.parametrize("prefix", ["VLLM_SAIL", "VLLM_PPU"])
def test_invalid_value_raises_value_error(
    monkeypatch: pytest.MonkeyPatch, prefix: str
) -> None:
    monkeypatch.setenv(f"{prefix}_KPOOL_DECODE_LOGITS", "paged_fp8")
    with pytest.raises(ValueError, match="Expected one of non_paged, paged"):
        _ = envs.VLLM_SAIL_KPOOL_DECODE_LOGITS


def test_explicit_non_paged_opt_in_still_selects_the_dense_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SAIL_NAME, "non_paged")
    assert envs.VLLM_SAIL_KPOOL_DECODE_LOGITS == "non_paged"
    monkeypatch.setenv(PPU_NAME, "PAGED")
    monkeypatch.setenv(SAIL_NAME, "non_paged")
    assert envs.VLLM_SAIL_KPOOL_DECODE_LOGITS == "non_paged"


def test_dispatch_treats_non_paged_as_the_only_opt_in_branch() -> None:
    """Only the explicit ``non_paged`` value may reach the guard-bound path."""
    source = KPOOL_SOURCE.read_text(encoding="utf-8")
    assert re.search(
        rf"decode_backend = ppu_envs\.{SAIL_NAME}\n\s*"
        r'if decode_backend == "non_paged":\n\s*'
        r"logits, row_ks, row_ke = _decode_logits_non_paged\(",
        source,
    )
    # The dense guard is read inside the fallback only, so the default paged
    # branch can never be blocked by it.
    assert source.count("ppu_envs.VLLM_SAIL_KPOOL_MAX_DENSE_POOLS") == 1


def _load_kpool_function(name: str) -> dict:
    """Execute one original function body without device imports or patches."""
    tree = ast.parse(
        KPOOL_SOURCE.read_text(encoding="utf-8"), filename=str(KPOOL_SOURCE)
    )
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(functions) == 1, f"{name} not found exactly once"
    functions[0].decorator_list = []
    future = ast.parse("from __future__ import annotations").body
    module = ast.Module(body=[*future, *functions], type_ignores=[])
    namespace: dict = {}
    exec(compile(module, str(KPOOL_SOURCE), "exec"), namespace)
    return namespace


class _FakeValue:
    """Stand-in for a tensor: the guard only moves values around."""

    def reshape(self, *args):
        return self

    def view(self, *args):
        return self

    def squeeze(self, *args):
        return self

    def __getitem__(self, key):
        return self


def _fallback_call(batch_size: int, max_seq_len: int, kpool: int) -> dict:
    """Run ``_decode_logits_non_paged`` with fakes and report the pool count."""
    calls: dict = {}
    namespace = _load_kpool_function("_decode_logits_non_paged")
    namespace.update(
        ppu_envs=envs,
        torch=SimpleNamespace(float32="float32", int8="int8", uint8="uint8"),
        _decode_logits_layout=lambda *args: (
            _FakeValue(),
            _FakeValue(),
            _FakeValue(),
        ),
        _gather_workspace_shapes=lambda total_pools, head_dim: (
            calls.update(max_pools=total_pools)
            or ((total_pools, head_dim), (total_pools, 4))
        ),
        current_workspace_manager=lambda: SimpleNamespace(
            get_simultaneous=lambda *args: (_FakeValue(), _FakeValue())
        ),
        ops=SimpleNamespace(
            cp_gather_indexer_k_quant_cache=lambda *args: calls.update(gathered=True)
        ),
        int8_mqa_logits=lambda *args, **kwargs: (
            calls.update(clean_logits=kwargs["clean_logits"]) or _FakeValue()
        ),
    )
    namespace["_decode_logits_non_paged"](
        _FakeValue(),  # kv_cache
        _FakeValue(),  # padded_q
        _FakeValue(),  # padded_weights
        _FakeValue(),  # seq_lens
        _FakeValue(),  # decode_lens
        _FakeValue(),  # block_table
        batch_size,
        2,  # next_n
        batch_size * 2,
        max_seq_len,
        kpool,
        128,  # head_dim
    )
    return calls


def test_dense_guard_keeps_the_ceiling_division_formula() -> None:
    # ceil(8 / 4) = 2 pools per row, two rows.
    assert _fallback_call(2, 8, 4)["max_pools"] == 4
    # The division rounds up: ceil(9 / 4) = 3 pools per row.
    assert _fallback_call(2, 9, 4)["max_pools"] == 6
    # The fallback keeps calling the logits kernel with the untrimmed logits.
    assert _fallback_call(2, 8, 4)["clean_logits"] is False


def test_dense_guard_threshold_is_unchanged() -> None:
    assert envs.VLLM_SAIL_KPOOL_MAX_DENSE_POOLS == DENSE_POOL_LIMIT
    assert envs.VLLM_PPU_KPOOL_MAX_DENSE_POOLS == DENSE_POOL_LIMIT
    # Exactly at the limit the fallback is still allowed (strict ``>``).
    assert _fallback_call(1 << 10, 1 << 13, 4)["max_pools"] == DENSE_POOL_LIMIT
    # One pool above the limit aborts, reporting the unchanged ceil formula.
    above = (1 << 10) * (((1 << 13) + 4 + 3) // 4)
    with pytest.raises(RuntimeError, match="VLLM_SAIL_KPOOL_MAX_DENSE_POOLS") as raised:
        _fallback_call(1 << 10, (1 << 13) + 4, 4)
    assert f"{above} pool logits" in str(raised.value)
    assert above == DENSE_POOL_LIMIT + (1 << 10)


def test_dense_guard_still_blocks_the_startup_shape_that_motivated_the_switch() -> None:
    """1 MiB context, kpool 4, 249 captured rows: the request that aborted startup."""
    with pytest.raises(RuntimeError, match="65273856") as raised:
        _fallback_call(249, 1 << 20, 4)
    message = str(raised.value)
    assert "VLLM_SAIL_KPOOL_DECODE_LOGITS=non_paged" in message
    assert "default paged kernel" in message
