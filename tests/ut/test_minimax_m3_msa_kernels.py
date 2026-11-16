# SPDX-License-Identifier: Apache-2.0
"""Tests for the MiniMax M3 SAIL-MSA per-token top-k helpers.

The CPU-only tier has neither torch nor Triton, so ``kernels`` is imported with
a fake ``triton.jit`` that runs the kernel body once per program id on NumPy
storage. This covers the launcher contract and the kernel's masking math; it
does not test Triton compilation.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

import vllm_sail.models.minimax_m3_msa  # noqa: F401  (marks the parent package)

np = pytest.importorskip("numpy")

_MODULE = "vllm_sail.models.minimax_m3_msa.kernels"
_NEG_INF = float("-inf")


class _Tensor:
    """The tensor surface ``force_init_local_scores`` reads."""

    def __init__(self, array, device: str = "cuda:0") -> None:
        self.array = array
        self.device = device
        self.shape = array.shape

    def is_contiguous(self) -> bool:
        return bool(self.array.flags.c_contiguous)

    def stride(self) -> tuple[int, ...]:
        return tuple(s // self.array.itemsize for s in self.array.strides)

    def numel(self) -> int:
        return int(self.array.size)


class _Pointer:
    def __init__(self, storage, offset=0) -> None:
        self.storage, self.offset = storage, offset

    def __add__(self, offset) -> _Pointer:
        return _Pointer(self.storage, self.offset + offset)


def _store(pointer: _Pointer, value, mask=True) -> None:
    offset, value, mask = np.broadcast_arrays(pointer.offset, value, mask)
    pointer.storage[offset[mask]] = value[mask]


class _Kernel:
    """``triton.jit`` stand-in: ``kernel[grid](...)`` loops over the grid."""

    def __init__(self, fn, tl) -> None:
        self.fn, self.tl, self.launches = fn, tl, []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append((grid, kwargs))
            args = [
                _Pointer(a.array.reshape(-1)) if isinstance(a, _Tensor) else a
                for a in args
            ]
            for pid in np.ndindex(*grid):
                self.tl.program_id = lambda axis, pid=pid: pid[axis]
                self.fn(*args, **kwargs)

        return launch


@pytest.fixture
def kernels(monkeypatch: pytest.MonkeyPatch):
    tl = types.ModuleType("triton.language")
    tl.constexpr = object()
    tl.arange = lambda start, stop: np.arange(start, stop)
    tl.load = lambda pointer: pointer.storage[pointer.offset]
    tl.store = _store
    tl.minimum = np.minimum
    tl.maximum = np.maximum

    triton = types.ModuleType("triton")
    triton.language = tl
    triton.jit = lambda fn: _Kernel(fn, tl)
    triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()

    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", tl)
    monkeypatch.delitem(sys.modules, _MODULE, raising=False)
    module = importlib.import_module(_MODULE)
    yield module
    sys.modules.pop(_MODULE, None)


def _scores(heads: int, tiles: int, tokens: int) -> _Tensor:
    return _Tensor(np.full((heads, tiles, tokens), _NEG_INF, dtype=np.float32))


def _pages(values) -> _Tensor:
    return _Tensor(np.asarray(values, dtype=np.int32))


def _expected(pages, heads: int, tiles: int, init: int, local: int, module):
    expected = np.full((heads, tiles, len(pages)), _NEG_INF, dtype=np.float32)
    for token, count in enumerate(pages):
        expected[:, : min(init, count), token] = module._INIT_SCORE
        # The local window is stored second and wins where the two overlap.
        expected[:, max(count - local, 0) : count, token] = module._LOCAL_SCORE
    return expected


@pytest.mark.parametrize(
    ("init", "local"), [(1, 4), (2, 3), (3, 0), (0, 2), (5, 5)], ids=str
)
def test_forces_per_token_init_and_local_tiles(kernels, init: int, local: int):
    # Causal page counts below, at and above both windows; the extra trailing
    # entry mirrors the padded positions buffer, which must not be read.
    pages = [1, 2, 3, 5, 9, 10]
    heads, tiles, tokens = 2, 10, 5
    scores = _scores(heads, tiles, tokens)
    kernels.force_init_local_scores(scores, _pages(pages), init, local)
    np.testing.assert_array_equal(
        scores.array, _expected(pages[:tokens], heads, tiles, init, local, kernels)
    )
    [(grid, kwargs)] = kernels._force_init_local_scores_kernel.launches
    assert grid == (tokens, heads)
    assert kwargs["BLOCK"] == 1 << (max(init, local, 1) - 1).bit_length()


def test_no_forced_tiles_skips_the_launch(kernels):
    scores = _scores(1, 4, 2)
    kernels.force_init_local_scores(scores, _pages([1, 2]), 0, 0)
    assert np.all(scores.array == _NEG_INF)
    assert not kernels._force_init_local_scores_kernel.launches


@pytest.mark.parametrize(
    ("shape", "transpose", "pages", "needle"),
    [
        ((1, 3, 4), True, [1] * 3, "contiguous"),
        ((1, 4, 3), False, [1, 1], "< tokens"),
        ((1, 4, 2), False, [1, 1], "pages on cpu"),
    ],
    ids=["non-contiguous", "short-pages", "device-mismatch"],
)
def test_rejects_inputs_the_kernel_cannot_address(
    kernels, shape, transpose: bool, pages, needle: str
):
    array = np.zeros(shape, np.float32)
    scores = _Tensor(array.transpose(0, 2, 1) if transpose else array)
    num_valid_pages = _pages(pages)
    if needle == "pages on cpu":
        num_valid_pages.device = "cpu"
    with pytest.raises(AssertionError, match=needle):
        kernels.force_init_local_scores(scores, num_valid_pages, 1, 1)
    assert not kernels._force_init_local_scores_kernel.launches


def test_clamp_keeps_ascending_causal_ids_with_a_negative_tail():
    torch = pytest.importorskip("torch")
    from vllm_sail.models.minimax_m3_msa.common import clamp_topk_to_causal

    topk = torch.tensor(
        [[[3, 0, 7, 1]], [[5, -1, 2, 6]], [[9, 8, 7, 6]]], dtype=torch.int32
    )
    clamp_topk_to_causal(topk, torch.tensor([4, 3, 2], dtype=torch.int32))
    assert topk.tolist() == [
        [[0, 1, 3, -1]],
        [[2, -1, -1, -1]],
        [[-1, -1, -1, -1]],
    ]
