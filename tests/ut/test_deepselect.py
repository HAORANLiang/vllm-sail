# SPDX-License-Identifier: Apache-2.0
"""SAIL DeepSelect regression coverage."""

import builtins
from types import SimpleNamespace as NS

import pytest

from tests.support.source import function

DEEP_GEMM = "vllm_sail/utils/deep_gemm.py"
DEEP_SELECT = "vllm_sail/utils/deep_select.py"
INDEXER = "vllm_sail/attention/ops/mla_sparse.py"


class Tensor:
    def __init__(self, shape, dtype="bf16"):
        self.shape, self.dtype = shape, dtype
        self.ndim = len(shape)

    def to(self, dtype):
        self.dtype = dtype
        return self

    def contiguous(self):
        return self


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
@pytest.mark.parametrize("available", [False, True])
def test_deepselect_accepts_bf16_and_fp32_when_wheel_is_available(dtype, available):
    ns = {
        "torch": NS(bfloat16="bf16", float32="fp32"),
        "is_deep_select_available": lambda: available,
    }
    assert (
        function(DEEP_SELECT, "can_use_deep_select", ns)(Tensor((2, 32), dtype), 8)
        is available
    )


@pytest.mark.parametrize("dtype", ["fp16", "fp64", "int32"])
def test_unsupported_dtype_never_imports_or_calls_the_wheel(dtype):
    def unavailable():
        pytest.fail("unsupported dtype must not trigger wheel import or call")

    ns = {
        "torch": NS(bfloat16="bf16", float32="fp32"),
        "is_deep_select_available": unavailable,
        "_wheel": unavailable,
    }
    assert not function(DEEP_SELECT, "can_use_deep_select", ns)(
        Tensor((2, 32), dtype), 8
    )
    with pytest.raises(TypeError, match="BF16 or FP32"):
        function(DEEP_SELECT, "deep_select_topk", ns)(
            Tensor((2, 32), dtype), 8, None, None
        )


def test_deepselect_keeps_dependency_and_abi_errors(monkeypatch):
    original_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "deep_select":
            raise ImportError("undefined symbol: wheel_abi")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    with pytest.raises(ImportError, match="wheel_abi"):
        function(DEEP_SELECT, "_wheel", {})()


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_deepselect_pads_input_and_preserves_k_and_strided_output(dtype):
    torch = pytest.importorskip("torch")
    calls = []

    def wheel_topk(scores, k, **kwargs):
        assert scores.dtype == logits.dtype
        torch.testing.assert_close(scores, logits, rtol=0, atol=0)
        calls.append((scores.stride(), k, kwargs))
        ends = kwargs["end"]
        cols = torch.arange(scores.shape[1])
        masked = scores.masked_fill(cols[None, :] >= ends[:, None], -float("inf"))
        ids = masked.topk(k, dim=1).indices.to(torch.int32)
        ids.masked_fill_(torch.arange(k)[None, :] >= ends[:, None], -1)
        return None, ids

    wheel = NS(topk=wheel_topk, get_stride_requirement=lambda: (1024, 32))
    fn = function(
        DEEP_SELECT,
        "deep_select_topk",
        {
            "torch": torch,
            "_wheel": lambda: wheel,
            "can_use_deep_select": lambda *args: True,
        },
    )
    dtype = getattr(torch, dtype)
    logits = torch.arange(39, dtype=dtype).reshape(3, 13)
    if dtype == torch.float32:
        logits = 1 + logits * torch.finfo(dtype).eps
    out = torch.empty((3, 9), dtype=torch.int32)[:, ::3]
    fn(logits, 3, torch.tensor([13, 1, 0], dtype=torch.int32), out)
    assert calls[0][0] == (1024 // logits.element_size(), 1)
    assert calls[0][1] == 3 and calls[0][2]["output_idx"] is None
    assert calls[0][2]["abort_when_nan_found"] is False
    assert set(out[0].tolist()) == {10, 11, 12}
    assert out[1].tolist() == [0, -1, -1]
    assert out[2].tolist() == [-1, -1, -1]


def test_deepselect_preserves_native_fp32_rejection():
    """Old-library FP32 rejection raises directly; no downcast or silent switch."""
    torch = pytest.importorskip("torch")
    scores = torch.zeros((1, 256), dtype=torch.float32)

    def topk(input, k, **kwargs):
        assert input is scores and input.dtype == torch.float32
        raise RuntimeError("PPU DeepSelect currently supports bfloat16 input only")

    fn = function(
        DEEP_SELECT,
        "deep_select_topk",
        {
            "torch": torch,
            "_wheel": lambda: NS(topk=topk, get_stride_requirement=lambda: (1024, 32)),
            "can_use_deep_select": lambda *args: True,
        },
    )
    with pytest.raises(RuntimeError, match="bfloat16 input only"):
        fn(scores, 8)


def test_deepselect_can_allocate_int64_indices_without_row_ends():
    torch = pytest.importorskip("torch")
    expected = torch.tensor([[1, 0]], dtype=torch.int64)
    calls = []

    def topk(scores, k, **kwargs):
        calls.append(kwargs)
        return None, expected

    wheel = NS(topk=topk, get_stride_requirement=lambda: (16, 32))
    fn = function(
        DEEP_SELECT,
        "deep_select_topk",
        {
            "torch": torch,
            "_wheel": lambda: wheel,
            "can_use_deep_select": lambda *args: True,
        },
    )
    assert (
        fn(torch.zeros((1, 8), dtype=torch.bfloat16), 2, indices_dtype=torch.int64)
        is expected
    )
    assert calls[0]["indices_type"] == torch.int64
    assert calls[0]["end"] is None


@pytest.mark.parametrize(
    "backend,available,expected",
    [
        ("auto", True, True),
        ("auto", False, False),
        ("persistent", True, False),
    ],
)
def test_indexer_backend_selection(backend, available, expected):
    fn = function(
        INDEXER, "_use_deep_select", {"can_use_deep_select": lambda *a: available}
    )
    assert fn(None, 2048, backend) is expected


def test_explicit_deepselect_fails_instead_of_rounding_fp32():
    fn = function(
        INDEXER, "_use_deep_select", {"can_use_deep_select": lambda *a: False}
    )
    with pytest.raises(RuntimeError, match="never rounded"):
        fn(None, 2048, "deep_select")
