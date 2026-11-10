# SPDX-License-Identifier: Apache-2.0
"""Optional PPU 1.5 BF16 dense DeepGEMM selection with a compile-safe op."""

from __future__ import annotations

import sys

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers import utils as _utils
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_sail.patch.utils import patch

logger = init_logger(__name__)
_META = dict(
    reason="PPU BF16 dense DeepGEMM needs a platform dispatch hook and fake implementation.",
    affected_versions=">=0.33.0,<0.34.0",
    remove_when="Unquantized dense GEMM supports backend registration.",
)


def ppu_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    # PPU NOTE: when VLLM_PPU_DENSE_BF16_DEEPGEMM is enabled, route K-major
    # BF16 dense GEMMs on PPU 1.5 to DeepGEMM; everything else keeps the
    # default acblas path (F.linear).
    from vllm_sail.utils.deep_gemm import should_use_deepgemm_for_bf16_linear

    if should_use_deepgemm_for_bf16_linear(x, weight, bias):
        logger.info_once("Using PPU DeepGEMM for unquantized BF16 dense GEMM")
        return torch.ops.vllm.ppu_bf16_deepgemm_linear(x, weight, bias)
    return torch.nn.functional.linear(x, weight, bias)


def ppu_bf16_deepgemm_linear_impl(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    from vllm_sail.utils.deep_gemm import bf16_dense_linear

    return bf16_dense_linear(x, weight, bias)


def ppu_bf16_deepgemm_linear_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="ppu_bf16_deepgemm_linear",
    op_func=ppu_bf16_deepgemm_linear_impl,
    fake_impl=ppu_bf16_deepgemm_linear_fake,
)
_original = _utils.dispatch_unquantized_gemm


@patch(_utils.__name__, "dispatch_unquantized_gemm", **_META)
def dispatch_unquantized_gemm(linear_backend: str = "auto"):
    if current_platform.is_ppu():
        return ppu_unquantized_gemm
    return _original(linear_backend)


_CONSUMERS = [
    "vllm.model_executor.layers.linear",
    "vllm.model_executor.layers.vocab_parallel_embedding",
    "vllm.model_executor.layers.quantization.compressed_tensors.transform.module",
]
for _name in _CONSUMERS:
    _consumer = sys.modules.get(_name)
    if (
        _consumer is not None
        and getattr(_consumer, "dispatch_unquantized_gemm", None) is _original
    ):
        patch(_name, "dispatch_unquantized_gemm", **_META)(dispatch_unquantized_gemm)
