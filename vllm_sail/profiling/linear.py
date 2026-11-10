# ruff: noqa: E731, W291, W293, UP037
# ruff: noqa: F821
# Copied bodies resolve globals in their upstream module through bind_body.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in GEMM scopes, keeping communication outside the measured kernel."""

from __future__ import annotations

from vllm.model_executor.layers import linear as _linear

from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch


def _ReplicatedLinear_forward(
    self,
    x: torch.Tensor,
) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
    # PPU MODIFICATION: begin
    from vllm_sail.utils.nvtx_ops import (
        nvtx_pop_range_for_gemm,
        nvtx_push_range_for_gemm,
    )

    NVTX_PROFILE = True
    # PPU MODIFICATION: end
    bias = self.bias if not self.skip_bias_add else None
    # PPU MODIFICATION: begin
    assert self.quant_method is not None
    if NVTX_PROFILE:
        tmp_weight = getattr(self, "weight", None)
        nvtx_push_range_for_gemm("ReplicatedLinear", x, tmp_weight, None, None, bias)
    # PPU MODIFICATION: end
    output = self.quant_method.apply(self, x, bias)
    # PPU MODIFICATION: begin
    if NVTX_PROFILE:
        nvtx_pop_range_for_gemm(output)
    # PPU MODIFICATION: end
    if not self.return_bias:
        return output
    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias


patch(
    _linear.__name__,
    "ReplicatedLinear.forward",
    reason="Opt-in PPU profiling adds the fork's per-linear GEMM scopes.",
    affected_versions=">=0.33.0,<0.34.0",
    remove_when="Upstream linear kernels expose equivalent profiling hooks.",
)(bind_body(_ReplicatedLinear_forward, _linear))


def _ColumnParallelLinear_forward(
    self,
    input_,
) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
    # PPU MODIFICATION: begin
    from vllm_sail.utils.nvtx_ops import (
        nvtx_pop_range_for_gemm,
        nvtx_push_range_for_gemm,
    )

    NVTX_PROFILE = True
    # PPU MODIFICATION: end
    bias = self.bias if not self.skip_bias_add else None

    # Matrix multiply.
    # PPU MODIFICATION: begin
    assert self.quant_method is not None

    if NVTX_PROFILE:
        tmp_weight = getattr(self, "weight", None)
        nvtx_push_range_for_gemm(
            "ColumnParallelLinear", input_, tmp_weight, None, None, bias
        )
    # PPU MODIFICATION: end
    output_parallel = self.quant_method.apply(self, input_, bias)
    # PPU MODIFICATION: begin
    if NVTX_PROFILE:
        nvtx_pop_range_for_gemm(output_parallel)
    # PPU MODIFICATION: end

    if self.gather_output and self.tp_size > 1:
        # All-gather across the partitions.
        output = tensor_model_parallel_all_gather(output_parallel)
    else:
        output = output_parallel

    if not self.return_bias:
        return output
    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias


patch(
    _linear.__name__,
    "ColumnParallelLinear.forward",
    reason="Opt-in PPU profiling adds the fork's per-linear GEMM scopes.",
    affected_versions=">=0.33.0,<0.34.0",
    remove_when="Upstream linear kernels expose equivalent profiling hooks.",
)(bind_body(_ColumnParallelLinear_forward, _linear))


def _RowParallelLinear_forward(
    self,
    input_,
) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
    # PPU MODIFICATION: begin
    from vllm_sail.utils.nvtx_ops import (
        nvtx_pop_range_for_gemm,
        nvtx_push_range_for_gemm,
    )

    NVTX_PROFILE = True
    # PPU MODIFICATION: end
    if self.input_is_parallel:
        input_parallel = input_
    else:
        split_input = split_tensor_along_last_dim(input_, num_partitions=self.tp_size)
        input_parallel = split_input[self.tp_rank].contiguous()

    # Matrix multiply.
    # Only fuse bias add into GEMM for rank 0 (this ensures that
    # bias will not get added more than once in TP>1 case)
    bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
    # PPU MODIFICATION: begin

    if NVTX_PROFILE:
        tmp_weight = getattr(self, "weight", None)
        nvtx_push_range_for_gemm(
            "RowParallelLinear", input_, tmp_weight, None, None, bias_
        )
    # PPU MODIFICATION: end
    output_parallel = self.quant_method.apply(self, input_parallel, bias_)
    # PPU MODIFICATION: begin
    if NVTX_PROFILE:
        nvtx_pop_range_for_gemm(output_parallel)
    # PPU MODIFICATION: end

    if self.reduce_results and self.tp_size > 1:
        output = tensor_model_parallel_all_reduce(output_parallel)
    else:
        output = output_parallel

    if not self.return_bias:
        return output
    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias


patch(
    _linear.__name__,
    "RowParallelLinear.forward",
    reason="Opt-in PPU profiling adds the fork's per-linear GEMM scopes.",
    affected_versions=">=0.33.0,<0.34.0",
    remove_when="Upstream linear kernels expose equivalent profiling hooks.",
)(bind_body(_RowParallelLinear_forward, _linear))
