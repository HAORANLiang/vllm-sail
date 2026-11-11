# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton

from vllm_sail.patch.utils import patch


@patch(
    "vllm.model_executor.kernels.attention.dsa.candidate_blocks",
    "_max_with_nan",
    reason="SAIL Triton promotes BF16 maximum to FP32, violating reduction dtype matching.",
    affected_versions=">=0.31.0,<0.32.0",
    remove_when="The candidate maximum or SAIL Triton preserves the reduction input dtype.",
)
@triton.jit
def _max_with_nan(a, b):
    # PPU MODIFICATION: begin
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL).to(a.dtype)
    # PPU MODIFICATION: end
