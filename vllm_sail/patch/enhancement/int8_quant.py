# SPDX-License-Identifier: Apache-2.0
"""Triton per-token-group INT8 quantization with round-to-nearest, for PPU.

PPU's INT8 dense GEMM (``PPUInt8ScaledMMLinearKernel``) needs a per-token-group
quantization that (a) can be forced onto the Triton path even though the compiled
``torch.ops._C`` kernel is "available", and (b) rounds to nearest instead of
truncating. The compiled kernel truncates, which costs measurable accuracy on
PPU's INT8 W8A8 path.

Two patches, both verbatim copies of the upstream bodies with the changed lines
marked. Verbatim rather than delegating because the additions are interleaved with
the existing logic — a new ``tl.constexpr`` inside the kernel and two new
parameters threaded through the wrapper — and Triton kernels cannot be wrapped
after the fact anyway.

``@patch`` sits **outside** ``@triton.jit`` so that the fully-decorated Triton
object is what gets installed; see ``vllm_sail/patch/README.md``.

## Divergence from the in-tree fork

The fork also replaces the wrapper's ``num_warps`` heuristic with a hardcoded
``num_warps = 2`` **unconditionally**, i.e. for every caller on every platform.
Two other upstream call sites use this Triton path
(``fused_moe/utils.py`` and ``models/deepseek_v2.py``), so in the fork they
silently lost the heuristic too. Here it is guarded on ``current_platform.is_ppu()``
so the tuning applies where it was measured and upstream behaviour is preserved
everywhere else.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.33.0,<0.34.0"


@patch(
    "vllm.model_executor.layers.quantization.utils.int8_utils",
    "_per_token_group_quant_int8",
    reason=(
        "PPU's INT8 W8A8 dense path needs round-to-nearest quantization; the "
        "upstream kernel truncates, which costs accuracy. Adds a `use_rounding` "
        "tl.constexpr that defaults to False, so every existing caller keeps "
        "upstream behaviour exactly."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream's per-token-group INT8 quantization offers a rounding mode "
        "(e.g. a `rounding=` argument), making this copy unnecessary."
    ),
)
@triton.jit
def _per_token_group_quant_int8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    # Stride of input
    y_stride,
    # Columns of input
    N,
    # Avoid to divide zero
    eps,
    # Information for int8
    int8_min,
    int8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
    # PPU MODIFICATION: begin -- opt-in round-to-nearest; default preserves upstream
    use_rounding: tl.constexpr = False,
    # PPU MODIFICATION: end
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.

    This function converts the tensor values into int8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / int8_max
    # PPU MODIFICATION: begin
    if use_rounding:
        clamp_val = tl.clamp(y / y_s, int8_min, int8_max)
        y_q = tl.extra.cuda.libdevice.round(clamp_val).to(y_q_ptr.dtype.element_ty)
    else:
        y_q = tl.clamp(y / y_s, int8_min, int8_max).to(y_q_ptr.dtype.element_ty)
    # PPU MODIFICATION: end

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@patch(
    "vllm.model_executor.layers.quantization.utils.int8_utils",
    "per_token_group_quant_int8",
    reason=(
        "PPU needs to force the Triton path (the compiled torch.ops._C kernel is "
        "present but truncates) and to enable round-to-nearest. Adds `use_triton` "
        "and `use_rounding`, both defaulting to False so existing callers are "
        "unaffected."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream exposes a rounding mode and a way to force the Triton path, or "
        "PPU's compiled INT8 quant kernel gains round-to-nearest."
    ),
)
def per_token_group_quant_int8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.int8,
    # PPU MODIFICATION: begin
    use_triton: bool = False,
    use_rounding: bool = False,
    # PPU MODIFICATION: end
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.

    It converts the tensor values into signed int8 values and returns the
    quantized tensor along with the scaling factor used for quantization.

    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor. Note that only `torch.int8`
            is supported for now.
        use_triton: Force the Triton kernel even when the compiled kernel is
            available. PPU needs this to reach `use_rounding`.
        use_rounding: Round to nearest instead of truncating.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
            scaling factor for quantization.
    """
    from vllm.platforms import current_platform

    assert x.shape[-1] % group_size == 0, (
        "the last dimension of `x` cannot be divisible by `group_size`"
    )
    assert x.is_contiguous(), "`x` is not contiguous"

    iinfo = torch.iinfo(dtype)
    int8_max = iinfo.max
    int8_min = iinfo.min

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )
    # Prefer native stable kernel on CUDA/ROCm when available.
    # PPU MODIFICATION: begin -- `use_triton` opt-out for the rounding path
    if (not use_triton) and current_platform.is_cuda_alike():
        # PPU MODIFICATION: end
        torch.ops._C.per_token_group_quant_int8(
            x, x_q, x_s, group_size, eps, float(int8_min), float(int8_max)
        )
        return x_q, x_s

    M = x.numel() // group_size
    N = group_size

    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    # PPU MODIFICATION: begin
    # The fork hardcodes num_warps = 2 for every platform. Two other upstream
    # callers use this Triton path (fused_moe/utils.py, models/deepseek_v2.py),
    # so the tuning is applied only on PPU, where it was measured.
    if current_platform.is_ppu():
        num_warps = 2
    else:
        num_warps = min(max(BLOCK // 256, 1), 8)
    # PPU MODIFICATION: end
    num_stages = 1
    _per_token_group_quant_int8[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        N,
        eps,
        int8_min=int8_min,
        int8_max=int8_max,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
        # PPU MODIFICATION: begin
        use_rounding=use_rounding,
        # PPU MODIFICATION: end
    )

    return x_q, x_s
