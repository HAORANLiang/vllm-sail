# SPDX-License-Identifier: Apache-2.0
"""PPU hunks in the generic fused-MoE plumbing.

Four small fork deltas that no other patch covers:

* ``fused_moe/config.py``: ``_get_config_dtype_str`` gains ``use_int8_w8a8``
  (tuned-config name ``"int8_w8a8"``) and ``int8_w8a8_moe_quant_config``
  accepts ``gemm1_clamp_limit``.
* ``fused_moe/oracle/int8.py``: ``make_int8_moe_quant_config`` builds the
  PPU int8 config with the swiglu limit as ``gemm1_clamp_limit`` (the PPU
  int8 experts read it).
* ``fused_moe/utils.py``: ``moe_kernel_quantize_input`` quantizes PPU MXFP4
  activations with the plugin's ``downcast_to_mxfp4``.
* ``fused_moe/experts/triton_moe.py``: ``TritonExperts.moe_sum`` uses the
  PPU Triton reduction above 1024 tokens. Per-direction tuning lives in
  ``enhancement/triton_moe.py``.

All patches delegate to upstream; only the PPU-specific case is intercepted.
Leaf-style module (like ``patch/enhancement/residual/``): importable on a bare
CPU runner, patches install when ``install()`` runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_sail.patch.utils import patch

if TYPE_CHECKING:
    import torch

_CONFIG_MODULE = "vllm.model_executor.layers.fused_moe.config"
_INT8_ORACLE_MODULE = "vllm.model_executor.layers.fused_moe.oracle.int8"
_UTILS_MODULE = "vllm.model_executor.layers.fused_moe.utils"
_TRITON_MODULE = "vllm.model_executor.layers.fused_moe.experts.triton_moe"
# By-value imports present in vLLM 0.30. Keep this inventory checked against
# upstream source; importing optional EP backends here would load their SDKs.
_QUANTIZE_INPUT_CONSUMERS = tuple(
    "vllm.model_executor.layers.fused_moe." + suffix
    for suffix in (
        "experts.fused_batched_moe",
        "experts.nvfp4_emulation_moe",
        "experts.triton_moe",
        "fused_moe",
        "prepare_finalize.batched",
        "prepare_finalize.deepep_ht",
        "prepare_finalize.deepep_ll",
        "prepare_finalize.deepep_v2",
        "prepare_finalize.flashinfer_nvlink_one_sided",
        "prepare_finalize.flashinfer_nvlink_two_sided",
        "prepare_finalize.naive_dp_ep",
        "prepare_finalize.nixl_ep",
        "prepare_finalize.no_dp_ep",
    )
) + ("vllm.model_executor.models.k2_horizon",)
AFFECTED_VERSIONS = ">=0.33.0,<0.34.0"

REASON_DTYPE_STR = (
    "PPU int8 W8A8 MoE flows look up tuned configs by the name int8_w8a8; "
    "upstream's _get_config_dtype_str has no use_int8_w8a8 flag and never "
    "returns it."
)
REASON_INT8_CONFIG = (
    "The fork threads gemm1_clamp_limit through int8_w8a8_moe_quant_config "
    "and builds the PPU int8 MoE quant config with the swiglu limit in that "
    "slot; the PPU int8 experts read gemm1_clamp_limit."
)
REASON_MXFP4_QUANTIZE = (
    "PPU MXFP4 activations are quantized with the plugin's downcast_to_mxfp4; "
    "upstream raises NotImplementedError for native mxfp4 input quantization "
    "on non-XPU platforms."
)
REASON_MOE_SUM = (
    "The fork switches TritonExperts.moe_sum to a Triton reduction for more "
    "than 1024 tokens on PPU; upstream's C++ moe_sum is the slow path there."
)
REMOVE_WHEN_DTYPE_STR = "upstream adds use_int8_w8a8 to _get_config_dtype_str itself."
REMOVE_WHEN_INT8_CONFIG = (
    "upstream threads gemm1_clamp_limit/swiglu_limit through the int8 MoE "
    "quant-config builders itself."
)
REMOVE_WHEN_MXFP4_QUANTIZE = (
    "upstream's moe_kernel_quantize_input grows a PPU (or generic mxfp4 "
    "downcast) branch."
)
REMOVE_WHEN_MOE_SUM = (
    "upstream's moe_sum is fast on PPU for large token counts, or the fork's "
    "moe_sum_reduce_triton kernel is upstreamed and selected by platform."
)

METADATA = (
    (
        f"{_CONFIG_MODULE}._get_config_dtype_str",
        REASON_DTYPE_STR,
        AFFECTED_VERSIONS,
        REMOVE_WHEN_DTYPE_STR,
    ),
    (
        f"{_CONFIG_MODULE}.int8_w8a8_moe_quant_config",
        REASON_INT8_CONFIG,
        AFFECTED_VERSIONS,
        REMOVE_WHEN_INT8_CONFIG,
    ),
    (
        f"{_INT8_ORACLE_MODULE}.make_int8_moe_quant_config",
        REASON_INT8_CONFIG,
        AFFECTED_VERSIONS,
        REMOVE_WHEN_INT8_CONFIG,
    ),
    (
        f"{_UTILS_MODULE}.moe_kernel_quantize_input",
        REASON_MXFP4_QUANTIZE,
        AFFECTED_VERSIONS,
        REMOVE_WHEN_MXFP4_QUANTIZE,
    ),
    (
        f"{_TRITON_MODULE}.TritonExperts.moe_sum",
        REASON_MOE_SUM,
        AFFECTED_VERSIONS,
        REMOVE_WHEN_MOE_SUM,
    ),
)

_installed = False


def _rebind_loaded_quantize_input_aliases(original, replacement) -> None:
    """Repair consumers loaded before the provider patch, without importing them."""
    import sys

    for name in _QUANTIZE_INPUT_CONSUMERS:
        consumer = sys.modules.get(name)
        if consumer is None:
            continue
        if getattr(consumer, "moe_kernel_quantize_input", None) is original:
            patch(
                name,
                "moe_kernel_quantize_input",
                reason=(
                    "Preloaded MoE consumers retain the upstream quantizer and "
                    "bypass PPU MXFP4 input quantization."
                ),
                affected_versions=AFFECTED_VERSIONS,
                remove_when=REMOVE_WHEN_MXFP4_QUANTIZE,
            )(replacement)


def install() -> None:
    """Apply the patches. Requires vLLM to be importable; idempotent."""
    global _installed
    if _installed:
        return

    from vllm.model_executor.layers.fused_moe import config as _fm_config
    from vllm.model_executor.layers.fused_moe import utils as _fm_utils
    from vllm.model_executor.layers.fused_moe.experts import triton_moe
    from vllm.model_executor.layers.fused_moe.oracle import int8 as _int8_oracle

    _upstream_dtype_str = _fm_config._get_config_dtype_str
    _upstream_int8_config = _fm_config.int8_w8a8_moe_quant_config
    _upstream_make_int8 = _int8_oracle.make_int8_moe_quant_config
    _upstream_quantize_input = _fm_utils.moe_kernel_quantize_input
    _upstream_moe_sum = triton_moe.TritonExperts.moe_sum

    @patch(
        _CONFIG_MODULE,
        "_get_config_dtype_str",
        reason=REASON_DTYPE_STR,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN_DTYPE_STR,
    )
    def _get_config_dtype_str(
        dtype,
        use_fp8_w8a8: bool = False,
        use_fp8_w8a16: bool = False,
        use_int8_w8a8: bool = False,
        **kwargs,
    ):
        if use_int8_w8a8 and not (use_fp8_w8a8 or use_fp8_w8a16):
            return "int8_w8a8"
        return _upstream_dtype_str(
            dtype,
            use_fp8_w8a8=use_fp8_w8a8,
            use_fp8_w8a16=use_fp8_w8a16,
            **kwargs,
        )

    @patch(
        _CONFIG_MODULE,
        "int8_w8a8_moe_quant_config",
        reason=REASON_INT8_CONFIG,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN_INT8_CONFIG,
    )
    def _int8_w8a8_moe_quant_config(
        *args, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None, **kwargs
    ):
        config = _upstream_int8_config(*args, **kwargs)
        config.gemm1_alpha = gemm1_alpha
        config.gemm1_beta = gemm1_beta
        if gemm1_clamp_limit is not None:
            config.gemm1_clamp_limit = gemm1_clamp_limit
        return config

    @patch(
        _INT8_ORACLE_MODULE,
        "make_int8_moe_quant_config",
        reason=REASON_INT8_CONFIG,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN_INT8_CONFIG,
    )
    def _make_int8_moe_quant_config(
        int8_backend,
        w1_scale,
        w2_scale,
        a1_scale=None,
        a2_scale=None,
        w1_bias=None,
        w2_bias=None,
        per_act_token_quant: bool = False,
        layer=None,
        swiglu_limit: float | None = None,
        gemm1_alpha: float | None = None,
        gemm1_beta: float | None = None,
    ):
        from vllm.platforms import current_platform

        if current_platform.is_ppu():
            return _int8_w8a8_moe_quant_config(
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                per_act_token_quant=per_act_token_quant,
                gemm1_clamp_limit=swiglu_limit,
                gemm1_alpha=gemm1_alpha,
                gemm1_beta=gemm1_beta,
                w1_bias=w1_bias,
                w2_bias=w2_bias,
            )
        return _upstream_make_int8(
            int8_backend,
            w1_scale,
            w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            per_act_token_quant=per_act_token_quant,
            layer=layer,
        )

    @patch(
        _UTILS_MODULE,
        "moe_kernel_quantize_input",
        reason=REASON_MXFP4_QUANTIZE,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN_MXFP4_QUANTIZE,
    )
    def _moe_kernel_quantize_input(
        A: torch.Tensor,
        A_scale: torch.Tensor | None,
        quant_dtype,
        per_act_token_quant: bool,
        block_shape=None,
        **kwargs,
    ):
        from vllm.platforms import current_platform

        if quant_dtype == "mxfp4" and current_platform.is_ppu():
            from vllm_sail.model_executor.layers.quantization.utils.mxfp4_utils import (  # noqa: E501
                downcast_to_mxfp4,
            )

            return downcast_to_mxfp4(A, axis=1)
        return _upstream_quantize_input(
            A,
            A_scale,
            quant_dtype,
            per_act_token_quant,
            block_shape=block_shape,
            **kwargs,
        )

    _rebind_loaded_quantize_input_aliases(
        _upstream_quantize_input, _moe_kernel_quantize_input
    )

    @patch(
        _TRITON_MODULE,
        "TritonExperts.moe_sum",
        reason=REASON_MOE_SUM,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN_MOE_SUM,
    )
    def _moe_sum(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        topk_ids: torch.Tensor | None = None,
        expert_map: torch.Tensor | None = None,
    ) -> None:
        from vllm.platforms import current_platform

        if (
            current_platform.is_ppu()
            and input.shape[0] > 1024
            and expert_map is None
        ):
            from vllm_sail.model_executor.layers.fused_moe.triton_kernels import (
                moe_sum_reduce_triton,
            )

            moe_sum_reduce_triton(input, output)
            return
        _upstream_moe_sum(self, input, output, topk_ids, expert_map)

    _installed = True
