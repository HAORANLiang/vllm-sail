# SPDX-License-Identifier: Apache-2.0
"""Drop the HC down+inject padding shard on platforms without the fused GEMM.

``GatedResidual`` merges the low-rank down projection and the block-injection
logits into one ``MergedColumnParallelLinear`` and physically pads the merged
output to 16 rows for CuBLAS alignment. The padding is declared as a third
logical shard (``_input_mix_padding`` in the model's ``packed_modules_mapping``)
even though no checkpoint tensor backs it: only the fused down+SiLU GEMM
(``hc_down_silu``, gated on device capability 90) consumes the padded layout.

PPU has no capability-90 fused kernel, so ``_down_and_inject`` always takes the
split path, which discards the padding rows. The shard is still declared with
real rows, and the loader's per-shard tracking
(``StackedColumnParallelLinear.pop_missing_shards`` ->
``DefaultModelLoader.track_weights_loading``) then rejects the load: every HC
mixer reports one missing weight (96 across the model) and serving dies with a
``ValueError`` before the first request.

Gate ``pad_size`` on the same ``has_device_capability(90)`` predicate that
gates the fused kernel, so the padding shard exists exactly where the fused
path can consume it. Without the fused path the merged layer declares only the
two shards the checkpoint actually stores. Numerically equivalent on PPU: the
split path dropped those rows already. ``is_cuda()`` cannot serve as the gate
because ``PPUPlatform`` inherits ``CudaPlatform`` and reports True.

Leaf-style module: importable on a bare CPU runner, patches install when
``install()`` runs. The body is copied verbatim from upstream except for the
``pad_size`` computation and bound to the upstream module's globals, so live
helper lookups keep working.
"""

from __future__ import annotations

import importlib

from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch

_MODULE = "vllm.models.qwen4_exp.nvidia.hyperconnection"
_TARGET = "GatedResidual.__init__"
_AFFECTED = ">=0.34.0,<0.35.0"

REASON = (
    "GatedResidual declares the 16-row CuBLAS-alignment padding of the merged "
    "down+inject GEMM as a third logical shard on every platform, but only the "
    "capability-90 fused down+SiLU kernel consumes it and no checkpoint tensor "
    "backs it. On PPU the split path discards the padding rows, yet the "
    "loader's per-shard tracking still demands checkpoint data for the shard, "
    "so track_weights_loading raises ValueError with one missing weight per HC "
    "mixer (96 model-wide). Gate pad_size on has_device_capability(90), the "
    "same predicate as the fused kernel; is_cuda() is unusable because "
    "PPUPlatform inherits CudaPlatform."
)
REMOVE_WHEN = (
    "the release upstream pin advances to a vLLM whose MergedColumnParallelLinear "
    "takes padding_size and keeps the padding rows out of shard_ids (upstream "
    "main e83d82a already does), or GatedResidual itself gates pad_size on "
    "has_device_capability(90); verify by reading GatedResidual.__init__ and "
    "MergedColumnParallelLinear.__init__ at the pinned upstream_ref."
)

METADATA = ((f"{_MODULE}.{_TARGET}", REASON, _AFFECTED, REMOVE_WHEN),)

_installed = False
#: True when the upstream module cannot be imported; the fix has no consumer.
_unavailable = False


def install() -> None:
    """Register the capability-gated padding replacement once."""
    global _installed, _unavailable
    if _installed or _unavailable:
        return

    from vllm.logger import init_logger

    logger = init_logger(__name__)
    try:
        hyperconnection = importlib.import_module(_MODULE)
    except ImportError as exc:
        _unavailable = True
        logger.warning(
            "Skipping the HC padding-shard patch: the upstream qwen4_exp "
            "nvidia hyperconnection module cannot be imported (%s).",
            exc,
        )
        return

    def _with_target_globals(fn):
        return bind_body(fn, hyperconnection)

    @patch(
        _MODULE,
        _TARGET,
        reason=REASON,
        affected_versions=_AFFECTED,
        remove_when=REMOVE_WHEN,
    )
    @_with_target_globals
    def __init__(
        self,
        config: HyperConnectionConfig,
        use_combine: bool = True,
        prefix: str = "",
    ) -> None:
        # PPU MODIFICATION: begin
        super(GatedResidual, self).__init__()
        # PPU MODIFICATION: end
        self.config = config
        self.hc_lowrank = config.hc_lowrank
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.use_combine = use_combine

        norm_size = (
            self.hyper_hidden_size if config.hc_per_branch_norm else config.hidden_size
        )
        group_size = config.hidden_size if config.hc_per_branch_norm else None
        # Normalize each H-sized HC stream independently while retaining a
        # separate affine weight for every element of the HC*H layout.
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=group_size,
            dtype=config.params_dtype,
        )

        # -- vLLM Linear weights --------------------------------------------
        # The merged skinny-GEMM shape is physically padded to 16 rows to ensure
        # good alignment and performant implementation chosen by CuBLAS heuristics.
        # PPU MODIFICATION: begin
        # The padding shard only serves the fused down+SiLU GEMM (capability
        # 90). Elsewhere the split path discards those rows, but the declared
        # shard still demands a checkpoint tensor that does not exist.
        self.pad_size = (
            (-(self.hc_lowrank + self.hc_count)) % 16
            if use_combine and current_platform.has_device_capability(90)
            else 0
        )
        # PPU MODIFICATION: end
        if use_combine:
            self.input_mix_weight_down_block_inject = MergedColumnParallelLinear(
                self.hyper_hidden_size,
                [self.hc_lowrank, self.hc_count]
                + ([self.pad_size] if self.pad_size else []),
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down_block_inject"),
                return_bias=False,
                disable_tp=True,
            )
            weight = self.input_mix_weight_down_block_inject.weight
            self._use_hc_down_silu = (
                weight.shape[1] % 8 == 0
                and weight.dtype == torch.bfloat16
                and current_platform.has_device_capability(90)
            )
        else:
            self.input_mix_weight_down = ReplicatedLinear(
                self.hyper_hidden_size,
                self.hc_lowrank,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down"),
                return_bias=False,
            )
        self.input_mix_weight_up = ReplicatedLinear(
            self.hc_lowrank,
            self.hyper_hidden_size,
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "input_mix_weight_up"),
            return_bias=False,
        )
        self._use_hc_up_gate_mix = not envs.VLLM_BATCH_INVARIANT and (
            hc_up_gate_mix_supported(self.input_mix_weight_up.weight, self.hc_count)
        )

    _installed = True