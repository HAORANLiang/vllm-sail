# SPDX-License-Identifier: Apache-2.0
"""Match Step5 MTP quantization exclusions to the draft modules' names.

The PPU INT8 Step5 checkpoint keeps its MTP layers in BF16 and lists them in
the compressed-tensors ``ignore`` under checkpoint names
(``model.layers.92.mlp.down_proj``). ``Step5MTP`` registers the draft block's
modules one level deeper (``model.layers.92.mtp_block.mlp.down_proj``) and
renames the shared head (``transformer.shared_head.output`` ->
``shared_head.head``), but only rewrites the *weight* names on load. The
ignore entries never match, the BF16 draft layers are allocated as INT8 and
the MTP completeness check raises on the missing ``weight_scale``.

``Step5MTP.__init__`` is wrapped to apply the loader's renames to a private
copy of the quantization config before upstream's own ``language_model.``
strip; the target model's config is left untouched.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import replace

from vllm.config import VllmConfig
from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.step5.nvidia import mtp as _mtp

from vllm_sail.patch.utils import patch

_TARGET = "vllm.models.step5.nvidia.mtp"
_AFFECTED = ">=0.32.0,<0.33.0"

# Spec-layer modules that live beside, not inside, the draft block
# (Step5MTP._rewrite_spec_layer_name).
_AUXILIARY = r"(?:embed_tokens|enorm|hnorm|eh_proj|shared_head|mtp_block)"

_upstream_init = _mtp.Step5MTP.__init__


def mtp_quant_mapper(start_layer: int, num_layers: int) -> WeightsMapper:
    """Map checkpoint names of the MTP layers to their registered modules.

    Names may still carry the VL target's ``language_model.`` prefix.
    """
    patterns = {}
    for idx in range(start_layer, start_layer + num_layers):
        layer = rf"((?:language_model\.)?model\.layers\.{idx}\.)"
        patterns[re.compile(rf"^{layer}transformer\.(shared_head\..*)$")] = r"\1\2"
        patterns[re.compile(rf"^{layer}shared_head\.output(.*)$")] = (
            r"\1shared_head.head\2"
        )
        patterns[
            re.compile(rf"^{layer}(?!{_AUXILIARY}(?:\.|$))(?:transformer\.)?(.*)$")
        ] = r"\1mtp_block.\2"
    return WeightsMapper(orig_to_new_regex=patterns)


@patch(
    _TARGET,
    "Step5MTP.__init__",
    reason=(
        "Upstream maps only the language_model. prefix of the quantization "
        "config for the Step5 draft model; ignore entries naming the BF16 MTP "
        "layers miss the mtp_block./shared_head.head module paths, so those "
        "layers are allocated quantized and the load fails on missing scales."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "Step5MTP remaps the quantization config to the mtp_block / "
        "shared_head.head module names itself."
    ),
)
def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
    if vllm_config.quant_config is not None:
        config = _mtp._get_mtp_config(vllm_config)
        quant_config = deepcopy(vllm_config.quant_config)
        quant_config.apply_vllm_mapper(
            mtp_quant_mapper(config.num_hidden_layers, config.num_nextn_predict_layers)
        )
        vllm_config = replace(vllm_config, quant_config=quant_config)
    _upstream_init(self, vllm_config=vllm_config, prefix=prefix)
