# SPDX-License-Identifier: Apache-2.0
"""Find SAIL's FP8 block-GEMM tuning data after upstream lookup.

vLLM 0.30 removed the unused INT8 block-GEMM implementation and its config
lookup. SAIL's ACEXT INT8 linear path is independent of that deleted code.
"""

from __future__ import annotations

import json
import os
from typing import Any

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils import fp8_utils
from vllm.utils.platform_utils import get_device_name_as_file_name

from vllm_sail.patch.utils import patch
from vllm_sail.registry.tuned_configs import block_quant_config_path

logger = init_logger(__name__)

_AFFECTED = ">=0.33.0,<0.34.0"
_REMOVE_WHEN = (
    "upstream gives get_w8a8_block_fp8_configs a user-config-folder hook "
    "like fused_moe.get_moe_configs already has (envs.VLLM_TUNED_CONFIG_FOLDER)."
)

_upstream_fp8 = fp8_utils.get_w8a8_block_fp8_configs


def _load_plugin_config(json_file_name: str, kind: str) -> dict[int, Any] | None:
    """Load a plugin-shipped block-quant config, or return None."""
    path = block_quant_config_path(json_file_name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        logger.info(
            "Using vllm-sail bundled configuration from %s for W8A8 Block %s kernel.",
            path,
            kind,
        )
        return {int(key): val for key, val in json.load(f).items()}


@patch(
    "vllm.model_executor.layers.quantization.utils.fp8_utils",
    "get_w8a8_block_fp8_configs",
    reason=(
        "Upstream resolves its tuned-config directory relative to its own "
        "__file__ and offers no user-folder hook, so the FP8 block-quant configs "
        "this plugin ships for PPU-ZW810/ZW810E/ZW-M890P are never found and the "
        "kernel silently falls back to an untuned default."
    ),
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def get_w8a8_block_fp8_configs(
    N: int, K: int, block_n: int, block_k: int
) -> dict[int, Any] | None:
    config = _upstream_fp8(N, K, block_n, block_k)
    if config is not None:
        return config

    device_name = get_device_name_as_file_name()
    # Match upstream fp8_utils.get_w8a8_block_fp8_configs, including the lack
    # of a space after the comma in block_shape.
    json_file_name = (
        f"N={N},K={K},device_name={device_name},dtype=fp8_w8a8,"
        f"block_shape=[{block_n},{block_k}].json"
    )
    return _load_plugin_config(json_file_name, "FP8")
