# SPDX-License-Identifier: Apache-2.0
"""PPU changes to ``vllm.model_executor.models.step3p5_mtp``.

Ports the fork's +12 hunk: the MTP completeness check must treat int8/int4
quant zero-point buffers (``.{k,v,q}_zero_point``) as optional checkpoint
parameters, exactly like the existing ``.{k,v,q,prob}_scale`` ones. Verbatim
upstream body with one marked block; the set is built method-locally, so
delegation is impossible.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from vllm.model_executor.model_loader.mtp_validation import (
    is_mtp_completeness_check_enabled,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.step3p5_mtp import Step3p5MTP
from vllm.model_executor.models.utils import get_spec_layer_idx_from_weight_name

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.33.0,<0.34.0"


@patch(
    "vllm.model_executor.models.step3p5_mtp",
    "Step3p5MTP.load_weights",
    reason=(
        "PPU int8/int4 quant checkpoints omit the zero-point buffers, and "
        "upstream's completeness check then raises on load. Adds the three "
        "zero-point suffixes to the optional-parameter set. Verbatim "
        "upstream body with one marked block; the set is method-local."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream's optional_params set in Step3p5MTP.load_weights includes "
        "the zero_point suffixes itself."
    ),
)
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    params_dict = dict(self.named_parameters())
    base_layer = (
        "base_layer." if any(".base_layer." in name for name in params_dict) else ""
    )

    expert_params_mapping = [
        (f".moe.experts.{base_layer}w13_weight", ".moe.gate_proj.weight", "w1"),
        (f".moe.experts.{base_layer}w13_weight", ".moe.up_proj.weight", "w3"),
        (f".moe.experts.{base_layer}w2_weight", ".moe.down_proj.weight", "w2"),
    ]

    loaded_params: set[str] = set()
    for name, loaded_weight in weights:
        if "rotary_emb.inv_freq" in name:
            continue
        spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
        if "embed_tokens" not in name and spec_layer is None:
            continue
        name = self._rewrite_spec_layer_name(spec_layer, name)
        for param_name, weight_name, shard_id in stacked_params_mapping:
            # Skip non-stacked layers and experts (experts handled below).
            if weight_name not in name:
                continue
            # We have mlp.experts[0].gate_proj in the checkpoint.
            # Since we handle the experts below in expert_params_mapping,
            # we need to skip here BEFORE we update the name, otherwise
            # name will be updated to mlp.experts[0].gate_up_proj, which
            # will then be updated below in expert_params_mapping
            # for mlp.experts[0].gate_gate_up_proj, which breaks load.
            if ("mlp.experts." in name) and name not in params_dict:
                continue
            if "experts" in name or "moe" in name:
                continue
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            for mapping in expert_params_mapping:
                param_name, weight_name, shard_id = mapping
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias") or name.endswith("_bias")
                ) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                for expert_id in range(loaded_weight.shape[0]):
                    loaded_weight_expert = loaded_weight[expert_id]
                    weight_loader(
                        param,
                        loaded_weight_expert,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                loaded_params.add(name)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias")
                    and name not in params_dict
                    or "tok_embeddings" in name
                ):
                    continue

                if spec_layer is not None and ".transformer." in name:
                    name = name.replace(".transformer.", ".")
                if "shared_head" in name:
                    name = name.replace("shared_head.output", "shared_head.head")
                if "embed_tokens" in name:
                    assert (
                        hasattr(self.config, "num_nextn_predict_layers")
                        and self.config.num_nextn_predict_layers > 0
                    )
                    name = "model.embed_tokens.weight"
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
        loaded_params.add(name)
    params_need_to_load = set(params_dict.keys())
    # Some KV cache scales are optional: checkpoints may omit them and vLLM
    # will fall back to default scales during initialization.
    optional_params = {
        name
        for name, param in params_dict.items()
    # PPU MODIFICATION: begin
        if name.endswith(
            (
                ".k_scale",
                ".v_scale",
                ".q_scale",
                ".prob_scale",
                ".k_zero_point",
                ".v_zero_point",
                ".q_zero_point",
            )
        )
    # PPU MODIFICATION: end
        and getattr(param, "numel", lambda: 0)() == 1
        and getattr(param, "requires_grad", False) is False
    }
    params_need_to_load -= optional_params
    if params_need_to_load != loaded_params and is_mtp_completeness_check_enabled():
        missing_params = list(params_need_to_load - loaded_params)
        param_name_example = missing_params[0]
        raise RuntimeError(
            "Some parameters like "
            f"{param_name_example} are not in the checkpoint and will falsely "
            "use random initialization"
        )
    return loaded_params
