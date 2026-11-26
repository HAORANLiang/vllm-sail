# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CI_DIR = ROOT / "scripts" / "ci"
RECIPES_DIR = ROOT / "tests" / "ppu" / "recipes"
POLICY_DIR = ROOT / "tests" / "ppu" / "policy"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ppu_recipe = _load_module("ppu_recipe", CI_DIR / "ppu_recipe.py")
ppu_recipe_runner = _load_module("ppu_recipe_runner", CI_DIR / "ppu_recipe_runner.py")
ppu_e2e_select = _load_module("ppu_e2e_select", CI_DIR / "ppu_e2e_select.py")


def _parse_sample_kv_transfer_config(raw: str) -> dict:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid json: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("kv-transfer-config must be a JSON object")
    allowed = {
        "kv_connector",
        "engine_id",
        "kv_buffer_device",
        "kv_role",
        "kv_port",
        "kv_connector_extra_config",
        "kv_connector_module_path",
        "enable_permute_local_kv",
        "kv_load_failure_policy",
    }
    extra = sorted(set(parsed) - allowed)
    if extra:
        raise argparse.ArgumentTypeError(
            f"unexpected KVTransferConfig fields: {', '.join(extra)}"
        )
    if "kv_port" in parsed and type(parsed["kv_port"]) is not int:
        raise argparse.ArgumentTypeError(
            f"kv_port must be int, got {type(parsed['kv_port']).__name__}"
        )
    if parsed.get("kv_role") not in {None, "kv_producer", "kv_consumer", "kv_both"}:
        raise argparse.ArgumentTypeError(f"invalid kv_role: {parsed.get('kv_role')!r}")
    if "kv_connector_extra_config" in parsed and not isinstance(
        parsed["kv_connector_extra_config"], dict
    ):
        raise argparse.ArgumentTypeError("kv_connector_extra_config must be dict")
    return parsed


def _sample_vllm_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vllm serve")
    parser.add_argument("model_tag")
    parser.add_argument("--served-model-name")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--pipeline-parallel-size", type=int)
    parser.add_argument("--prefill-context-parallel-size", type=int)
    parser.add_argument("--data-parallel-size", type=int)
    parser.add_argument("--data-parallel-size-local", type=int)
    parser.add_argument("--data-parallel-rank", type=int)
    parser.add_argument("--data-parallel-backend", default="mp")
    parser.add_argument("--distributed-executor-backend")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--data-parallel-external-lb", action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--data-parallel-hybrid-lb", action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--enable-expert-parallel", action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--disable-expert-parallel", action="store_true")
    parser.add_argument("--all2all-backend")
    parser.add_argument("--nnodes", type=int)
    parser.add_argument("--node-rank", type=int)
    parser.add_argument("--master-addr")
    parser.add_argument("--master-port", type=int)
    parser.add_argument("--kv-transfer-config", type=_parse_sample_kv_transfer_config)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-model-len")
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--distributed-timeout-seconds", type=int)
    parser.add_argument("--cpu-distributed-timeout-seconds", type=int)
    parser.add_argument("--enable-auto-tool-choice", action="store_true")
    parser.add_argument("--tool-call-parser")
    parser.add_argument("--tokenizer-mode")
    parser.add_argument("--reasoning-parser")
    parser.add_argument("--reasoning-config")
    parser.add_argument("--speculative-config")
    parser.add_argument("--compilation-config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--no-enable-prefix-caching", action="store_true")
    parser.add_argument("--load-format")
    parser.add_argument("--hf-overrides")
    return parser


def _yaml_files() -> list[Path]:
    return sorted(
        p
        for p in (ROOT / "tests" / "ppu").rglob("*.yaml")
        if not p.name.startswith(".")
    )


def test_checked_in_recipes_and_policies_validate_cleanly() -> None:
    files = _yaml_files()
    assert files
    full_report = ppu_recipe.validate_paths(files, vllm_parser_mode="skip")
    assert len(full_report["validated_files"]) == len(files)
    assert full_report["checked_combinations"] >= 35
    # The CPU stub covers the pilot flags. The runtime parser validates new recipes.
    report = ppu_recipe.validate_paths(
        [
            RECIPES_DIR / "zai-org" / "GLM-5.3.yaml",
            RECIPES_DIR / "deepseek-ai" / "DeepSeek-V4-Flash.yaml",
            RECIPES_DIR / "moonshotai" / "Kimi-K2.6.yaml",
        ],
        vllm_parser_mode="required",
        parser=_sample_vllm_parser(),
    )
    assert report["parser_checked"] is True
    assert report["checked_combinations"] >= 18
    assert report["checked_flag_sets"] >= 20


def test_glm53_flash_matches_legacy_tp2_and_default_fast_sanity() -> None:
    recipe = RECIPES_DIR / "zai-org" / "GLM-5.3-Flash.yaml"
    policy = POLICY_DIR / "models" / "glm-5.3-flash.yaml"
    legacy = next(
        case
        for case in json.loads((CI_DIR / "ppu_e2e_models.json").read_text())
        if case["key"] == "glm-5.3-flash-mxfp4-tp2"
    )
    resolved = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="model_sanity"
    )
    assert resolved["ppu_count"] == legacy["ppu_count"] == 2
    assert resolved["hf_model_id"] == legacy["model_id"]
    assert resolved["golden_model_key"] == legacy["model_key"]
    assert {
        flag: value
        for flag, value, _ in ppu_recipe._parse_flag_entries(resolved["flags"])
        if flag != "--enable-prefix-caching"
    } == {
        flag: value
        for flag, value, _ in ppu_recipe._parse_flag_entries(legacy["server_cmd"][9:])
    }
    fast = ppu_e2e_select.build_recipe_matrix("", "fast_sanity")["include"]
    assert any(
        item["key"] == "glm-5.3-flash" and item["ppu_count"] == 2 for item in fast
    )


def test_resolve_glm53_matches_catalog_and_supports_presets() -> None:
    recipe = RECIPES_DIR / "zai-org" / "GLM-5.3.yaml"
    policy = POLICY_DIR / "models" / "glm-5.3.yaml"
    decode_policy = POLICY_DIR / "models" / "glm-5.3-decode.yaml"

    resolved_ms = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="model_sanity"
    )
    assert resolved_ms["ppu_count"] == 4
    assert resolved_ms["preset"] == "prefill"
    assert resolved_ms["hardware"] == "zhenwu_m890"
    assert resolved_ms["precision"] == "mxfp4"
    assert resolved_ms["runtime_env"] == {"VLLM_SERVER_DEV_MODE": "1"}
    assert resolved_ms["flags"][:-1] == [
        "--trust-remote-code",
        "--tensor-parallel-size",
        "4",
        "--max-model-len",
        "16384",
        "--gpu-memory-utilization",
        "0.9",
        "--distributed-timeout-seconds",
        "3600",
        "--cpu-distributed-timeout-seconds",
        "3600",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "glm47",
        "--reasoning-parser",
        "glm47",
    ]
    assert resolved_ms["flags"][-1] == "--enable-prefix-caching"
    assert "--speculative-config" not in resolved_ms["flags"]

    resolved_fs = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="fast_sanity"
    )
    assert resolved_fs["preset"] == "prefill"
    assert "--load-format" in resolved_fs["flags"]
    assert "dummy" in resolved_fs["flags"]
    assert resolved_fs["layer_override"] == "auto"
    assert "--hf-overrides" not in resolved_fs["flags"]

    resolved_fs_7l = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="fast_sanity", layer_override=7
    )
    idx_ov = resolved_fs_7l["flags"].index("--hf-overrides")
    assert json.loads(resolved_fs_7l["flags"][idx_ov + 1]) == {"num_hidden_layers": 7}
    assert resolved_fs_7l["config_digest"] != resolved_fs["config_digest"]

    resolved_fs_full = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="fast_sanity", layer_override=None
    )
    assert "--hf-overrides" not in resolved_fs_full["flags"]
    assert resolved_fs_full["config_digest"] != resolved_fs["config_digest"]
    assert resolved_fs_full["config_digest"] != resolved_fs_7l["config_digest"]

    # Single-node TP8 decode with MTP
    resolved_decode = ppu_recipe.resolve_recipe(
        recipe, policy_path=decode_policy, tier="fast_sanity"
    )
    assert resolved_decode["ppu_count"] == 8
    assert resolved_decode["preset"] == "decode"
    assert resolved_decode["enabled_features"] == [
        "tool_calling",
        "reasoning",
        "spec_decoding",
    ]
    idx_tp = resolved_decode["flags"].index("--tensor-parallel-size")
    assert resolved_decode["flags"][idx_tp + 1] == "8"
    idx_spec = resolved_decode["flags"].index("--speculative-config")
    assert json.loads(resolved_decode["flags"][idx_spec + 1]) == {
        "method": "mtp",
        "num_speculative_tokens": 2,
    }

    # Multi-node 4x8 DP32+EP decode_ep preset
    resolved_ep = ppu_recipe.resolve_recipe(recipe, preset="decode_ep")
    assert resolved_ep["ppu_count"] == 32
    assert resolved_ep["topology"] == {
        "kind": "multi_node_ep",
        "nodes": 4,
        "devices_per_node": 8,
    }
    assert "--tensor-parallel-size" not in resolved_ep["flags"]
    assert "--enable-expert-parallel" in resolved_ep["flags"]
    assert "{master_addr}" in resolved_ep["flags"]
    assert "{node_rank}" in resolved_ep["flags"]
    assert resolved_ep["server_env"]["HGGC_EMBEDDED_COPY_THRESHOLD"] == "0x10000"

    # Paired Mooncake PD disagg presets (wired to actual pinned MooncakeConnector runtime settings)
    for slot, expected_role, expected_kv_role in (
        ("pd_prefill", "prefill", "kv_producer"),
        ("pd_decode", "decode", "kv_consumer"),
    ):
        res_pd = ppu_recipe.resolve_recipe(recipe, preset=slot)
        assert res_pd["ppu_count"] == 8
        assert res_pd["topology"]["kind"] == "pd_disagg"
        assert res_pd["topology"]["role"] == expected_role
        kv_idx = res_pd["flags"].index("--kv-transfer-config")
        kv_cfg = json.loads(res_pd["flags"][kv_idx + 1])
        assert kv_cfg["kv_connector"] == "MooncakeConnector"
        assert kv_cfg["kv_role"] == expected_kv_role
        assert "kv_port" not in kv_cfg
        assert "kv_ip" not in kv_cfg["kv_connector_extra_config"]
        assert kv_cfg["kv_connector_extra_config"]["device_name"] == "{ib_devices}"
        assert res_pd["server_env"]["VLLM_HOST_IP"] == "{kv_ip}"
        if expected_role == "prefill":
            assert res_pd["server_env"]["VLLM_MOONCAKE_BOOTSTRAP_PORT"] == "{kv_port}"
        else:
            assert "VLLM_MOONCAKE_BOOTSTRAP_PORT" not in res_pd["server_env"]

    # Multi-variant asset and hardware overrides (fp8_channel on 890P, w8a8_int8 on 810E)
    resolved_fp8 = ppu_recipe.resolve_recipe(
        recipe, variant="fp8_channel", preset="prefill"
    )
    assert resolved_fp8["hardware"] == "zhenwu_m890"
    assert resolved_fp8["precision"] == "fp8"
    assert resolved_fp8["ppu_count"] == 8
    assert (
        resolved_fp8["hf_model_id"]
        == "T-HEAD/GLM-5.3-Quant-W-FP8-PerChannel-A-FP8-PerChannel"
    )
    assert resolved_fp8["served_name"] == "glm-5.3-fp8-channel"

    resolved_int8 = ppu_recipe.resolve_recipe(
        recipe, variant="w8a8_int8", preset="prefill"
    )
    assert resolved_int8["hardware"] == "zhenwu_810e"
    assert resolved_int8["precision"] == "w8a8_int8"
    assert resolved_int8["ppu_count"] == 16
    assert (
        resolved_int8["hf_model_id"]
        == "T-HEAD/GLM-5.3-Quant-W-INT8-PerChannel-A-INT8-PerToken"
    )
    idx_mem = resolved_int8["flags"].index("--gpu-memory-utilization")
    assert resolved_int8["flags"][idx_mem + 1] == "0.95"


def test_resolve_dsv4_matches_catalog_and_dpep_preset() -> None:
    recipe = RECIPES_DIR / "deepseek-ai" / "DeepSeek-V4-Flash.yaml"
    policy = POLICY_DIR / "models" / "deepseek-v4-flash.yaml"
    resolved_default = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="model_sanity"
    )
    assert resolved_default["ppu_count"] == 8
    assert resolved_default["flags"] == [
        "--tensor-parallel-size",
        "8",
        "--gpu-memory-utilization",
        "0.9",
        "--max-model-len",
        "16384",
        "--trust-remote-code",
        "--seed",
        "0",
        "--enable-prefix-caching",
    ]

    resolved_dpep = ppu_recipe.resolve_recipe(
        recipe, preset="dp_ep_low_latency", cache="disabled"
    )
    assert resolved_dpep["ppu_count"] == 8
    assert resolved_dpep["server_env"] == {"VLLM_ENGINE_READY_TIMEOUT_S": "6000"}
    assert "--tensor-parallel-size" not in resolved_dpep["flags"]
    assert "--seed" not in resolved_dpep["flags"]
    assert "--data-parallel-size" in resolved_dpep["flags"]
    assert "--no-enable-prefix-caching" in resolved_dpep["flags"]
    assert "--enable-prefix-caching" not in resolved_dpep["flags"]


def test_resolve_minimax_reproduces_the_retired_catalog_case() -> None:
    recipe = RECIPES_DIR / "MiniMaxAI" / "MiniMax-M3.yaml"
    policy = POLICY_DIR / "models" / "minimax-m3.yaml"
    catalog = {
        entry["key"]: entry
        for entry in json.loads((CI_DIR / "ppu_e2e_models.json").read_text())
    }
    legacy = catalog["minimax-m3-fp8"]
    policy_doc = ppu_recipe._load_yaml(policy)
    assert policy_doc["replaces_legacy_cases"] == ["minimax-m3-fp8"]
    # The shared recipe also carries the MXFP4 variant, so the FP8 policy has to
    # pin the default variant explicitly instead of inheriting whatever is first.
    assert policy_doc["variant"] == "default"
    resolved_ms = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="model_sanity"
    )
    assert resolved_ms["variant"] == "default"
    assert resolved_ms["ppu_count"] == legacy["ppu_count"] == 8
    assert resolved_ms["hf_model_id"] == legacy["model_id"]
    assert resolved_ms["golden_model_key"] == legacy["model_key"]
    assert resolved_ms["served_name"] == legacy["served_name"]
    assert resolved_ms["runtime_env"] == {"VLLM_SERVER_DEV_MODE": "1"}
    # The recipe follows the P0 single-node serving baseline (FP8 KV cache,
    # block size 128, M3 tool-call/reasoning parsers), so the legacy argv is no
    # longer reproduced flag by flag. Retirement only requires the TP layout,
    # timeout and cache contract below; the accuracy claim rests on the golden
    # band of the Nightly model_sanity run.
    flags = resolved_ms["flags"]
    assert flags[flags.index("--tensor-parallel-size") + 1] == "8"
    assert flags[flags.index("--distributed-timeout-seconds") + 1] == "3600"
    assert flags[flags.index("--cpu-distributed-timeout-seconds") + 1] == "3600"
    assert "--enable-prefix-caching" in flags
    assert "--no-enable-prefix-caching" not in flags

    golden_path = (
        ROOT
        / "tests"
        / "e2e"
        / "models"
        / "_golden"
        / "accuracy_minimax-m3-fp8_890p.json"
    )
    golden = json.loads(golden_path.read_text())
    model_sanity = policy_doc["tiers"]["model_sanity"]["stages"]
    stages = {stage["id"]: stage for stage in model_sanity}
    assert stages["eval-gsm8k"]["generation"] == golden["generation"]

    resolved_fs = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="fast_sanity"
    )
    assert resolved_fs["ppu_count"] == 8
    assert resolved_fs["dummy_weights"] is True
    assert resolved_fs["layer_override"] == "auto"
    assert "--load-format" in resolved_fs["flags"]
    assert "dummy" in resolved_fs["flags"]
    assert "--hf-overrides" not in resolved_fs["flags"]


@pytest.mark.parametrize(
    (
        "slug",
        "recipe_rel",
        "hf_model_id",
        "golden_key",
        "tool_parser",
        "reasoning_parser",
        "extra_flags",
        "expected_env",
    ),
    [
        (
            "minimax-m3.1-flash",
            "MiniMaxAI/MiniMax-M3.1-Flash.yaml",
            "T-HEAD/MiniMax-M3.1-Flash-Quant-W-FP8-PerChannel-A-FP8-PerToken",
            "minimax-m3.1-flash-fp8",
            "minimax_m3",
            "minimax_m3",
            {"--block-size": "128", "--kv-cache-dtype": "fp8"},
            {},
        ),
        (
            "step-5-preview",
            "stepfun-ai/Step-5-Preview.yaml",
            "T-HEAD/Step-5-Preview-MoE-Quant-W-FP8-PerChannel-A-FP8-PerToken",
            "step-5-preview-fp8",
            "step5",
            "step5",
            {},
            {"VLLM_ENGINE_READY_TIMEOUT_S": "3600"},
        ),
    ],
)
def test_resolve_new_tp8_fp8_recipes_and_golden_contracts(
    slug: str,
    recipe_rel: str,
    hf_model_id: str,
    golden_key: str,
    tool_parser: str,
    reasoning_parser: str,
    extra_flags: dict[str, str],
    expected_env: dict[str, str],
) -> None:
    recipe = RECIPES_DIR / recipe_rel
    policy = POLICY_DIR / "models" / f"{slug}.yaml"
    policy_doc = ppu_recipe._load_yaml(policy)

    resolved_ms = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="model_sanity"
    )
    assert resolved_ms["ppu_count"] == 8
    assert resolved_ms["hf_model_id"] == hf_model_id
    assert resolved_ms["golden_model_key"] == golden_key
    assert resolved_ms["server_env"] == expected_env
    assert resolved_ms["runtime_env"] == {"VLLM_SERVER_DEV_MODE": "1"}

    flags = resolved_ms["flags"]
    assert flags[flags.index("--tensor-parallel-size") + 1] == "8"
    assert flags[flags.index("--gpu-memory-utilization") + 1] == "0.9"
    assert flags[flags.index("--max-model-len") + 1] == "16384"
    assert flags[flags.index("--tool-call-parser") + 1] == tool_parser
    assert flags[flags.index("--reasoning-parser") + 1] == reasoning_parser
    for flag, value in extra_flags.items():
        assert flags[flags.index(flag) + 1] == value
    assert "--enable-auto-tool-choice" in flags
    assert "--enable-prefix-caching" in flags
    assert "--no-enable-prefix-caching" not in flags

    golden_path = (
        ROOT
        / "tests"
        / "e2e"
        / "models"
        / "_golden"
        / f"accuracy_{golden_key}_890p.json"
    )
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    assert golden["model"] == golden_key
    assert golden["checkpoint"] == hf_model_id
    assert golden["checkpoint_tail"] == hf_model_id
    stages = {
        stage["id"]: stage for stage in policy_doc["tiers"]["model_sanity"]["stages"]
    }
    assert stages["eval-gsm8k"]["generation"] == golden["generation"]

    resolved_fs = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="fast_sanity"
    )
    assert resolved_fs["ppu_count"] == 8
    assert resolved_fs["dummy_weights"] is True
    assert resolved_fs["layer_override"] == "auto"
    assert "--load-format" in resolved_fs["flags"]
    assert "dummy" in resolved_fs["flags"]

    for tier in ("fast_sanity", "model_sanity"):
        matrix = ppu_e2e_select.build_recipe_matrix("", tier)["include"]
        assert any(item["key"] == slug and item["ppu_count"] == 8 for item in matrix)


def test_model_sanity_policy_selects_eight_card_dp_ep_preset() -> None:
    base_slug = "deepseek-v4-flash"
    slug = f"{base_slug}-dp-ep"
    legacy_case = "deepseek-v4-mxfp4-dp-ep"
    matrix = ppu_e2e_select.build_recipe_matrix(slug, "model_sanity")
    (entry,) = matrix["include"]
    recipe = ROOT / entry["recipe_path"]
    resolved = ppu_recipe.resolve_recipe(
        recipe,
        policy_path=ROOT / entry["policy_path"],
        tier="model_sanity",
        preset=entry["preset"],
    )
    base = ppu_recipe.resolve_recipe(
        recipe,
        policy_path=POLICY_DIR / "models" / f"{base_slug}.yaml",
        tier="model_sanity",
    )
    legacy = next(
        case
        for case in json.loads((CI_DIR / "ppu_e2e_models.json").read_text())
        if case["key"] == legacy_case
    )
    assert entry["ppu_count"] == resolved["ppu_count"] == 8
    assert entry["preset"] == resolved["preset"] == "dp_ep_low_latency"
    assert entry["config_digest"] == resolved["config_digest"] != base["config_digest"]
    assert set(legacy["server_env"].items()) <= set(resolved["server_env"].items())

    def flags(args):
        return {name: value for name, value, _ in ppu_recipe._parse_flag_entries(args)}

    effective = flags(resolved["flags"])
    assert "--tensor-parallel-size" not in effective
    for key, value in flags(legacy["server_cmd"][9:]).items():
        if key.endswith("-config"):
            assert json.loads(effective[key]) == json.loads(value)
        else:
            assert effective[key] == value
    fast_matrix = ppu_e2e_select.build_recipe_matrix("", "fast_sanity")
    assert slug not in {item["key"] for item in fast_matrix["include"]}


def test_resolve_and_validate_variant_overrides_and_policy_variant(
    tmp_path: Path,
) -> None:
    recipe = RECIPES_DIR / "MiniMaxAI" / "MiniMax-M3.yaml"
    policy = POLICY_DIR / "models" / "minimax-m3-mxfp4.yaml"

    resolved_default = ppu_recipe.resolve_recipe(recipe, cache="enabled")
    assert resolved_default["variant"] == "default"
    assert (
        resolved_default["hf_model_id"]
        == "T-HEAD/MiniMax-M3-Quant-W-FP8-PerChannel-A-FP8-PerToken"
    )
    assert resolved_default["golden_model_key"] == "minimax-m3-fp8"
    assert (
        resolved_default["served_name"]
        == "MiniMax-M3-Quant-W-FP8-PerChannel-A-FP8-PerToken"
    )

    resolved_mxfp4 = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="fast_sanity"
    )
    assert resolved_mxfp4["variant"] == "mxfp4_fp8"
    assert resolved_mxfp4["hf_model_id"] == "T-HEAD/MiniMax-M3-MXFP4-FP8"
    assert resolved_mxfp4["golden_model_key"] == "minimax-m3-mxfp4-fp8"
    assert resolved_mxfp4["served_name"] == "minimax-m3-mxfp4-fp8"
    assert resolved_mxfp4["ppu_count"] == 8
    assert resolved_mxfp4["config_digest"] != resolved_default["config_digest"]

    resolved_dpep = ppu_recipe.resolve_recipe(
        recipe, variant="mxfp4_fp8", preset="dp_ep_low_latency", cache="disabled"
    )
    assert resolved_dpep["hf_model_id"] == "T-HEAD/MiniMax-M3-MXFP4-FP8"
    assert resolved_dpep["ppu_count"] == 8
    assert resolved_dpep["server_env"]["HGGC_EMBEDDED_COPY_THRESHOLD"] == "0x10000"
    assert "--data-parallel-size" in resolved_dpep["flags"]
    assert "--tensor-parallel-size" not in resolved_dpep["flags"]

    matrix = ppu_e2e_select.build_recipe_matrix("minimax-m3-mxfp4", tier="fast_sanity")
    assert len(matrix["include"]) == 1
    entry = matrix["include"][0]
    assert entry["variant"] == "mxfp4_fp8"
    assert entry["model_id"] == "T-HEAD/MiniMax-M3-MXFP4-FP8"
    assert entry["golden_model_key"] == "minimax-m3-mxfp4-fp8"

    # Reject unknown or blank policy variant.
    bad_pol = yaml.safe_load(policy.read_text())
    bad_pol["variant"] = "nonexistent"
    bad_pol_path = tmp_path / "bad_var_pol.yaml"
    bad_pol_path.write_text(yaml.safe_dump(bad_pol), encoding="utf-8")
    with pytest.raises(ppu_recipe.ValidationError, match="unknown variant"):
        ppu_recipe.validate_paths([bad_pol_path], vllm_parser_mode="skip")

    bad_pol["variant"] = "   "
    bad_pol_path.write_text(yaml.safe_dump(bad_pol), encoding="utf-8")
    with pytest.raises(ppu_recipe.ValidationError, match="must be non-empty string"):
        ppu_recipe.validate_paths([bad_pol_path], vllm_parser_mode="skip")

    # Reject invalid variant metadata and sensitive leaks.
    base_rec = yaml.safe_load(recipe.read_text())
    for mutator, match in (
        (
            lambda d: d["variants"]["mxfp4_fp8"].update({"hf_model_id": ""}),
            "hf_model_id must be non-empty string",
        ),
        (
            lambda d: d["variants"]["mxfp4_fp8"].update(
                {"hf_model_id": "T-HEAD/" + "a" * 97}
            ),
            "max 96 chars",
        ),
        (
            lambda d: d["variants"]["mxfp4_fp8"].update({"assets": {"model": "only"}}),
            "missing required key",
        ),
        (
            lambda d: d["variants"]["mxfp4_fp8"].update(
                {"served_name": "/data/secret"}
            ),
            "local_path",
        ),
    ):
        bad_rec = copy.deepcopy(base_rec)
        mutator(bad_rec)
        bad_rec_path = tmp_path / "bad_var_rec.yaml"
        bad_rec_path.write_text(yaml.safe_dump(bad_rec), encoding="utf-8")
        with pytest.raises(ppu_recipe.ValidationError, match=match):
            ppu_recipe.validate_paths([bad_rec_path], vllm_parser_mode="skip")


def test_resolve_deepseek_v41_and_v4_pro_recipes() -> None:
    v41 = ppu_recipe.resolve_recipe(
        RECIPES_DIR / "deepseek-ai" / "DeepSeek-V4.1-Flash.yaml",
        policy_path=POLICY_DIR / "models" / "deepseek-v4.1-flash.yaml",
        tier="fast_sanity",
    )
    assert v41["ppu_count"] == 8
    assert v41["layer_override"] == "auto"
    assert "--hf-overrides" not in v41["flags"]
    assert v41["hf_model_id"].endswith("Engram-MXFP8")
    assert "--engram-config" in v41["flags"]
    assert "--mm-encoder-tp-mode" in v41["flags"]
    assert v41["server_env"]["VLLM_ENGINE_READY_TIMEOUT_S"] == "3600"

    v4p = ppu_recipe.resolve_recipe(
        RECIPES_DIR / "deepseek-ai" / "DeepSeek-V4-Pro.yaml",
        policy_path=POLICY_DIR / "models" / "deepseek-v4-pro.yaml",
        tier="fast_sanity",
    )
    assert v4p["ppu_count"] == 8
    assert v4p["layer_override"] == "auto"
    assert "--kv-cache-dtype" in v4p["flags"]
    assert "--compilation-config" in v4p["flags"]
    assert "--block-size" in v4p["flags"]
    assert "--hf-overrides" not in v4p["flags"]

    v4p_5l = ppu_recipe.resolve_recipe(
        RECIPES_DIR / "deepseek-ai" / "DeepSeek-V4-Pro.yaml",
        policy_path=POLICY_DIR / "models" / "deepseek-v4-pro.yaml",
        tier="fast_sanity",
        layer_override=5,
    )
    assert json.loads(v4p_5l["flags"][v4p_5l["flags"].index("--hf-overrides") + 1]) == {
        "num_hidden_layers": 5
    }


def test_resolve_kimi_k26_matches_catalog_and_sampling_contract() -> None:
    recipe = RECIPES_DIR / "moonshotai" / "Kimi-K2.6.yaml"
    policy = POLICY_DIR / "models" / "kimi-k2.6.yaml"
    catalog = {
        entry["key"]: entry
        for entry in json.loads((CI_DIR / "ppu_e2e_models.json").read_text())
    }
    legacy = catalog["kimi-k2.6-mxfp4-tp8"]
    cat_flags = legacy["server_cmd"][9:]

    resolved_ms = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="model_sanity"
    )
    assert resolved_ms["ppu_count"] == 8
    assert resolved_ms["server_env"] == legacy["server_env"]
    # The executor injects prefix caching from the policy; the recipe itself
    # must not hardcode it, and the resolved command still covers every
    # legacy serve flag.
    assert resolved_ms["flags"][-1] == "--enable-prefix-caching"
    assert set(cat_flags) <= set(resolved_ms["flags"])
    recipe_doc = yaml.safe_load(recipe.read_text())
    assert "--enable-prefix-caching" not in recipe_doc["model"]["base_args"]
    assert "--load-format" not in recipe_doc["model"]["base_args"]
    assert set(recipe_doc["ppu"]["external_env"]) == set(legacy["server_env"])

    resolved_fs = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="fast_sanity"
    )
    assert resolved_fs["layer_override"] == "auto"
    assert "--load-format" in resolved_fs["flags"]
    assert "dummy" in resolved_fs["flags"]
    assert "--hf-overrides" not in resolved_fs["flags"]

    # The eval generation block is the sampling contract the golden gate
    # compares key by key: any drift from the committed 890p golden fails
    # the run before a score is even compared.
    policy_doc = yaml.safe_load(policy.read_text())
    eval_stage = next(
        stage
        for stage in policy_doc["tiers"]["model_sanity"]["stages"]
        if stage["kind"] == "evalscope"
    )
    golden = json.loads(
        (
            ROOT
            / "tests"
            / "e2e"
            / "models"
            / "_golden"
            / "accuracy_kimi_k2_6_mxfp4_890p.json"
        ).read_text()
    )
    assert eval_stage["generation"] == golden["generation"]
    assert resolved_ms["golden_model_key"] == golden["model"]
    assert resolved_ms["hf_model_id"] == golden["checkpoint_tail"]


def test_config_digest_ignores_ephemeral_endpoint_and_tracks_changes() -> None:
    recipe = RECIPES_DIR / "zai-org" / "GLM-5.3.yaml"
    policy = POLICY_DIR / "models" / "glm-5.3.yaml"
    r1 = ppu_recipe.resolve_recipe(
        recipe,
        policy_path=policy,
        tier="model_sanity",
        port=9985,
        served_model_name="a",
    )
    r2 = ppu_recipe.resolve_recipe(
        recipe,
        policy_path=policy,
        tier="model_sanity",
        port=9999,
        served_model_name="b",
    )
    r3 = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="fast_sanity", port=9985, served_model_name="a"
    )
    assert r1["config_digest"] == r2["config_digest"]
    assert r1["config_digest"] != r3["config_digest"]


def test_validate_rejects_unknown_keys_bad_env_json_and_leaks(tmp_path: Path) -> None:
    base_recipe = yaml.safe_load((RECIPES_DIR / "zai-org" / "GLM-5.3.yaml").read_text())

    def _check_bad_recipe(mutator, match: str) -> None:
        bad = copy.deepcopy(base_recipe)
        mutator(bad)
        path = tmp_path / "bad_recipe.yaml"
        path.write_text(yaml.safe_dump(bad), encoding="utf-8")
        with pytest.raises(ppu_recipe.ValidationError, match=match):
            ppu_recipe.validate_paths([path], vllm_parser_mode="skip")

    _check_bad_recipe(lambda d: d.update({"unknown_top": 1}), "unknown key")
    _check_bad_recipe(
        lambda d: d["ppu"].update({"visibility": "internal_restricted"}),
        "internal_restricted",
    )
    _check_bad_recipe(
        lambda d: d["model"]["base_env"].update({"HF_HUB_OFFLINE": "1"}),
        "runtime-mode env",
    )
    _check_bad_recipe(
        lambda d: d["model"]["base_env"].update({"CUSTOM_VAR": "1"}),
        "allowed prefixes",
    )
    _check_bad_recipe(
        lambda d: d["model"]["base_env"].update({"DG_USE_MOE_DYNAMIC_TILE": "1"}),
        "ppu.external_env",
    )
    _check_bad_recipe(
        lambda d: d["model"]["base_args"].extend(["--hf-overrides", "not-json"]),
        "invalid JSON",
    )
    _check_bad_recipe(
        lambda d: d["model"]["base_args"].extend(["--hf-overrides", '{"a": "$(id)"}']),
        "shell substitution",
    )
    _check_bad_recipe(
        lambda d: d.update({"notes": "/nas_aisw/datasets/secret"}),
        "local_path",
    )
    _check_bad_recipe(
        lambda d: d.update({"notes": "host.eng.t-head.cn"}),
        "internal_domain",
    )
    _check_bad_recipe(
        lambda d: d.update({"notes": "10.1.2.3"}),
        "ipv4_literal",
    )
    prohibited = "".join(chr(c) for c in (102, 108, 121, 116, 105, 103, 101, 114))
    _check_bad_recipe(
        lambda d: d.update({"notes": prohibited}),
        "prohibited_token",
    )


def test_validate_policy_guardrails_and_redaction_gate(tmp_path: Path) -> None:
    base_pol = yaml.safe_load((POLICY_DIR / "models" / "glm-5.3.yaml").read_text())

    def _check_bad_pol(mutator, match: str) -> None:
        bad = copy.deepcopy(base_pol)
        mutator(bad)
        path = tmp_path / "bad_pol.yaml"
        path.write_text(yaml.safe_dump(bad), encoding="utf-8")
        with pytest.raises(ppu_recipe.ValidationError, match=match):
            ppu_recipe.validate_paths([path], vllm_parser_mode="skip")

    _check_bad_pol(
        lambda d: d["tiers"]["model_sanity"]["stages"][1].update({"cache": "disabled"}),
        "requires cache='enabled'",
    )
    _check_bad_pol(
        lambda d: d["tiers"]["model_sanity"]["stages"][2].__delitem__("cache_mode"),
        "requires cache_mode",
    )
    _check_bad_pol(
        lambda d: d["tiers"]["model_sanity"]["stages"][2]["input"].update(
            {"range_ratio": 1.0}
        ),
        "range_ratio must be in",
    )
    _check_bad_pol(
        lambda d: d["tiers"]["model_sanity"]["stages"][2].update({"kind": "evalscope"}),
        "duplicate stage kind",
    )
    _check_bad_pol(
        lambda d: d["tiers"]["model_sanity"]["stages"].pop(),
        "missing required stage kind",
    )
    _check_bad_pol(
        lambda d: d["tiers"]["model_sanity"]["stages"][1].update(
            {"dataset": "missing@2026-11"}
        ),
        "referenced dataset definition not found",
    )

    with pytest.raises(ppu_recipe.ValidationError, match="redaction gate blocked"):
        ppu_recipe.redact_and_verify_public_payload({"err": "token ghp_1234567890ab"})
    with pytest.raises(ppu_recipe.ValidationError, match="redaction gate blocked"):
        ppu_recipe.redact_and_verify_public_payload({"path": "/nas_aisw/datasets/foo"})
    ppu_recipe.redact_and_verify_public_payload(
        {"url": "http://127.0.0.1:9985"}, allow_loopback=True
    )


def test_validate_flags_with_parser_rejects_bad_flag_and_type() -> None:
    parser = _sample_vllm_parser()
    with pytest.raises(ppu_recipe.ValidationError, match="unknown vLLM serve flag"):
        ppu_recipe.validate_flags_with_parser(
            parser, ["--nonexistent-flag"], context="test"
        )
    with pytest.raises(ppu_recipe.ValidationError, match="parser rejected args"):
        ppu_recipe.validate_flags_with_parser(
            parser, ["--tensor-parallel-size", "not-an-int"], context="test"
        )


def test_recipe_selector_matrix_and_cli() -> None:
    matrix = ppu_e2e_select.build_recipe_matrix(
        "glm-5.3,deepseek-v4-flash", tier="fast_sanity"
    )
    assert [item["key"] for item in matrix["include"]] == [
        "glm-5.3",
        "deepseek-v4-flash",
    ]
    assert [item["ppu_count"] for item in matrix["include"]] == [4, 8]
    with pytest.raises(ValueError, match="unknown recipe policy key"):
        ppu_e2e_select.build_recipe_matrix("unknown-model")


@pytest.mark.parametrize("claims", [[], "legacy", [""], [" "], [1], ["same", "same"]])
def test_replacement_policy_requires_unique_nonempty_case_ids(claims) -> None:
    policy = yaml.safe_load((POLICY_DIR / "models" / "glm-5.3.yaml").read_text())
    policy["replaces_legacy_cases"] = claims
    with pytest.raises(ValueError, match="nonempty list of unique case IDs"):
        ppu_recipe.validate_policy_structure(policy, context="test policy")


def test_replacement_policy_requires_model_sanity() -> None:
    policy = yaml.safe_load((POLICY_DIR / "models" / "glm-5.3.yaml").read_text())
    del policy["tiers"]["model_sanity"]
    with pytest.raises(ValueError, match="requires tiers.model_sanity"):
        ppu_recipe.validate_policy_structure(policy, context="test policy")


def test_recipe_defaults_include_every_policy_defining_the_requested_tier(
    tmp_path, monkeypatch
) -> None:
    shutil.copytree(ROOT / "tests/ppu", tmp_path / "tests/ppu")
    monkeypatch.setattr(ppu_e2e_select, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ppu_recipe, "REPO_ROOT", tmp_path)
    policy_dir = tmp_path / "tests/ppu/policy/models"
    pilot = yaml.safe_load((policy_dir / "glm-5.3.yaml").read_text())
    for tier in ("fast_sanity", "model_sanity"):
        single_tier = {**pilot, "tiers": {tier: pilot["tiers"][tier]}}
        single_tier.pop("replaces_legacy_cases")
        (policy_dir / f"{tier}-only.yaml").write_text(yaml.safe_dump(single_tier))
    for tier, other in (
        ("fast_sanity", "model_sanity"),
        ("model_sanity", "fast_sanity"),
    ):
        expected = [
            path.stem
            for path in sorted(policy_dir.glob("*.yaml"))
            if tier in yaml.safe_load(path.read_text()).get("tiers", {})
        ]
        assert tier + "-only" in expected
        assert other + "-only" not in expected
        assert [
            entry["key"]
            for entry in ppu_e2e_select.build_recipe_matrix("", tier)["include"]
        ] == expected
        with pytest.raises(ValueError, match=f"{other}-only.*has no tier {tier!r}"):
            ppu_e2e_select.build_recipe_matrix(f"{other}-only", tier)


def test_qwen_fast_sanity_is_selected_at_tp2() -> None:
    matrix = ppu_e2e_select.build_recipe_matrix("", tier="fast_sanity")
    qwen = next(item for item in matrix["include"] if item["key"] == "qwen3.8-27b")
    assert qwen["ppu_count"] == 2
    assert qwen["model_id"] == "Qwen/Qwen3.8-27B"
    nightly = ppu_e2e_select.build_recipe_matrix("", tier="model_sanity")
    assert "qwen3.8-27b" not in {item["key"] for item in nightly["include"]}


@pytest.mark.parametrize("value", [False, {"max_tokens": True}, {"max_tokens": 129}])
def test_fast_sanity_stream_policy_checks_token_budget(value) -> None:
    policy = yaml.safe_load((POLICY_DIR / "models" / "qwen3.8-27b.yaml").read_text())
    policy["tiers"]["fast_sanity"]["probe"]["stream"] = value
    with pytest.raises(ppu_recipe.ValidationError, match=r"fast_sanity\.probe\.stream"):
        ppu_recipe.validate_policy_structure(policy, context="test policy")


def _stream_event(delta, finish=None):
    chunk = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(chunk)}\r\n\r\n".encode()


_CONTENT = _stream_event({"role": "assistant", "content": "42"})
_FINISH = _stream_event({}, finish="stop")
_DONE = b"data: [DONE]\r\n\r\n"


@pytest.fixture
def probe_server():
    calls = []

    def start(stream_body, *, status=200, keep_sending=False):
        stopped = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append((self.path, body))
                if not body.get("stream"):
                    completed = {
                        "choices": [
                            {"message": {"content": "42"}, "finish_reason": "stop"}
                        ],
                        "usage": {"completion_tokens": 1},
                    }
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(json.dumps(completed).encode())
                    return
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.end_headers()
                try:
                    self.wfile.write(stream_body)
                    if keep_sending:
                        while not stopped.wait(0.01):
                            self.wfile.write(b": keepalive\n\n")
                except OSError:
                    pass

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        return server, thread, stopped

    servers = []

    def listen(*args, **kwargs):
        server, thread, stopped = start(*args, **kwargs)
        servers.append((server, thread, stopped))
        return server.server_port, calls

    yield listen
    for server, thread, stopped in servers:
        stopped.set()
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize(
    ("body", "status", "reason"),
    [
        (_CONTENT + _FINISH + _DONE, 200, None),
        (_CONTENT + _FINISH, 200, "missing_done"),
        (_CONTENT, 200, "truncated_stream"),
        (_CONTENT + _DONE, 200, "truncated_stream"),
        (
            _stream_event({"reasoning": "42", "content": "  "}) + _FINISH + _DONE,
            200,
            "empty_stream_content",
        ),
        (b"data: {synthetic-private-response}\n\n", 200, "invalid_stream_chunk"),
        (b": " + b"x" * 65536, 200, "stream_response_too_large"),
        (b"synthetic-private-response", 500, "stream_http_error"),
    ],
)
def test_stream_chat_probe_validates_loopback_http(probe_server, body, status, reason):
    port, calls = probe_server(body, status=status)
    kwargs = {
        "port": port,
        "served_name": "dummy",
        "probe": {
            "prompt": "Say 42",
            "stream": {"max_tokens": 64, "enable_thinking": False},
        },
    }
    if reason is None:
        result = ppu_recipe_runner._send_stream_chat_probe(**kwargs)
        assert result["status"] == "pass"
        assert result["content_chars"] == 2
    else:
        with pytest.raises(ppu_recipe_runner._StreamProbeError, match=f"^{reason}$"):
            ppu_recipe_runner._send_stream_chat_probe(**kwargs)
    assert calls == [
        (
            "/v1/chat/completions",
            {
                "model": "dummy",
                "messages": [{"role": "user", "content": "Say 42"}],
                "stream": True,
                "temperature": 0.0,
                "max_tokens": 64,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    ]


def test_stream_probe_deadline_includes_keepalive_traffic(probe_server):
    port, _ = probe_server(_CONTENT, keep_sending=True)
    started = time.monotonic()
    with pytest.raises(ppu_recipe_runner._StreamProbeError, match="^stream_timeout$"):
        ppu_recipe_runner._send_stream_chat_probe(
            port=port,
            served_name="dummy",
            probe={"prompt": "Say 42", "stream": {"max_tokens": 16}},
            timeout=0.15,
        )
    assert time.monotonic() - started < 2


def test_missing_done_fails_fast_sanity_without_restarting_server(
    tmp_path, monkeypatch, probe_server
):
    port, calls = probe_server(
        _stream_event({"content": "synthetic-private-value"}) + _FINISH
    )
    launches = []
    monkeypatch.setattr(
        ppu_recipe_runner, "_launch_resolved_server", lambda *a, **k: launches.append(a)
    )
    monkeypatch.setattr(ppu_recipe_runner, "_wait_server_ready", lambda *a, **k: 10.0)
    phases = tmp_path / "phase.txt"
    monkeypatch.setenv("PPU_RECIPE_PHASE_FILE", str(phases))

    exit_code = ppu_recipe_runner.main(
        [
            "--recipe",
            str(RECIPES_DIR / "Qwen" / "Qwen3.8-27B.yaml"),
            "--policy",
            str(POLICY_DIR / "models" / "qwen3.8-27b.yaml"),
            "--tier",
            "fast_sanity",
            "--results-dir",
            str(tmp_path),
            "--port",
            str(port),
            "--vllm-parser",
            "skip",
        ]
    )
    assert exit_code == 1
    assert len(launches) == 1
    assert [body.get("stream", False) for _, body in calls] == [False, True]
    assert calls[0][1]["model"] == calls[1][1]["model"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "fail"
    assert summary["attempts"][0]["stream_probe"]["error"] == "missing_done"
    assert summary["timings"]["cold_start_s"] == 10.0
    assert "stream_s" in summary["timings"] and "nonstream_s" in summary["timings"]
    assert phases.read_text().strip() == "missing_done"
    assert "synthetic-private-value" not in json.dumps(summary)
    ppu_recipe.redact_and_verify_public_payload(summary, allow_loopback=True)


def test_runner_fast_sanity_falls_back_from_2l_to_full_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub_dir = tmp_path / "hf_hub"
    snap_dir = (
        hub_dir
        / "models--T-HEAD--GLM-5.3-MoE-Quant-W-MXFP4-A-MXFP4-Dense-Quant-W-FP8-PerChannel-A-FP8-PerToken"
        / "snapshots"
        / "abc123"
    )
    snap_dir.mkdir(parents=True)
    (snap_dir.parent.parent / "refs").mkdir(parents=True)
    (snap_dir.parent.parent / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    (snap_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm5_moe",
                "num_hidden_layers": 78,
                "first_k_dense_replace": 3,
                "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 75,
                "indexer_types": ["full"] * 3 + ["shared"] * 3 + ["full"] * 72,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub_dir))

    calls: list[int | None] = []

    def fake_attempt(*, layer_override, label, **kwargs):
        calls.append(layer_override)
        if layer_override == 7:
            return {
                "mode": label,
                "layer_override": 7,
                "status": "fail",
                "cold_start_s": 12.5,
                "eval_s": 0.0,
                "config_digest": "a" * 64,
                "ppu_count": 4,
                "error": ppu_recipe_runner._sanitize_error_text(
                    "7-layer failed at /nas_aisw/datasets/x"
                ),
            }
        return {
            "mode": label,
            "layer_override": None,
            "status": "pass",
            "cold_start_s": 45.0,
            "eval_s": 1.2,
            "config_digest": "b" * 64,
            "ppu_count": 4,
            "probe": {"status": "pass", "elapsed_s": 1.2},
        }

    monkeypatch.setattr(ppu_recipe_runner, "_attempt_fast_sanity_once", fake_attempt)
    res = ppu_recipe_runner.run_fast_sanity(
        recipe_path=RECIPES_DIR / "zai-org" / "GLM-5.3.yaml",
        policy_path=POLICY_DIR / "models" / "glm-5.3.yaml",
        results_dir=tmp_path / "out",
        parser_mode="skip",
        queue_s=3.5,
    )
    assert calls == [7, None]
    assert res["layer_inference"]["layer_override"] == 7
    assert res["layer_inference"]["reason"] == "inferred_prefix_cover"
    assert res["status"] == "pass"
    assert res["timings"] == {
        "queue_s": 3.5,
        "cold_start_s": 45.0,
        "eval_s": 1.2,
        "bench_s": 0.0,
        "nonstream_s": 1.2,
    }
    assert (tmp_path / "out" / "summary.json").is_file()


def test_runner_fast_sanity_uses_policy_variant_and_null_layer_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str | None, int | None]] = []

    def fake_attempt(*, variant, layer_override, label, **kwargs):
        seen.append((variant, layer_override))
        return {
            "mode": label,
            "layer_override": layer_override,
            "status": "pass",
            "cold_start_s": 30.0,
            "eval_s": 1.0,
            "config_digest": "c" * 64,
            "ppu_count": 8,
            "probe": {"status": "pass", "elapsed_s": 1.0},
        }

    pol_doc = yaml.safe_load(
        (POLICY_DIR / "models" / "minimax-m3-mxfp4.yaml").read_text()
    )
    pol_doc["tiers"]["fast_sanity"]["layer_override"] = None
    pol_path = tmp_path / "minimax-full.yaml"
    pol_path.write_text(yaml.safe_dump(pol_doc), encoding="utf-8")

    monkeypatch.setattr(ppu_recipe_runner, "_attempt_fast_sanity_once", fake_attempt)
    res = ppu_recipe_runner.run_fast_sanity(
        recipe_path=RECIPES_DIR / "MiniMaxAI" / "MiniMax-M3.yaml",
        policy_path=pol_path,
        results_dir=tmp_path / "out",
        parser_mode="skip",
    )
    assert seen == [("mxfp4_fp8", None)]
    assert res["status"] == "pass"
    assert res["variant"] == "mxfp4_fp8"


@pytest.mark.parametrize(
    ("score", "failed_file", "counts", "failed_phase"),
    [
        (0.9689, None, {}, None),
        (0.9, None, {}, None),
        (0.9689, "bench_c16.json", {"completed": 0, "failed": 32}, "measure"),
        (0.9689, "bench_c16.json", {"completed": 31, "failed": 1}, "measure"),
        (0.9689, "bench_c16.json", {"completed": 32}, "measure"),
        (0.9689, "warmup_c16.json", {"completed": 7, "failed": 1}, "warmup"),
    ],
)
def test_runner_model_sanity_end_to_end_with_checked_in_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    score: float,
    failed_file: str | None,
    counts: dict[str, int],
    failed_phase: str | None,
) -> None:
    """Exercise run_model_sanity against checked-in GLM-5.3 recipe/policy and golden."""
    fake_proc = type("FakeProc", (), {"pid": 999999})()
    monkeypatch.setenv("GSM8K_DATASET_PATH", str(tmp_path / "gsm8k_unused"))
    monkeypatch.setattr(
        ppu_recipe_runner, "_launch_resolved_server", lambda *a, **kw: fake_proc
    )
    monkeypatch.setattr(ppu_recipe_runner, "_stop_server", lambda *a, **kw: None)
    monkeypatch.setattr(ppu_recipe_runner, "_wait_server_ready", lambda *a, **kw: 12.34)
    monkeypatch.setattr(
        ppu_recipe_runner,
        "_send_chat_probe",
        lambda **kw: {
            "status": "pass",
            "elapsed_s": 0.4,
            "finish_reason": "stop",
            "content_chars": 2,
            "completion_tokens": 1,
        },
    )
    monkeypatch.setattr(
        ppu_recipe_runner,
        "_drain_and_reset_cache",
        lambda port: {"cache_reset_evidence": "endpoint_ack_only", "http_status": 200},
    )

    class _FakeEvalModule:
        @staticmethod
        def _ensure_launched_server_alive(*a, **kw):
            return None

        @staticmethod
        def run_evalscope(
            served_name,
            port,
            gsm8k_path,
            *,
            generation_config,
            limit,
            work_dir,
            seed,
            eval_batch_size,
            dataset_options,
        ):
            assert (seed, eval_batch_size) == (42, 128)
            assert dataset_options == {
                "subset_list": ["main"],
                "few_shot_num": 4,
                "few_shot_random": False,
                "filters": {"remove_until": "</think>"},
            }
            _write_review(Path(work_dir), _review_fixture()[:2], model=served_name)
            return {"score": score, "num": 1319}

        @staticmethod
        def extract_score(result):
            return float(result["score"]), int(result["num"])

    real_import = importlib.import_module

    def fake_import(name: str):
        if name == "scripts.eval.ppu_gsm8k_eval":
            return _FakeEvalModule
        return real_import(name)

    monkeypatch.setattr(ppu_recipe_runner.importlib, "import_module", fake_import)

    bench_calls: list[dict[str, object]] = []

    def fake_bench(**kw):
        bench_calls.append(kw)
        return {
            **(
                counts
                if kw["result_filename"] == failed_file
                else {"completed": kw["num_prompts"], "failed": 0}
            ),
            "request_throughput": 12.5,
            "output_throughput": 420.0,
            "mean_ttft_ms": 80.0,
            "mean_tpot_ms": 15.0,
        }

    monkeypatch.setattr(ppu_recipe_runner, "_run_vllm_bench", fake_bench)

    out_dir = tmp_path / "ms_out"
    res = ppu_recipe_runner.run_model_sanity(
        recipe_path=RECIPES_DIR / "zai-org" / "GLM-5.3.yaml",
        policy_path=POLICY_DIR / "models" / "glm-5.3.yaml",
        results_dir=out_dir,
        parser_mode="skip",
        queue_s=1.5,
        post_ready_sleep=0,
    )
    acc_ok = score == 0.9689
    assert res["status"] == ("pass" if acc_ok and not failed_phase else "fail")
    assert res["accuracy"]["score"] == score
    assert res["accuracy"]["num"] == 1319
    assert res["accuracy"]["expected_num"] == 1319
    assert res["accuracy"]["minimum_score"] == pytest.approx(0.9489)
    assert res["accuracy"]["maximum_score"] == pytest.approx(0.9889)
    assert res["accuracy"]["within_band"] is acc_ok
    assert res["micro_e2e_passed"] is (failed_phase is None)
    assert res["accuracy"]["per_question"]["count"] == 2
    assert res["accuracy"]["per_question"]["status"] == "baseline_incomplete"
    assert set(res["bench"].keys()) == {"c1", "c16"}
    assert res["bench"]["c16"]["failed_phase"] == failed_phase
    expected_prompts = [2, 8, 8] if failed_phase == "warmup" else [2, 8, 8, 32]
    assert [c["num_prompts"] for c in bench_calls] == expected_prompts
    assert all(c["range_ratio"] == 0.0 for c in bench_calls)
    assert res["timings"]["queue_s"] == 1.5
    assert res["timings"]["cold_start_s"] == 12.34
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "gsm8k_per_question.jsonl").is_file()


def test_stop_server_kills_outliving_process_group_worker(tmp_path: Path) -> None:
    """Verify _stop_server SIGKILLs workers in the pgid even after the leader exits."""
    import os
    import time

    pid_file = tmp_path / "worker.pid"
    leader_code = (
        "import os, pathlib, signal, subprocess, sys, time; "
        "worker_code = 'import os, pathlib, signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
        "time.sleep(60)'; "
        "w = subprocess.Popen([sys.executable, '-c', worker_code, sys.argv[1]]); "
        "signal.signal(signal.SIGTERM, lambda *a: sys.exit(0)); "
        "time.sleep(60)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", leader_code, str(pid_file)],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5.0
        raw_pid = ""
        while time.monotonic() < deadline:
            if pid_file.exists():
                raw_pid = pid_file.read_text(encoding="utf-8").strip()
                if raw_pid:
                    break
            time.sleep(0.05)
        assert raw_pid, "detached worker did not start"
        worker_pid = int(raw_pid)

        ppu_recipe_runner._stop_server(proc, grace_s=0.3)
        assert proc.poll() is not None
        kill_deadline = time.monotonic() + 2.0
        while time.monotonic() < kill_deadline:
            try:
                os.kill(worker_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            pytest.fail(f"worker {worker_pid} survived _stop_server")
    finally:
        if proc.poll() is None:
            proc.kill()


def test_recipe_workflows_follow_security_and_shell_contracts() -> None:
    for wf_name in (
        "_ppu-recipe-fast-sanity.yaml",
        "ppu-recipe-fast-sanity.yaml",
        "_ppu-recipe-model-sanity.yaml",
        "ppu-accuracy-nightly.yaml",
    ):
        wf = yaml.safe_load((ROOT / ".github" / "workflows" / wf_name).read_text())
        assert wf["cache-mode"] == "none"
        assert wf["permissions"] == {"contents": "read", "actions": "read"}
    fast_wf = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "_ppu-recipe-fast-sanity.yaml").read_text()
    )
    fast_recipe = next(
        step
        for step in fast_wf["jobs"]["sanity"]["steps"]
        if step.get("id") == "recipe"
    )
    assert fast_recipe["env"]["TIER"] == "fast_sanity"
    model_wf = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "_ppu-recipe-model-sanity.yaml").read_text()
    )
    model_recipe = next(
        step
        for step in model_wf["jobs"]["sanity"]["steps"]
        if step.get("id") == "recipe"
    )
    assert model_recipe["env"]["TIER"] == "model_sanity"
    assert model_recipe["env"]["PRESET"] == "${{ matrix.model.preset || '' }}"
    pr_wf = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "ppu-build-and-test.yaml").read_text()
    )
    assert (
        pr_wf["jobs"]["recipe-sanity"]["uses"]
        == "./.github/workflows/_ppu-recipe-fast-sanity.yaml"
    )
    nightly_wf = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "ppu-accuracy-nightly.yaml").read_text()
    )
    assert (
        nightly_wf["jobs"]["recipe-model-sanity"]["uses"]
        == "./.github/workflows/_ppu-recipe-model-sanity.yaml"
    )
    script = CI_DIR / "ppu_e2e_recipe_sanity.sh"
    res = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


@pytest.mark.parametrize(
    ("tier", "expected_install"),
    [
        ("model_sanity", True),
        ("fast_sanity", False),
    ],
)
def test_recipe_evalscope_install_preserves_runtime_numpy(
    tmp_path: Path, tier: str, expected_install: bool
) -> None:
    script = (CI_DIR / "ppu_e2e_recipe_sanity.sh").read_text()
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("torch===2.13.0\n")
    python = tmp_path / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1] == '-c' and 'version(\"numpy\")' in sys.argv[2]:\n"
        "    print('1.26.4')\n"
        "elif sys.argv[1] == '-c':\n"
        "    assert sys.argv[2] == 'from evalscope import TaskConfig, run_task'\n"
        "    print('import evalscope')\n"
        "else:\n"
        "    assert sys.argv[1:4] == ['-m', 'pip', 'install']\n"
        "    print(sys.argv[-1], Path(os.environ['PIP_CONSTRAINT']).read_text().split())\n"
    )
    python.chmod(0o755)
    start = script.index('--constraints "${PIP_CONSTRAINT}"')
    start = script.index("\n", start) + 1
    end = script.index('python "${CI_SCRIPT_DIR}/ppu_image_stack.py" verify')
    result = subprocess.run(
        ["bash", "-euc", script[start:end]],
        env={
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "PIP_CONSTRAINT": str(constraints),
            "TIER": tier,
            "ROOT_DIR": str(ROOT),
            "SKIP_WHEEL_INSTALL": "false",
            "vllm_wheel": "vllm.whl",
            "sail_wheel": "sail.whl",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    expected = ["torch===2.13.0"]
    if expected_install:
        expected.extend(["numpy===1.26.4", "pyarrow<26"])
    packages = ["vllm.whl", "sail.whl", str(ROOT / "requirements/ci-runtime.txt")]
    if expected_install:
        packages.append("evalscope==1.14.1")
    expected_output = [f"{package} {expected}" for package in packages]
    if expected_install:
        expected_output.append("import evalscope")
    assert result.stdout.splitlines() == expected_output


def test_prediction_only_file_cannot_supply_baseline(tmp_path: Path) -> None:
    (tmp_path / "predictions.jsonl").write_text(
        json.dumps({"id": "q0", "prediction": "text must not be published"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "per_question.jsonl"
    result = ppu_recipe_runner._extract_per_question_evalscope(
        tmp_path, output, expected_count=1, aggregate_score=1.0, aggregate_num=1
    )
    assert result["count"] == 0
    assert result["status"] == "baseline_incomplete"
    assert output.read_text(encoding="utf-8") == ""


def _review_fixture(run: str = "a") -> list[dict]:
    path = (
        ROOT
        / "tests"
        / "ut"
        / "fixtures"
        / (f"evalscope_1_14_1_gsm8k_main_reviews_{run}.jsonl")
    )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_review(root: Path, rows: list[dict], model: str = "fixture") -> Path:
    path = root / "20261118_102334" / "reviews" / model / "gsm8k_main.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _extract_review(root: Path) -> tuple[dict, list[dict]]:
    report = json.loads(
        (ROOT / "tests/ut/fixtures/evalscope_1_14_1_gsm8k_report_v2.json").read_text()
    )
    primary = next(
        metric
        for metric in report["metrics"]
        if metric["identity"] == report["primary_metric_identity"]
    )
    output = root / "per_question.jsonl"
    result = ppu_recipe_runner._extract_per_question_evalscope(
        root,
        output,
        expected_count=12,
        aggregate_score=primary["score"],
        aggregate_num=primary["num"],
    )
    records = [json.loads(line) for line in output.read_text().splitlines()]
    return result, records


@pytest.mark.parametrize("generation", ["plain answer", "answer\u2028step\u0085result"])
def test_extract_review_uses_only_jsonl_newlines(
    tmp_path: Path, generation: str
) -> None:
    rows = _review_fixture()
    rows[0]["prediction"] = {"content": generation}
    source = _write_review(tmp_path, rows)
    assert generation in source.read_text(encoding="utf-8")

    result, records = _extract_review(tmp_path)

    assert result["status"] == "baseline_established"
    assert result["count"] == result["unique_ids"] == len(rows)
    assert result["unscored"] == 0
    assert records[0] == {"question_id": "6", "correct": True, "score": 1.0}


@pytest.mark.parametrize("prediction_first", [False, True])
def test_extract_real_review_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prediction_first: bool
) -> None:
    monkeypatch.setattr(
        ppu_recipe_runner.importlib.metadata, "version", lambda name: "1.14.1"
    )
    source = _write_review(tmp_path, _review_fixture())
    if prediction_first:
        (tmp_path / "000_predictions.jsonl").write_text(
            json.dumps({"id": "wrong", "prediction": "not a scored review"}) + "\n"
        )
        wrong_subset = source.with_name("gsm8k_other.jsonl")
        wrong_subset.write_text(source.read_text())
        wrong_dataset = source.with_name("aime24_main.jsonl")
        wrong_dataset.write_text(source.read_text())
    result, records = _extract_review(tmp_path)
    assert result["status"] == "baseline_established"
    assert result["reasons"] == []
    assert result["count"] == result["unique_ids"] == 12
    assert result["unscored"] == 0
    assert result["correct_sum"] == 6
    assert result["source_file"] == str(source.relative_to(tmp_path))
    assert result["evalscope_version"] == "1.14.1"
    assert result["dataset"] == "gsm8k"
    assert result["subset"] == "main"
    assert records[0] == {"question_id": "6", "correct": True, "score": 1.0}
    assert all(set(item) == {"question_id", "correct", "score"} for item in records)


@pytest.mark.parametrize("shape", ["accuracy", "acc", "numeric"])
def test_extract_binary_accuracy_shapes(tmp_path: Path, shape: str) -> None:
    rows = _review_fixture()
    for row in rows:
        score = row["sample_score"]["score"]
        value = score["value"]["accuracy"]
        score["value"] = value if shape == "numeric" else {shape: value}
    _write_review(tmp_path, rows)
    result, records = _extract_review(tmp_path)
    assert result["status"] == "baseline_established"
    assert sum(item["correct"] for item in records) == 6


@pytest.mark.parametrize(
    "case", ["prediction", "null", "fractional", "wrong_metric", "error", "malformed"]
)
def test_extract_unscored_rows_are_excluded(tmp_path: Path, case: str) -> None:
    rows = _review_fixture()
    if case == "prediction":
        rows[0].pop("sample_score")
        rows[0]["prediction"] = "not scored"
    elif case == "null":
        rows[0]["sample_score"]["score"]["value"] = None
    elif case == "fractional":
        rows[0]["sample_score"]["score"]["value"] = {"accuracy": 0.75}
    elif case == "wrong_metric":
        rows[0]["sample_score"]["score"]["value"] = {"latency": 1.0}
    elif case == "error":
        rows[0]["sample_score"]["score"]["status"] = "error"
    source = _write_review(tmp_path, rows)
    if case == "malformed":
        source.write_text(source.read_text() + "{broken json\n")
    result, records = _extract_review(tmp_path)
    assert result["status"] == "baseline_incomplete"
    assert result["unscored"] == 1
    assert any("存在未评分题" in reason for reason in result["reasons"])
    assert len(records) == (12 if case == "malformed" else 11)
    assert all(item["correct"] is not None for item in records)


@pytest.mark.parametrize("candidates", [0, 2])
def test_review_source_must_be_unique(tmp_path: Path, candidates: int) -> None:
    for i in range(candidates):
        _write_review(tmp_path, _review_fixture(), model=f"fixture-{i}")
    result, records = _extract_review(tmp_path)
    assert result["status"] == "baseline_incomplete"
    assert result["source_file"] is None
    assert any("评分文件" in reason for reason in result["reasons"])
    assert records == []


def test_review_ids_pair_across_real_run_row_orders(tmp_path: Path) -> None:
    assert _review_fixture("a") != _review_fixture("b")
    paired = []
    for run in ("a", "b"):
        root = tmp_path / run
        _write_review(root, _review_fixture(run))
        result, records = _extract_review(root)
        assert result["status"] == "baseline_established"
        paired.append({item["question_id"]: item["correct"] for item in records})
    assert paired[0] == paired[1]


@pytest.mark.parametrize("missing", [False, True])
def test_review_uses_dataset_index_not_line_number(
    tmp_path: Path, missing: bool
) -> None:
    rows = _review_fixture()
    rows[0]["sample_score"].pop("sample_id")
    if missing:
        rows[0].pop("index")
    _write_review(tmp_path, rows)
    result, records = _extract_review(tmp_path)
    if missing:
        assert result["status"] == "baseline_incomplete"
        assert any("缺少稳定题目 ID" in reason for reason in result["reasons"])
        assert len(records) == 11
    else:
        assert result["status"] == "baseline_established"
        assert records[0]["question_id"] == "6"


def test_review_output_drops_all_text(tmp_path: Path) -> None:
    rows = _review_fixture()
    for row in rows:
        row["messages"] = [{"role": "user", "content": "private question"}]
        row["prediction"] = "private prediction"
        row["sample_score"]["score"]["explanation"] = "private explanation"
    _write_review(tmp_path, rows)
    _, records = _extract_review(tmp_path)
    assert all(set(item) == {"question_id", "correct", "score"} for item in records)
    assert "private" not in (tmp_path / "per_question.jsonl").read_text()


def _baseline_records(count: int = 10) -> list[dict[str, object]]:
    return [
        {"question_id": str(i), "correct": i % 2 == 0, "score": float(i % 2 == 0)}
        for i in range(count)
    ]


def test_per_question_baseline_established() -> None:
    records = _baseline_records()
    original = copy.deepcopy(records)
    result = ppu_recipe_runner.assess_per_question_baseline(
        records, expected_count=10, aggregate_score=0.5, aggregate_num=10
    )
    assert result == {
        "status": "baseline_established",
        "reasons": [],
        "count": 10,
        "unique_ids": 10,
        "unscored": 0,
        "correct_sum": 5,
    }
    assert records == original


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("duplicate", "重复 ID"),
        ("missing_id", "缺少稳定题目 ID"),
        ("empty_id", "缺少稳定题目 ID"),
        ("count", "数量不符"),
        ("unscored", "存在未评分题"),
        ("null", "存在未评分题"),
        ("aggregate_count", "与汇总不一致"),
        ("aggregate_score", "与汇总不一致"),
        ("nonfinite", "与汇总不一致"),
    ],
)
def test_per_question_baseline_incomplete(case: str, reason: str) -> None:
    records = _baseline_records()
    expected_count, aggregate_num, aggregate_score, unscored = 10, 10, 0.5, 0
    if case == "duplicate":
        records[1]["question_id"] = records[0]["question_id"]
    elif case == "missing_id":
        records[0]["question_id"] = None
    elif case == "empty_id":
        records[0]["question_id"] = ""
    elif case == "count":
        records = _baseline_records(1318)
        expected_count, aggregate_num, aggregate_score = 1319, 1318, 0.5
    elif case == "unscored":
        unscored = 1
    elif case == "null":
        records[1]["correct"] = None
    elif case == "aggregate_count":
        aggregate_num = 11
    elif case == "aggregate_score":
        aggregate_score = 0.6
    elif case == "nonfinite":
        aggregate_score = float("nan")
    result = ppu_recipe_runner.assess_per_question_baseline(
        records,
        expected_count=expected_count,
        aggregate_score=aggregate_score,
        aggregate_num=aggregate_num,
        unscored=unscored,
    )
    assert result["status"] == "baseline_incomplete"
    assert any(reason in item for item in result["reasons"])
    if case in {"unscored", "null"}:
        assert result["unscored"] == 1


def test_per_question_baseline_reconciles_rounded_full_report() -> None:
    records = [
        {"question_id": str(i), "correct": i < 1278, "score": float(i < 1278)}
        for i in range(1319)
    ]
    result = ppu_recipe_runner.assess_per_question_baseline(
        records, expected_count=1319, aggregate_score=0.9689, aggregate_num=1319
    )
    assert result["status"] == "baseline_established"
    assert result["correct_sum"] == 1278
    records[0]["correct"] = False
    records[0]["score"] = 0.0
    result = ppu_recipe_runner.assess_per_question_baseline(
        records, expected_count=1319, aggregate_score=0.9689, aggregate_num=1319
    )
    assert abs(1277 / 1319 - 0.9689) < 1 / 1319
    assert result["status"] == "baseline_incomplete"
    assert "与汇总不一致" in result["reasons"]


def test_empty_per_question_baseline_is_incomplete() -> None:
    result = ppu_recipe_runner.assess_per_question_baseline(
        [], expected_count=1319, aggregate_score=0.0, aggregate_num=0
    )
    assert result["status"] == "baseline_incomplete"
    assert result["count"] == result["unique_ids"] == result["correct_sum"] == 0


def test_resolve_recipe_reports_enabled_features() -> None:
    dsv4_recipe = RECIPES_DIR / "deepseek-ai" / "DeepSeek-V4-Flash.yaml"
    dsv4_policy = POLICY_DIR / "models" / "deepseek-v4-flash.yaml"
    assert (
        ppu_recipe.resolve_recipe(
            dsv4_recipe, policy_path=dsv4_policy, tier="model_sanity"
        )["enabled_features"]
        == []
    )
    assert ppu_recipe.resolve_recipe(
        dsv4_recipe, preset="dp_ep_low_latency", cache="enabled"
    )["enabled_features"] == ["tool_calling", "reasoning", "spec_decoding"]

    glm_recipe = RECIPES_DIR / "zai-org" / "GLM-5.3.yaml"
    glm_policy = POLICY_DIR / "models" / "glm-5.3.yaml"
    assert ppu_recipe.resolve_recipe(
        glm_recipe, policy_path=glm_policy, tier="fast_sanity"
    )["enabled_features"] == ["tool_calling", "reasoning"]


def test_checked_in_recipe_matrix_doc_is_up_to_date() -> None:
    expected = ppu_recipe.render_ci_matrix_markdown()
    assert ppu_recipe.DEFAULT_MATRIX_DOC.is_file()
    assert ppu_recipe.DEFAULT_AUTHORING_DOC.is_file()
    actual = ppu_recipe.DEFAULT_MATRIX_DOC.read_text(encoding="utf-8")
    authoring = ppu_recipe.DEFAULT_AUTHORING_DOC.read_text(encoding="utf-8")
    assert actual == expected, (
        "tests/ppu/README.md is out of date; regenerate with: "
        "python3 scripts/ci/ppu_recipe.py render-matrix --output tests/ppu/README.md"
    )
    assert ppu_recipe._scan_text(actual, context="tests/ppu/README.md") == []
    assert (
        ppu_recipe._scan_text(
            authoring, context="tests/ppu/AUTHORING.md", allow_loopback=True
        )
        == []
    )
    assert "[AUTHORING.md](AUTHORING.md)" in actual
    assert "## PR CI (`fast_sanity`)" in actual
    assert "## Nightly CI — Recipe (`model_sanity`)" in actual
    assert "## Registered Serving Presets (Benchmark & Disagg Catalog)" in actual
    assert "## Nightly CI — Remaining Legacy Accuracy Cases" in actual
    assert "`default` (none)" in actual
    assert "`tool_calling`, `reasoning`" in actual
    fast_section = actual.split("## PR CI (`fast_sanity`)", 1)[1].split(
        "## Nightly CI — Recipe (`model_sanity`)", 1
    )[0]
    nightly_section = actual.split("## Nightly CI — Recipe (`model_sanity`)", 1)[
        1
    ].split("## Registered Serving Presets (Benchmark & Disagg Catalog)", 1)[0]
    catalog_section = actual.split(
        "## Registered Serving Presets (Benchmark & Disagg Catalog)", 1
    )[1].split("## Nightly CI — Remaining Legacy Accuracy Cases", 1)[0]
    assert "`glm-5.3-decode`" in fast_section
    assert "`glm-5.3-decode`" not in nightly_section
    for token in (
        "`fp8_channel`",
        "`w8a8_int8`",
        "`decode_ep`",
        "`pd_prefill`",
        "`pd_decode`",
        "`multi_node_ep` (4x8)",
        "`pd_disagg` (prefill)",
        "`pd_disagg` (decode)",
    ):
        assert token in catalog_section
    legacy_section = actual.split("## Nightly CI — Remaining Legacy Accuracy Cases", 1)[
        1
    ]
    assert "| `deepseek-v4-mxfp4` |" not in legacy_section
    assert "| `glm-5.3-mxfp4-tp4` |" not in legacy_section
    assert "| `deepseek-v4-mxfp4-dp-ep` |" not in legacy_section
    assert "| `deepseek-v4-int8-dp-ep` |" in legacy_section


def test_render_matrix_cli_check_passes_and_detects_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert ppu_recipe.main(["render-matrix", "--check"]) == 0

    missing = tmp_path / "missing.md"
    assert ppu_recipe.main(["render-matrix", "--output", str(missing), "--check"]) == 1
    assert "matrix doc missing" in capsys.readouterr().err

    target = tmp_path / "README.md"
    target.write_text("# stale\n", encoding="utf-8")
    assert ppu_recipe.main(["render-matrix", "--output", str(target), "--check"]) == 1
    assert "matrix doc out of date" in capsys.readouterr().err

    assert ppu_recipe.main(["render-matrix", "--output", str(target)]) == 0
    assert ppu_recipe.main(["render-matrix", "--output", str(target), "--check"]) == 0
    assert target.read_text(encoding="utf-8") == ppu_recipe.render_ci_matrix_markdown()


@pytest.mark.requires_runtime
def test_runtime_vllm_parser_validates_checked_in_recipes() -> None:
    pytest.importorskip("vllm")
    files = _yaml_files()
    report = ppu_recipe.validate_paths(files, vllm_parser_mode="required")
    assert report["parser_checked"] is True
    assert report["checked_flag_sets"] >= 10


def _write_checkpoint(
    root: Path,
    config: dict,
    *,
    weight_keys: list[str] | None = None,
    single_safetensors: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if weight_keys is not None:
        if single_safetensors:
            header = {
                key: {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}
                for key in weight_keys
            }
            header_bytes = json.dumps(header).encode("utf-8")
            (root / "model.safetensors").write_bytes(
                len(header_bytes).to_bytes(8, "little") + header_bytes + b"\x00\x00"
            )
        else:
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {"weight_map": {k: "shard.safetensors" for k in weight_keys}}
                ),
                encoding="utf-8",
            )
    return root


@pytest.mark.parametrize(
    (
        "case_name",
        "config",
        "weight_keys",
        "flags",
        "expected_layers",
        "expected_reason",
    ),
    [
        (
            "homogeneous_dense_with_binary_safetensors",
            {"model_type": "llama", "num_hidden_layers": 32},
            [f"model.layers.{i}.mlp.down_proj.weight" for i in range(32)],
            [],
            2,
            "inferred_prefix_cover",
        ),
        (
            "first_dense_then_moe_kimi_style",
            {
                "model_type": "kimi_moe",
                "num_hidden_layers": 61,
                "first_k_dense_replace": 1,
            },
            ["model.layers.0.mlp.down_proj.weight"]
            + [f"model.layers.{i}.mlp.experts.0.w1.weight" for i in range(1, 61)],
            [],
            2,
            "inferred_prefix_cover",
        ),
        (
            "vlm_vision_tower_excluded_minimax_style",
            {
                "model_type": "minimax_m3_vl",
                "vision_config": {"num_hidden_layers": 32},
                "text_config": {
                    "model_type": "minimax_m3",
                    "num_hidden_layers": 60,
                    "moe_layer_freq": [0, 0, 0] + [1] * 57,
                },
            },
            [
                f"vision_tower.encoder.layers.{i}.self_attn.q_proj.weight"
                for i in range(32)
            ]
            + [f"language_model.model.layers.{i}.mlp.fc1.weight" for i in range(3)]
            + [
                f"language_model.model.layers.{i}.block_sparse_moe.experts.{e}.w1.weight"
                for i in range(3, 60)
                for e in (0, 1)
            ],
            [],
            4,
            "inferred_prefix_cover",
        ),
        (
            "hybrid_linear_full_attention_with_ple_qwen_style",
            {
                "model_type": "qwen3_5",
                "text_config": {
                    "num_hidden_layers": 48,
                    "full_attention_interval": 4,
                    "ple_layer_ids": [2],
                    "layer_types": (
                        [
                            "linear_attention",
                            "linear_attention",
                            "linear_attention",
                            "full_attention",
                        ]
                        * 12
                    ),
                },
            },
            None,
            [],
            4,
            "inferred_prefix_cover",
        ),
        (
            "mtp_padded_compress_ratios_dsv4_flash_style",
            {
                "model_type": "deepseek_v4",
                "num_hidden_layers": 43,
                "num_nextn_predict_layers": 3,
                "compress_ratios": [0, 0, 4, 128] + [4, 128] * 21,
            },
            ["model.layers.0.mlp.down_proj.weight"]
            + ["model.layers.1.mlp.experts.0.w1.weight"]
            + [
                "model.layers.2.mlp.experts.0.w1.weight",
                "model.layers.2.self_attn.compressor.weight",
                "model.layers.2.engram.weight",
            ]
            + [f"model.layers.{i}.mlp.experts.0.w1.weight" for i in range(3, 44)]
            + [f"model.layers.{i}.self_attn.compressor.weight" for i in range(3, 44)],
            [],
            5,
            "inferred_prefix_cover",
        ),
        (
            "dense_linear_sparse_dsa_glm_flash_style",
            {
                "model_type": "glm5_moe_lite",
                "num_hidden_layers": 45,
                "first_k_dense_replace": 3,
                "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 42,
                "layer_types": ["linear_attention"] * 4
                + ["deepseek_sparse_attention"] * 41,
            },
            None,
            [],
            5,
            "inferred_prefix_cover",
        ),
        (
            "dense_shared_full_indexer_glm53_style",
            {
                "model_type": "glm5_moe",
                "num_hidden_layers": 78,
                "first_k_dense_replace": 3,
                "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 75,
                "indexer_types": ["full"] * 3 + ["shared"] * 3 + ["full"] * 72,
            },
            None,
            [],
            7,
            "inferred_prefix_cover",
        ),
        (
            "segmented_deepseek_v41_uses_full_layers",
            {"model_type": "deepseek_v41", "num_hidden_layers": 40},
            [f"model.layers.{i}.self_attn.q_proj.weight" for i in range(40)],
            [],
            None,
            "segmented_architecture",
        ),
        (
            "unknown_segmented_20_plus_20_exceeds_cap_uses_full",
            {"model_type": "custom_hybrid_enc_dec", "num_hidden_layers": 40},
            [f"model.layers.{i}.encoder_attn.q_proj.weight" for i in range(20)]
            + [f"model.layers.{i}.decoder_attn.q_proj.weight" for i in range(20, 40)],
            [],
            None,
            "exceeds_override_cap",
        ),
        (
            "tail_layer_specialization_exceeds_cap_uses_full",
            {
                "model_type": "custom_moe",
                "num_hidden_layers": 45,
                "moe_layers_enum": "3..44",
                "swiglu_limits": [7.0] * 43 + [None, None],
            },
            None,
            [],
            None,
            "exceeds_override_cap",
        ),
        (
            "online_quantization_flag_uses_full_layers",
            {"model_type": "qwen3_5", "num_hidden_layers": 64},
            [f"model.layers.{i}.mlp.down_proj.weight" for i in range(64)],
            ["--quantization", "w8a8_dynamic_token_symmetric_channel"],
            None,
            "online_quantization",
        ),
        (
            "auto_map_without_weights_or_config_schedule_fails_safe_to_full",
            {
                "model_type": "opaque_custom",
                "num_hidden_layers": 32,
                "auto_map": {
                    "AutoModelForCausalLM": "modeling_opaque.OpaqueForCausalLM"
                },
            },
            None,
            [],
            None,
            "missing_weight_metadata",
        ),
        (
            "compressed_tensors_tail_module_regex_ignore_uses_full",
            {
                "model_type": "llama",
                "num_hidden_layers": 32,
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "ignore": [r"re:^model\.layers\.31\.mlp\.down_proj$"],
                },
            },
            None,
            ["--load-format", "dummy"],
            None,
            "exceeds_override_cap",
        ),
        (
            "compressed_tensors_prefix_module_regex_ignore_selects_3l",
            {
                "model_type": "llama",
                "num_hidden_layers": 32,
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "ignore": [r"re:^model\.layers\.(0|1)\.mlp\.down_proj$"],
                },
            },
            None,
            ["--load-format", "dummy"],
            3,
            "inferred_prefix_cover",
        ),
        (
            "quark_tail_exclude_and_layer_quant_config_uses_full",
            {
                "model_type": "deepseek_v4",
                "num_hidden_layers": 32,
                "quantization_config": {
                    "quant_method": "quark",
                    "exclude": ["layers.31.ffn.shared_experts"],
                },
            },
            None,
            ["--load-format", "dummy"],
            None,
            "exceeds_override_cap",
        ),
        (
            "quark_layer_quant_config_regex_tail_uses_full",
            {
                "model_type": "deepseek_v4",
                "num_hidden_layers": 32,
                "quantization_config": {
                    "quant_method": "quark",
                    "layer_quant_config": {
                        r"re:layers\.31\.ffn\.shared_experts\.w1": {
                            "weight": {"dtype": "bf16"}
                        }
                    },
                },
            },
            None,
            ["--load-format", "dummy"],
            None,
            "exceeds_override_cap",
        ),
        (
            "compressed_tensors_config_groups_tail_target_uses_full",
            {
                "model_type": "llama",
                "num_hidden_layers": 32,
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "config_groups": {
                        "group_tail": {
                            "targets": [r"re:^model\.layers\.31\.self_attn\.q_proj$"],
                            "weights": {"num_bits": 16},
                        }
                    },
                },
            },
            None,
            ["--load-format", "dummy"],
            None,
            "exceeds_override_cap",
        ),
        (
            "unresolved_layer_specific_quant_regex_fails_safe_to_full",
            {
                "model_type": "llama",
                "num_hidden_layers": 32,
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "ignore": [r"re:^model\.layers\.31\.custom_proprietary_proj$"],
                },
            },
            None,
            ["--load-format", "dummy"],
            None,
            "unresolved_quantization_rule",
        ),
        (
            "uniform_router_and_lm_head_quant_regex_control_selects_2l",
            {
                "model_type": "llama",
                "num_hidden_layers": 32,
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "ignore": [r"re:.*lm_head$", r"re:.*\.mlp\.gate$"],
                },
            },
            None,
            ["--load-format", "dummy"],
            2,
            "inferred_prefix_cover",
        ),
    ],
)
def test_infer_fast_sanity_layer_override_archetypes(
    tmp_path: Path,
    case_name: str,
    config: dict,
    weight_keys: list[str] | None,
    flags: list[str],
    expected_layers: int | None,
    expected_reason: str,
) -> None:
    ckpt_dir = _write_checkpoint(
        tmp_path / case_name,
        config,
        weight_keys=weight_keys,
        single_safetensors=(case_name == "homogeneous_dense_with_binary_safetensors"),
    )
    info = ppu_recipe.infer_fast_sanity_layer_override(ckpt_dir, flags=flags)
    assert info["layer_override"] == expected_layers
    assert info["reason"] == expected_reason
    assert str(tmp_path) not in json.dumps(info)
    ppu_recipe.redact_and_verify_public_payload(info, allow_loopback=True)


@pytest.mark.parametrize(
    "bad_override",
    [0, -2, True, False, 2.5, "bogus", "2"],
)
def test_validate_policy_rejects_invalid_layer_override(bad_override: object) -> None:
    policy = yaml.safe_load((POLICY_DIR / "models" / "glm-5.3.yaml").read_text())
    policy["tiers"]["fast_sanity"]["layer_override"] = bad_override
    with pytest.raises(
        ppu_recipe.ValidationError, match=r"fast_sanity\.layer_override"
    ):
        ppu_recipe.validate_policy_structure(policy, context="test policy")


def test_resolve_cli_supports_auto_int_and_full_layer_override(
    capsys: pytest.CaptureFixture[str],
) -> None:
    recipe = str(RECIPES_DIR / "zai-org" / "GLM-5.3.yaml")
    policy = str(POLICY_DIR / "models" / "glm-5.3.yaml")
    for cli_val, expected in (("auto", "auto"), ("5", 5), ("full", None)):
        assert (
            ppu_recipe.main(
                [
                    "resolve",
                    "--recipe",
                    recipe,
                    "--policy",
                    policy,
                    "--tier",
                    "fast_sanity",
                    "--layer-override",
                    cli_val,
                ]
            )
            == 0
        )
        out = json.loads(capsys.readouterr().out)
        assert out["layer_override"] == expected
        if type(expected) is int:
            idx = out["flags"].index("--hf-overrides")
            assert json.loads(out["flags"][idx + 1]) == {"num_hidden_layers": expected}
        else:
            assert "--hf-overrides" not in out["flags"]


def test_resolve_and_runner_emit_nested_hf_overrides_for_nested_text_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub_cache = tmp_path / "hub_cache"
    rev = "3" * 40
    qwen_snap = hub_cache / "models--Qwen--Qwen3.8-27B" / "snapshots" / rev
    (qwen_snap.parent.parent / "refs").mkdir(parents=True)
    (qwen_snap.parent.parent / "refs" / "main").write_text(f"{rev}\n", encoding="utf-8")
    _write_checkpoint(
        qwen_snap,
        {
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "num_attention_heads": 24,
                "num_hidden_layers": 64,
                "full_attention_interval": 4,
                "layer_types": (
                    [
                        "linear_attention",
                        "linear_attention",
                        "linear_attention",
                        "full_attention",
                    ]
                    * 16
                ),
            },
        },
    )
    monkeypatch.setenv("HF_HUB_CACHE", str(hub_cache))

    qwen_recipe = RECIPES_DIR / "Qwen" / "Qwen3.8-27B.yaml"
    qwen_policy = POLICY_DIR / "models" / "qwen3.8-27b.yaml"
    resolved = ppu_recipe.resolve_recipe(
        qwen_recipe, policy_path=qwen_policy, tier="fast_sanity", layer_override=4
    )
    ov_idx = resolved["argv"].index("--hf-overrides")
    assert json.loads(resolved["argv"][ov_idx + 1]) == {
        "text_config": {"num_hidden_layers": 4}
    }

    launched_calls: list[tuple[list[str], Path | None]] = []
    fake_proc = type("FakeProc", (), {"pid": 999999})()
    monkeypatch.setattr(
        ppu_recipe_runner,
        "_launch_resolved_server",
        lambda res, **kw: (
            launched_calls.append((list(res["argv"]), kw.get("model_dir"))),
            fake_proc,
        )[1],
    )
    monkeypatch.setattr(ppu_recipe_runner, "_stop_server", lambda *a, **kw: None)
    monkeypatch.setattr(ppu_recipe_runner, "_wait_server_ready", lambda *a, **kw: 5.0)
    monkeypatch.setattr(
        ppu_recipe_runner,
        "_send_chat_probe",
        lambda **kw: {
            "status": "pass",
            "elapsed_s": 0.2,
            "finish_reason": "stop",
            "content_chars": 2,
            "completion_tokens": 1,
        },
    )
    monkeypatch.setattr(
        ppu_recipe_runner,
        "_send_stream_chat_probe",
        lambda **kw: {"status": "pass", "elapsed_s": 0.15, "content_chars": 2},
    )
    auto_qwen_policy = tmp_path / "qwen-auto.yaml"
    pol_doc = yaml.safe_load(qwen_policy.read_text(encoding="utf-8"))
    pol_doc["tiers"]["fast_sanity"]["layer_override"] = "auto"
    auto_qwen_policy.write_text(yaml.safe_dump(pol_doc), encoding="utf-8")

    summary = ppu_recipe_runner.run_fast_sanity(
        recipe_path=qwen_recipe,
        policy_path=auto_qwen_policy,
        results_dir=tmp_path / "qwen_out",
        parser_mode="skip",
    )
    assert summary["status"] == "pass"
    assert summary["layer_inference"]["layer_override"] == 4
    assert len(launched_calls) == 1
    launched_argv, launched_model_dir = launched_calls[0]
    assert launched_model_dir == qwen_snap
    run_ov_idx = launched_argv.index("--hf-overrides")
    assert json.loads(launched_argv[run_ov_idx + 1]) == {
        "text_config": {"num_hidden_layers": 4}
    }

    # Nested llm_config, thinker_config.text_config, dual root+text, and text-only control
    llm_ckpt = _write_checkpoint(
        tmp_path / "llm_vlm",
        {"model_type": "internvl_chat", "llm_config": {"num_hidden_layers": 32}},
    )
    assert ppu_recipe.build_layer_hf_overrides(3, llm_ckpt) == {
        "llm_config": {"num_hidden_layers": 3}
    }
    assert ppu_recipe.infer_fast_sanity_layer_override(llm_ckpt)["layer_override"] == 2

    omni_ckpt = _write_checkpoint(
        tmp_path / "omni_vlm",
        {
            "model_type": "qwen2_5_omni",
            "thinker_config": {"text_config": {"num_hidden_layers": 28}},
        },
    )
    assert ppu_recipe.build_layer_hf_overrides(4, omni_ckpt) == {
        "thinker_config": {"text_config": {"num_hidden_layers": 4}}
    }

    dual_ckpt = _write_checkpoint(
        tmp_path / "dual_vlm",
        {
            "model_type": "custom_vlm",
            "num_hidden_layers": 40,
            "text_config": {"num_hidden_layers": 40},
        },
    )
    assert ppu_recipe.build_layer_hf_overrides(4, dual_ckpt) == {
        "num_hidden_layers": 4,
        "text_config": {"num_hidden_layers": 4},
    }

    flat_ckpt = _write_checkpoint(
        tmp_path / "flat_text",
        {"model_type": "llama", "num_hidden_layers": 64},
    )
    assert ppu_recipe.build_layer_hf_overrides(4, flat_ckpt) == {"num_hidden_layers": 4}


def test_resolve_local_checkpoint_dir_honors_hf_hub_cache_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hf_home = tmp_path / "hf-home"
    hub_cache = tmp_path / "configured-hub-cache"
    model_id = "Example/LayerCoverage"
    for root, rev, dense_k in (
        (hf_home / "hub", "a" * 40, 1),
        (hub_cache, "b" * 40, 3),
    ):
        repo_dir = root / "models--Example--LayerCoverage"
        (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
        (repo_dir / "refs" / "main").write_text(rev, encoding="utf-8")
        _write_checkpoint(
            repo_dir / "snapshots" / rev,
            {
                "model_type": "llama",
                "num_hidden_layers": 32,
                "first_k_dense_replace": dense_k,
            },
        )

    both_env = {
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hub_cache),
        "HF_HUB_OFFLINE": "1",
    }
    selected = ppu_recipe.resolve_local_checkpoint_dir(model_id, env=both_env)
    assert selected == hub_cache / "models--Example--LayerCoverage" / "snapshots" / (
        "b" * 40
    )
    assert ppu_recipe.infer_fast_sanity_layer_override(selected)["layer_override"] == 4

    home_only = ppu_recipe.resolve_local_checkpoint_dir(
        model_id, env={"HF_HOME": str(hf_home)}
    )
    assert home_only == (
        hf_home / "hub" / "models--Example--LayerCoverage" / "snapshots" / ("a" * 40)
    )
    assert ppu_recipe.infer_fast_sanity_layer_override(home_only)["layer_override"] == 2

    # Explicit HF_HUB_CACHE must not silently fall back to stale HF_HOME/hub
    empty_hub = tmp_path / "empty-hub-cache"
    empty_hub.mkdir()
    assert (
        ppu_recipe.resolve_local_checkpoint_dir(
            model_id,
            env={"HF_HOME": str(hf_home), "HF_HUB_CACHE": str(empty_hub)},
        )
        is None
    )


def test_default_hub_cache_resolves_nested_overrides_without_hf_env_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in (
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HOME",
        "XDG_CACHE_HOME",
        "HOME",
    ):
        monkeypatch.delenv(var, raising=False)

    virtual_home = tmp_path / "virtual-home"
    rev = "a" * 40
    qwen_snap = (
        virtual_home
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--Qwen--Qwen3.8-27B"
        / "snapshots"
        / rev
    )
    (qwen_snap.parent.parent / "refs").mkdir(parents=True)
    (qwen_snap.parent.parent / "refs" / "main").write_text(f"{rev}\n", encoding="utf-8")
    _write_checkpoint(
        qwen_snap,
        {
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "num_attention_heads": 24,
                "num_hidden_layers": 64,
                "full_attention_interval": 4,
            },
        },
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: virtual_home))

    assert ppu_recipe.resolve_local_checkpoint_dir("Qwen/Qwen3.8-27B", env={}) == (
        qwen_snap
    )
    qwen_recipe = RECIPES_DIR / "Qwen" / "Qwen3.8-27B.yaml"
    qwen_policy = POLICY_DIR / "models" / "qwen3.8-27b.yaml"
    resolved = ppu_recipe.resolve_recipe(
        qwen_recipe, policy_path=qwen_policy, tier="fast_sanity", layer_override=4
    )
    ov_idx = resolved["argv"].index("--hf-overrides")
    assert json.loads(resolved["argv"][ov_idx + 1]) == {
        "text_config": {"num_hidden_layers": 4}
    }

    # XDG_CACHE_HOME overrides HOME/.cache when HF_HOME / HF_HUB_CACHE are unset
    xdg_root = tmp_path / "xdg-cache"
    rev_xdg = "c" * 40
    xdg_snap = (
        xdg_root
        / "huggingface"
        / "hub"
        / "models--Qwen--Qwen3.8-27B"
        / "snapshots"
        / rev_xdg
    )
    (xdg_snap.parent.parent / "refs").mkdir(parents=True)
    (xdg_snap.parent.parent / "refs" / "main").write_text(
        f"{rev_xdg}\n", encoding="utf-8"
    )
    _write_checkpoint(
        xdg_snap,
        {
            "model_type": "qwen3_5",
            "text_config": {"num_hidden_layers": 64, "full_attention_interval": 4},
        },
    )
    assert (
        ppu_recipe.resolve_local_checkpoint_dir(
            "Qwen/Qwen3.8-27B",
            env={"HOME": str(virtual_home), "XDG_CACHE_HOME": str(xdg_root)},
        )
        == xdg_snap
    )
    assert (
        ppu_recipe.resolve_local_checkpoint_dir(
            "Qwen/Qwen3.8-27B",
            env={"HOME": str(virtual_home), "HF_HOME": "~/custom-empty-hf"},
        )
        is None
    )


def test_short_layer_dummy_empty_stream_content_falls_back_to_full_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub_cache = tmp_path / "hub"
    rev = "d" * 40
    snap = hub_cache / "models--Qwen--Qwen3.8-27B" / "snapshots" / rev
    (snap.parent.parent / "refs").mkdir(parents=True)
    (snap.parent.parent / "refs" / "main").write_text(f"{rev}\n", encoding="utf-8")
    _write_checkpoint(
        snap,
        {
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "num_attention_heads": 24,
                "num_hidden_layers": 64,
                "full_attention_interval": 4,
            },
        },
    )
    monkeypatch.setenv("HF_HUB_CACHE", str(hub_cache))

    qwen_recipe = RECIPES_DIR / "Qwen" / "Qwen3.8-27B.yaml"
    pol_doc = yaml.safe_load(
        (POLICY_DIR / "models" / "qwen3.8-27b.yaml").read_text(encoding="utf-8")
    )
    assert pol_doc["tiers"]["fast_sanity"]["layer_override"] is None
    pol_doc["tiers"]["fast_sanity"]["layer_override"] = "auto"
    auto_pol = tmp_path / "qwen-auto.yaml"
    auto_pol.write_text(yaml.safe_dump(pol_doc), encoding="utf-8")

    # 1) Short-stack dummy empty_stream_content falls back to full layers and passes
    seen_layers: list[int | None] = []

    def attempt_recover(*, layer_override, label, **kw):
        seen_layers.append(layer_override)
        if layer_override == 4:
            return {
                "mode": label,
                "layer_override": 4,
                "status": "fail",
                "error": "empty_stream_content",
                "cold_start_s": 294.0,
                "eval_s": 0.44,
                "config_digest": "a" * 64,
                "ppu_count": 2,
                "probe": {"status": "pass", "elapsed_s": 0.29},
                "stream_probe": {
                    "status": "fail",
                    "error": "empty_stream_content",
                    "elapsed_s": 0.15,
                },
            }
        return {
            "mode": label,
            "layer_override": None,
            "status": "pass",
            "cold_start_s": 303.0,
            "eval_s": 1.77,
            "config_digest": "b" * 64,
            "ppu_count": 2,
            "probe": {"status": "pass", "elapsed_s": 0.56},
            "stream_probe": {"status": "pass", "elapsed_s": 1.21, "content_chars": 64},
        }

    monkeypatch.setattr(ppu_recipe_runner, "_attempt_fast_sanity_once", attempt_recover)
    res_ok = ppu_recipe_runner.run_fast_sanity(
        recipe_path=qwen_recipe,
        policy_path=auto_pol,
        results_dir=tmp_path / "out_ok",
        parser_mode="skip",
    )
    assert seen_layers == [4, None]
    assert res_ok["status"] == "pass"
    assert res_ok["timings"]["cold_start_s"] == 303.0
    assert res_ok["timings"]["stream_s"] == 1.21

    # 2) Control: full-layer empty_stream_content still fails and records phase
    phase_file = tmp_path / "phase.txt"
    monkeypatch.setenv("PPU_RECIPE_PHASE_FILE", str(phase_file))

    def attempt_both_empty(*, layer_override, label, **kw):
        return {
            "mode": label,
            "layer_override": layer_override,
            "status": "fail",
            "error": "empty_stream_content",
            "cold_start_s": 100.0,
            "eval_s": 0.3,
            "config_digest": "c" * 64,
            "ppu_count": 2,
            "probe": {"status": "pass", "elapsed_s": 0.2},
            "stream_probe": {
                "status": "fail",
                "error": "empty_stream_content",
                "elapsed_s": 0.1,
            },
        }

    monkeypatch.setattr(
        ppu_recipe_runner, "_attempt_fast_sanity_once", attempt_both_empty
    )
    res_fail = ppu_recipe_runner.run_fast_sanity(
        recipe_path=qwen_recipe,
        policy_path=auto_pol,
        results_dir=tmp_path / "out_fail",
        parser_mode="skip",
    )
    assert res_fail["status"] == "fail"
    assert len(res_fail["attempts"]) == 2
    assert phase_file.read_text(encoding="utf-8").strip() == "empty_stream_content"

    # 3) Control: framing error (missing_done) on short stack fails fast without restart
    framing_layers: list[int | None] = []

    def attempt_missing_done(*, layer_override, label, **kw):
        framing_layers.append(layer_override)
        return {
            "mode": label,
            "layer_override": layer_override,
            "status": "fail",
            "error": "missing_done",
            "cold_start_s": 100.0,
            "eval_s": 0.3,
            "config_digest": "d" * 64,
            "ppu_count": 2,
            "probe": {"status": "pass", "elapsed_s": 0.2},
            "stream_probe": {
                "status": "fail",
                "error": "missing_done",
                "elapsed_s": 0.1,
            },
        }

    monkeypatch.setattr(
        ppu_recipe_runner, "_attempt_fast_sanity_once", attempt_missing_done
    )
    res_framing = ppu_recipe_runner.run_fast_sanity(
        recipe_path=qwen_recipe,
        policy_path=auto_pol,
        results_dir=tmp_path / "out_framing",
        parser_mode="skip",
    )
    assert framing_layers == [4]
    assert res_framing["status"] == "fail"
    assert phase_file.read_text(encoding="utf-8").strip() == "missing_done"

    # 4) Terminal attribution: when short-stack empty_stream_content falls back and
    #    fast_full fails in server_startup or chat_probe, digest/timings/phase/diagnostics
    #    must belong to fast_full rather than reverting to fast_2l's empty_stream_content.
    ppu_recipe_diagnostics = _load_module(
        "ppu_recipe_diagnostics", CI_DIR / "ppu_recipe_diagnostics.py"
    )
    for fail_stage, expected_diag in (
        ("server_startup", "server_startup_failed"),
        ("chat_probe", "chat_probe_failed"),
    ):
        priv_dir = tmp_path / f"priv_{fail_stage}"
        res_dir = priv_dir / "results"
        res_dir.mkdir(parents=True)
        (priv_dir / "runtime.log").write_text("started\n", encoding="utf-8")
        stage_phase = priv_dir / "recipe-phase.txt"
        monkeypatch.setenv("PPU_RECIPE_PHASE_FILE", str(stage_phase))

        def attempt_full_nonstream_fail(
            *,
            layer_override,
            label,
            _stage=fail_stage,
            _phase=stage_phase,
            **kw,
        ):
            if layer_override == 4:
                _phase.write_text("stream_probe\n", encoding="utf-8")
                return {
                    "mode": label,
                    "layer_override": 4,
                    "status": "fail",
                    "error": "empty_stream_content",
                    "cold_start_s": 294.0,
                    "eval_s": 0.44,
                    "config_digest": "a" * 64,
                    "ppu_count": 2,
                    "probe": {"status": "pass", "elapsed_s": 0.29},
                    "stream_probe": {
                        "status": "fail",
                        "error": "empty_stream_content",
                        "elapsed_s": 0.15,
                    },
                }
            _phase.write_text(f"{_stage}\n", encoding="utf-8")
            return {
                "mode": label,
                "layer_override": None,
                "status": "fail",
                "error": f"{_stage} error",
                "cold_start_s": 123.0,
                "eval_s": 0.0,
                "config_digest": "b" * 64,
                "ppu_count": 2,
            }

        monkeypatch.setattr(
            ppu_recipe_runner,
            "_attempt_fast_sanity_once",
            attempt_full_nonstream_fail,
        )
        res_term = ppu_recipe_runner.run_fast_sanity(
            recipe_path=qwen_recipe,
            policy_path=auto_pol,
            results_dir=res_dir,
            parser_mode="skip",
        )
        assert res_term["status"] == "fail"
        assert res_term["config_digest"] == "b" * 64
        assert res_term["timings"]["cold_start_s"] == 123.0
        assert "nonstream_s" not in res_term["timings"]
        assert "stream_s" not in res_term["timings"]
        assert stage_phase.read_text(encoding="utf-8").strip() == fail_stage
        assert (
            ppu_recipe_diagnostics.failure_reason(priv_dir, "fast_sanity")
            == expected_diag
        )


def test_hardware_precision_binding_contract(tmp_path: Path) -> None:
    glm_doc = yaml.safe_load((RECIPES_DIR / "zai-org" / "GLM-5.3.yaml").read_text())
    glm_pol = yaml.safe_load((POLICY_DIR / "models" / "glm-5.3.yaml").read_text())
    qwen_doc = yaml.safe_load((RECIPES_DIR / "Qwen" / "Qwen3.8-27B.yaml").read_text())

    # 1. Valid M890 (bf16/mxfp4/fp4/fp8) and 810E (bf16/int8/w8a8_int8/w4a8_int8) controls
    for hw, prec in (
        ("zhenwu_m890", "bf16"),
        ("zhenwu_m890", "mxfp4"),
        ("zhenwu_m890", "mxfp4_fp8"),
        ("zhenwu_m890", "fp4"),
        ("zhenwu_m890", "fp8"),
        ("zhenwu_m890", "fp8_channel"),
        ("zhenwu_810e", "bf16"),
        ("zhenwu_810e", "int8"),
        ("zhenwu_810e", "w8a8_int8"),
        ("zhenwu_810e", "w4a8_int8"),
    ):
        ppu_recipe._validate_hardware_precision(
            hardware=hw, precision=prec, context="control"
        )

    # 2. Cross-binding & unsupported precision counterexamples rejected on recipe variants
    for hw, bad_prec, match in (
        ("zhenwu_m890", "w8a8_int8", "only allows bf16 or mxfp4/fp4/fp8"),
        ("zhenwu_m890", "int8", "only allows bf16 or mxfp4/fp4/fp8"),
        ("zhenwu_m890", "fp16", "only allows bf16 or mxfp4/fp4/fp8"),
        ("zhenwu_810e", "mxfp4", "only allows bf16 or int8 precision"),
        ("zhenwu_810e", "fp8", "only allows bf16 or int8 precision"),
        ("zhenwu_810e", "fp16", "only allows bf16 or int8 precision"),
    ):
        bad = copy.deepcopy(glm_doc)
        bad["variants"]["default"]["hardware"] = hw
        bad["variants"]["default"]["precision"] = bad_prec
        with pytest.raises(ppu_recipe.ValidationError, match=match):
            ppu_recipe.validate_serving_recipe_structure(bad, context="bad_hw_prec")

    # 3. Policy cross-binding rejected (810E policy pointing at MXFP4 variant, or M890 at w8a8_int8)
    bad_pol_810e = copy.deepcopy(glm_pol)
    bad_pol_810e["hardware"] = "zhenwu_810e"
    with pytest.raises(ppu_recipe.ValidationError, match="only allows bf16 or int8"):
        ppu_recipe.validate_policy_structure(bad_pol_810e, context="bad_pol_810e")

    bad_pol_m890 = copy.deepcopy(glm_pol)
    bad_pol_m890["variant"] = "w8a8_int8"
    with pytest.raises(
        ppu_recipe.ValidationError, match="only allows bf16 or mxfp4/fp4/fp8"
    ):
        ppu_recipe.validate_policy_structure(bad_pol_m890, context="bad_pol_m890")

    # 4. Cross-hardware preset resolution fails closed
    with pytest.raises(ppu_recipe.ValidationError, match="incompatible with"):
        ppu_recipe.resolve_recipe(
            RECIPES_DIR / "zai-org" / "GLM-5.3.yaml",
            variant="w8a8_int8",
            preset="decode",
        )

    # 5. BF16 is allowed across hardware with presets, sibling quantized variants, and model_sanity
    ppu_recipe.validate_serving_recipe_structure(qwen_doc, context="qwen_control")
    qwen_with_preset = copy.deepcopy(qwen_doc)
    qwen_with_preset["presets"] = {
        "prefill": {"hardware": "zhenwu_m890", "args": []},
        "decode": {"hardware": "zhenwu_810e", "args": []},
    }
    ppu_recipe.validate_serving_recipe_structure(
        qwen_with_preset, context="qwen_preset"
    )

    qwen_two_vars = copy.deepcopy(qwen_doc)
    qwen_two_vars["variants"]["fp8"] = {"precision": "fp8"}
    ppu_recipe.validate_serving_recipe_structure(qwen_two_vars, context="qwen_two_vars")

    qwen_pol = yaml.safe_load((POLICY_DIR / "models" / "qwen3.8-27b.yaml").read_text())
    qwen_pol_ms = copy.deepcopy(qwen_pol)
    qwen_pol_ms["tiers"]["model_sanity"] = copy.deepcopy(
        glm_pol["tiers"]["model_sanity"]
    )
    for stage in qwen_pol_ms["tiers"]["model_sanity"]["stages"]:
        stage.pop("preset", None)
    ppu_recipe.validate_policy_structure(qwen_pol_ms, context="qwen_ms")

    # 6. Unknown hardware rejected on recipe and policy
    bad_hw_recipe = copy.deepcopy(glm_doc)
    bad_hw_recipe["variants"]["default"]["hardware"] = "unknown_chip"
    with pytest.raises(ppu_recipe.ValidationError, match=r"hardware invalid"):
        ppu_recipe.validate_serving_recipe_structure(bad_hw_recipe, context="bad_hw")

    bad_hw_pol = copy.deepcopy(glm_pol)
    bad_hw_pol["hardware"] = "unknown_chip"
    with pytest.raises(ppu_recipe.ValidationError, match=r"hardware invalid"):
        ppu_recipe.validate_policy_structure(bad_hw_pol, context="bad_hw_pol")

    # 7. Orphan preset matching no variant rejected
    orphan = copy.deepcopy(glm_doc)
    orphan["variants"] = {"w8a8_int8": copy.deepcopy(glm_doc["variants"]["w8a8_int8"])}
    orphan["presets"] = {"decode": copy.deepcopy(glm_doc["presets"]["decode"])}
    with pytest.raises(ppu_recipe.ValidationError, match="matches no compatible"):
        ppu_recipe.validate_serving_recipe_structure(orphan, context="orphan_preset")

    # Verify temp file round-trip for a clean 810E-only recipe control
    only_810e = copy.deepcopy(glm_doc)
    only_810e["variants"] = {"default": copy.deepcopy(glm_doc["variants"]["w8a8_int8"])}
    only_810e["presets"] = {"prefill": {"args": []}}
    rec_file = tmp_path / "only_810e.yaml"
    rec_file.write_text(yaml.safe_dump(only_810e), encoding="utf-8")
    report = ppu_recipe.validate_paths(
        [rec_file], vllm_parser_mode="required", parser=_sample_vllm_parser()
    )
    assert report["checked_combinations"] == 2
    assert report["parser_checked"] is True


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda d: d["model"]["base_args"].append("--enable-prefix-caching"),
            "must not be hardcoded in recipe",
        ),
        (
            lambda d: d["model"]["base_args"].append("--no-enable-prefix-caching"),
            "must not be hardcoded in recipe",
        ),
        (
            lambda d: d["model"]["base_args"].append("--disable-radix-cache"),
            "must not be hardcoded in recipe",
        ),
        (
            lambda d: d["model"]["base_args"].extend(["--load-format", "dummy"]),
            "must not be hardcoded in recipe",
        ),
        (
            lambda d: d["model"]["base_args"].extend(["--model-path", "foo"]),
            "must not be hardcoded in recipe",
        ),
        (
            lambda d: d["model"]["base_args"].extend(["--host", "0.0.0.0"]),
            "must not be hardcoded in recipe",
        ),
        (
            lambda d: d["model"]["base_args"].extend(["--port", "12345"]),
            "must not be hardcoded in recipe",
        ),
        (
            lambda d: d["model"]["base_args"].extend(["--gpu-memory-utilization", "0"]),
            r"must be in \(0\.0, 1\.0\]",
        ),
        (
            lambda d: d["model"]["base_args"].extend(
                ["--gpu-memory-utilization", "1.2"]
            ),
            r"must be in \(0\.0, 1\.0\]",
        ),
        (
            lambda d: d["model"]["base_env"].update({"VLLM_USE_MODELSCOPE": "True"}),
            "runtime-mode env",
        ),
        (
            lambda d: d["presets"]["pd_prefill"].update(
                {"args": ["--kv-transfer-config", '{"device":"mlx5_bond_0"}']}
            ),
            "nic_device",
        ),
    ],
)
def test_recipe_boundary_and_sanitization_guardrails(mutator, match: str) -> None:
    doc = yaml.safe_load((RECIPES_DIR / "zai-org" / "GLM-5.3.yaml").read_text())
    mutator(doc)
    with pytest.raises(ppu_recipe.ValidationError, match=match):
        ppu_recipe.validate_serving_recipe_structure(doc, context="boundary_check")


def test_topology_placeholder_validation_and_substitution(tmp_path: Path) -> None:
    glm_doc = yaml.safe_load((RECIPES_DIR / "zai-org" / "GLM-5.3.yaml").read_text())

    # 1. Placeholder substitution wires real Mooncake env/extra_config and parses cleanly
    res_pd = ppu_recipe.resolve_recipe(
        RECIPES_DIR / "zai-org" / "GLM-5.3.yaml", preset="pd_prefill"
    )
    sub_flags, sub_env = ppu_recipe.substitute_validation_placeholders(
        res_pd["flags"], res_pd["server_env"]
    )
    kv_idx = sub_flags.index("--kv-transfer-config")
    kv_obj = json.loads(sub_flags[kv_idx + 1])
    assert "kv_port" not in kv_obj
    assert "kv_ip" not in kv_obj["kv_connector_extra_config"]
    assert kv_obj["kv_connector_extra_config"]["device_name"] == "rdma0"
    assert sub_env["VLLM_HOST_IP"] == "127.0.0.1"
    assert sub_env["VLLM_MOONCAKE_BOOTSTRAP_PORT"] == "8998"
    assert sub_env["MOONCAKE_PROTOCOL"] == "rdma"
    ppu_recipe.validate_flags_with_parser(
        _sample_vllm_parser(), sub_flags, context="pd_prefill_sub"
    )

    def _validate_doc(doc: dict, name: str) -> None:
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(doc), encoding="utf-8")
        ppu_recipe.validate_paths(
            [path], vllm_parser_mode="required", parser=_sample_vllm_parser()
        )

    # 2. single_node forbids placeholders and nodes > 1
    bad_single_ph = copy.deepcopy(glm_doc)
    bad_single_ph["presets"]["prefill"]["args"] = ["--master-addr", "{master_addr}"]
    with pytest.raises(
        ppu_recipe.ValidationError, match="single_node topology must not contain"
    ):
        _validate_doc(bad_single_ph, "single_ph")

    bad_single_nodes = copy.deepcopy(glm_doc)
    bad_single_nodes["presets"]["decode"]["topology"]["nodes"] = 2
    with pytest.raises(
        ppu_recipe.ValidationError, match="single_node topology requires nodes=1"
    ):
        _validate_doc(bad_single_nodes, "single_nodes")

    # 3. multi_node_ep requires nodes >= 2, {master_addr}, {node_rank}, and matching --nnodes
    bad_ep_nodes = copy.deepcopy(glm_doc)
    bad_ep_nodes["presets"]["decode_ep"]["topology"]["nodes"] = 1
    with pytest.raises(
        ppu_recipe.ValidationError, match="multi_node_ep topology requires nodes >= 2"
    ):
        _validate_doc(bad_ep_nodes, "ep_nodes")

    bad_ep_missing_ph = copy.deepcopy(glm_doc)
    bad_ep_missing_ph["presets"]["decode_ep"]["args"] = [
        tok
        for tok in bad_ep_missing_ph["presets"]["decode_ep"]["args"]
        if tok not in {"--master-addr", "{master_addr}"}
    ]
    with pytest.raises(
        ppu_recipe.ValidationError, match="missing required placeholder"
    ):
        _validate_doc(bad_ep_missing_ph, "ep_missing_ph")

    bad_ep_nnodes_mismatch = copy.deepcopy(glm_doc)
    ep_args = bad_ep_nnodes_mismatch["presets"]["decode_ep"]["args"]
    ep_args[ep_args.index("--nnodes") + 1] = "2"
    with pytest.raises(
        ppu_recipe.ValidationError, match="--nnodes .* must match topology.nodes"
    ):
        _validate_doc(bad_ep_nnodes_mismatch, "ep_nnodes_mismatch")

    bad_ep_no_nnodes = copy.deepcopy(glm_doc)
    ep_args2 = bad_ep_no_nnodes["presets"]["decode_ep"]["args"]
    idx_n = ep_args2.index("--nnodes")
    del ep_args2[idx_n : idx_n + 2]
    with pytest.raises(ppu_recipe.ValidationError, match="requires --nnodes"):
        _validate_doc(bad_ep_no_nnodes, "ep_no_nnodes")

    # 4. pd_disagg requires paired pd_prefill + pd_decode, placeholders, and matching kv_role
    unpaired_pd = copy.deepcopy(glm_doc)
    del unpaired_pd["presets"]["pd_decode"]
    with pytest.raises(ppu_recipe.ValidationError, match="must be declared as a pair"):
        _validate_doc(unpaired_pd, "unpaired_pd")

    wrong_role_pd = copy.deepcopy(glm_doc)
    p_args = wrong_role_pd["presets"]["pd_prefill"]["args"]
    p_args[p_args.index("--kv-transfer-config") + 1] = p_args[
        p_args.index("--kv-transfer-config") + 1
    ].replace("kv_producer", "kv_consumer")
    with pytest.raises(ppu_recipe.ValidationError, match="requires kv_role="):
        _validate_doc(wrong_role_pd, "wrong_role_pd")

    missing_ib_pd = copy.deepcopy(glm_doc)
    p_args2 = missing_ib_pd["presets"]["pd_prefill"]["args"]
    p_args2[p_args2.index("--kv-transfer-config") + 1] = p_args2[
        p_args2.index("--kv-transfer-config") + 1
    ].replace("{ib_devices}", "rdma0")
    with pytest.raises(
        ppu_recipe.ValidationError, match="missing required placeholder.*ib_devices"
    ):
        _validate_doc(missing_ib_pd, "missing_ib_pd")

    # 5. Unknown placeholder rejected in flags, env, and json_args
    unknown_ph_flag = copy.deepcopy(glm_doc)
    unknown_ph_flag["presets"]["decode_ep"]["args"].extend(
        ["--master-port", "{bogus_port}"]
    )
    with pytest.raises(ppu_recipe.ValidationError, match="unknown placeholder"):
        _validate_doc(unknown_ph_flag, "unknown_ph_flag")

    unknown_ph_json = copy.deepcopy(glm_doc)
    unknown_ph_json["presets"]["decode"]["json_args"] = {
        "--compilation-config": {"mode": "{bad_mode}"}
    }
    with pytest.raises(ppu_recipe.ValidationError, match="unknown placeholder"):
        _validate_doc(unknown_ph_json, "unknown_ph_json")


def test_single_node_selector_skips_multinode_pd_and_810e_policies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Checked-in repo matrix: GLM-5.3 has prefill (TP4) + decode (TP8) in fast_sanity,
    # and only prefill (TP4) in model_sanity; catalog-only fp8_channel / w8a8_int8 are absent.
    fast_checked = ppu_e2e_select.build_recipe_matrix("", "fast_sanity")["include"]
    fast_by_key = {item["key"]: item for item in fast_checked}
    assert fast_by_key["glm-5.3"]["ppu_count"] == 4
    assert fast_by_key["glm-5.3"]["preset"] == "prefill"
    assert fast_by_key["glm-5.3-decode"]["ppu_count"] == 8
    assert fast_by_key["glm-5.3-decode"]["preset"] == "decode"

    nightly_checked = ppu_e2e_select.build_recipe_matrix("", "model_sanity")["include"]
    nightly_by_key = {item["key"]: item for item in nightly_checked}
    assert nightly_by_key["glm-5.3"]["preset"] == "prefill"
    assert "glm-5.3-decode" not in nightly_by_key

    # Synthetic multi-node, PD disagg, and 810E policies in an isolated tree
    shutil.copytree(ROOT / "tests/ppu", tmp_path / "tests/ppu")
    monkeypatch.setattr(ppu_e2e_select, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ppu_recipe, "REPO_ROOT", tmp_path)
    policy_dir = tmp_path / "tests/ppu/policy/models"
    base_fs = yaml.safe_load((policy_dir / "glm-5.3-decode.yaml").read_text())

    ep_pol = copy.deepcopy(base_fs)
    ep_pol["tiers"]["fast_sanity"]["preset"] = "decode_ep"
    (policy_dir / "glm-5.3-decode-ep.yaml").write_text(yaml.safe_dump(ep_pol))

    pd_pol = copy.deepcopy(base_fs)
    pd_pol["tiers"]["fast_sanity"]["preset"] = "pd_prefill"
    (policy_dir / "glm-5.3-pd-prefill.yaml").write_text(yaml.safe_dump(pd_pol))

    e810_pol = copy.deepcopy(base_fs)
    e810_pol["variant"] = "w8a8_int8"
    e810_pol["hardware"] = "zhenwu_810e"
    e810_pol["tiers"]["fast_sanity"]["preset"] = "prefill"
    (policy_dir / "glm-5.3-810e.yaml").write_text(yaml.safe_dump(e810_pol))

    selected_keys = {
        item["key"]
        for item in ppu_e2e_select.build_recipe_matrix("", "fast_sanity")["include"]
    }
    assert {"glm-5.3", "glm-5.3-decode"} <= selected_keys
    assert "glm-5.3-decode-ep" not in selected_keys
    assert "glm-5.3-pd-prefill" not in selected_keys
    assert "glm-5.3-810e" not in selected_keys

    # Explicitly requesting non-single-node or wrong-hardware policy fails fast
    with pytest.raises(ValueError, match="not runnable on single-node CI"):
        ppu_e2e_select.build_recipe_matrix("glm-5.3-decode-ep", "fast_sanity")
    with pytest.raises(ValueError, match="not runnable on single-node CI"):
        ppu_e2e_select.build_recipe_matrix("glm-5.3-pd-prefill", "fast_sanity")
    with pytest.raises(ValueError, match="not runnable on single-node 'zhenwu_m890'"):
        ppu_e2e_select.build_recipe_matrix("glm-5.3-810e", "fast_sanity")

    # Positive control: selecting with hardware='zhenwu_810e' returns the 810E policy
    selected_810e = ppu_e2e_select.build_recipe_matrix(
        "", "fast_sanity", hardware="zhenwu_810e"
    )["include"]
    assert [item["key"] for item in selected_810e] == ["glm-5.3-810e"]
    assert selected_810e[0]["ppu_count"] == 16


def test_model_sanity_rejects_mismatched_stage_presets() -> None:
    pol = yaml.safe_load((POLICY_DIR / "models" / "glm-5.3.yaml").read_text())
    pol["tiers"]["model_sanity"]["stages"][1]["preset"] = "decode"
    with pytest.raises(
        ppu_recipe.ValidationError,
        match="all shared-server stages must agree on preset",
    ):
        ppu_recipe.validate_policy_structure(pol, context="mismatched_stage_preset")


def test_bf16_supported_across_hardware_and_respects_hardware_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qwen_doc = yaml.safe_load((RECIPES_DIR / "Qwen" / "Qwen3.8-27B.yaml").read_text())
    qwen_pol = yaml.safe_load((POLICY_DIR / "models" / "qwen3.8-27b.yaml").read_text())

    # 1a. Checked-in Qwen3.8-27B (unpinned BF16 without hardware_overrides) enumerates
    #     both zhenwu_m890 and zhenwu_810e in catalog and parser validation; unpinned
    #     presets also expand both chips without needing hardware_overrides.
    qwen_entries = ppu_recipe.iter_recipe_catalog_entries(
        RECIPES_DIR / "Qwen" / "Qwen3.8-27B.yaml", qwen_doc
    )
    assert [(e["variant"], e["preset"], e["hardware"]) for e in qwen_entries] == [
        ("default", None, "zhenwu_m890"),
        ("default", None, "zhenwu_810e"),
    ]
    no_override_bf16 = copy.deepcopy(qwen_doc)
    no_override_bf16["presets"] = {
        "prefill": {"args": ["--max-num-batched-tokens", "8192"]},
        "decode": {
            "hardware": "zhenwu_810e",
            "args": ["--max-num-batched-tokens", "2048"],
        },
    }
    no_override_path = tmp_path / "no_override_bf16.yaml"
    no_override_path.write_text(
        yaml.safe_dump(no_override_bf16, sort_keys=False), encoding="utf-8"
    )
    no_ov_report = ppu_recipe.validate_paths(
        [no_override_path], vllm_parser_mode="required", parser=_sample_vllm_parser()
    )
    assert no_ov_report["checked_combinations"] == 5
    assert [
        (e["variant"], e["preset"], e["hardware"])
        for e in ppu_recipe.iter_recipe_catalog_entries(no_override_path)
    ] == [
        ("default", None, "zhenwu_m890"),
        ("default", None, "zhenwu_810e"),
        ("default", "prefill", "zhenwu_m890"),
        ("default", "prefill", "zhenwu_810e"),
        ("default", "decode", "zhenwu_810e"),
    ]

    # 1b. Portable BF16 recipe with M890 and 810E presets + hardware_overrides passes
    #     structure validation, parser checks, catalog expansion, and hardware-aware policies.
    small_bf16 = copy.deepcopy(qwen_doc)
    small_bf16["model"]["model_id"] = "Qwen/Qwen3-8B"
    small_bf16["ppu"]["hf_model_id"] = "Qwen/Qwen3-8B"
    small_bf16["ppu"]["assets"] = {"model": "qwen3-8b", "tokenizer": "qwen3-8b"}
    small_bf16["ppu"]["served_name"] = "qwen3-8b"
    small_bf16["hardware_overrides"] = {
        "zhenwu_810e": {"extra_args": ["--max-num-seqs", "64"]}
    }
    small_bf16["presets"] = {
        "prefill": {
            "hardware": "zhenwu_m890",
            "args": ["--max-num-batched-tokens", "8192"],
        },
        "decode": {
            "hardware": "zhenwu_810e",
            "args": ["--max-num-batched-tokens", "2048"],
        },
    }
    ppu_recipe.validate_serving_recipe_structure(small_bf16, context="small_bf16")

    shutil.copytree(ROOT / "tests/ppu", tmp_path / "tests/ppu")
    monkeypatch.setattr(ppu_e2e_select, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ppu_recipe, "REPO_ROOT", tmp_path)
    rec_path = tmp_path / "tests/ppu/recipes/Qwen/Qwen3-8B.yaml"
    rec_path.write_text(yaml.safe_dump(small_bf16, sort_keys=False), encoding="utf-8")

    report = ppu_recipe.validate_paths(
        [rec_path], vllm_parser_mode="required", parser=_sample_vllm_parser()
    )
    # 2 baselines (m890 default + 810e hardware_override) + 2 presets (prefill/m890, decode/810e)
    assert report["checked_combinations"] == 4
    entries = ppu_recipe.iter_recipe_catalog_entries(rec_path)
    assert [(e["variant"], e["preset"], e["hardware"]) for e in entries] == [
        ("default", None, "zhenwu_m890"),
        ("default", None, "zhenwu_810e"),
        ("default", "prefill", "zhenwu_m890"),
        ("default", "decode", "zhenwu_810e"),
    ]
    decode_810e = ppu_recipe.resolve_recipe(rec_path, preset="decode")
    assert decode_810e["hardware"] == "zhenwu_810e"
    assert "--max-num-seqs" in decode_810e["flags"]
    assert "--max-num-batched-tokens" in decode_810e["flags"]

    pol_m890 = copy.deepcopy(qwen_pol)
    pol_m890["model_id"] = "Qwen/Qwen3-8B"
    pol_m890["recipe"] = "tests/ppu/recipes/Qwen/Qwen3-8B.yaml"
    pol_m890["tiers"]["fast_sanity"]["preset"] = "prefill"
    pol_m890_path = tmp_path / "tests/ppu/policy/models/qwen3-8b.yaml"
    pol_m890_path.write_text(yaml.safe_dump(pol_m890), encoding="utf-8")

    pol_810e = copy.deepcopy(pol_m890)
    pol_810e["hardware"] = "zhenwu_810e"
    pol_810e["tiers"]["fast_sanity"]["preset"] = "decode"
    pol_810e_path = tmp_path / "tests/ppu/policy/models/qwen3-8b-810e.yaml"
    pol_810e_path.write_text(yaml.safe_dump(pol_810e), encoding="utf-8")

    ppu_recipe.validate_policy_structure(
        pol_m890, context="pol_m890", base_dir=tmp_path
    )
    ppu_recipe.validate_policy_structure(
        pol_810e, context="pol_810e", base_dir=tmp_path
    )
    # Default M890 selector picks qwen3-8b and skips qwen3-8b-810e
    m890_keys = {
        item["key"]
        for item in ppu_e2e_select.build_recipe_matrix("", "fast_sanity")["include"]
    }
    assert "qwen3-8b" in m890_keys
    assert "qwen3-8b-810e" not in m890_keys

    # 2. Explicitly pinned BF16 variant still enforces hardware pin on presets and policies
    pinned_m890 = copy.deepcopy(small_bf16)
    pinned_m890["variants"]["default"]["hardware"] = "zhenwu_m890"
    with pytest.raises(ppu_recipe.ValidationError, match="matches no compatible"):
        ppu_recipe.validate_serving_recipe_structure(
            pinned_m890, context="pinned_m890_orphan_810e_preset"
        )

    pinned_m890["presets"]["decode"]["variants"] = ["default"]
    with pytest.raises(
        ppu_recipe.ValidationError, match="conflicts with preset hardware 'zhenwu_810e'"
    ):
        ppu_recipe.validate_serving_recipe_structure(
            pinned_m890, context="pinned_m890_conflict_preset"
        )

    pinned_only = copy.deepcopy(qwen_doc)
    pinned_only["variants"]["default"]["hardware"] = "zhenwu_m890"
    pinned_path = tmp_path / "tests/ppu/recipes/Qwen/Qwen3-8B-Pinned.yaml"
    pinned_path.write_text(yaml.safe_dump(pinned_only), encoding="utf-8")
    bad_810e_pol = copy.deepcopy(qwen_pol)
    bad_810e_pol["hardware"] = "zhenwu_810e"
    bad_810e_pol["recipe"] = "tests/ppu/recipes/Qwen/Qwen3-8B-Pinned.yaml"
    with pytest.raises(
        ppu_recipe.ValidationError,
        match="targets hardware 'zhenwu_m890', incompatible with policy hardware 'zhenwu_810e'",
    ):
        ppu_recipe.validate_policy_structure(
            bad_810e_pol, context="bad_pinned_bf16_pol", base_dir=tmp_path
        )
    with pytest.raises(
        ppu_recipe.ValidationError,
        match="targets hardware 'zhenwu_m890', incompatible with 'zhenwu_810e'",
    ):
        ppu_recipe.resolve_recipe(pinned_path, hardware="zhenwu_810e")

    # 3. Policy model_id mismatch against recipe canonical model.model_id fails closed
    #    across validate_policy_structure, validate_paths, resolve_recipe, and selector
    #    (while variant hf_model_id overrides like fp8_channel keep working).
    spoof_pol = copy.deepcopy(qwen_pol)
    spoof_pol["model_id"] = "zai-org/GLM-5.3"
    spoof_pol_path = tmp_path / "tests/ppu/policy/models/spoof-glm-on-qwen.yaml"
    spoof_pol_path.write_text(yaml.safe_dump(spoof_pol), encoding="utf-8")
    with pytest.raises(
        ppu_recipe.ValidationError,
        match="policy model_id 'zai-org/GLM-5.3' does not match recipe .* model.model_id 'Qwen/Qwen3.8-27B'",
    ):
        ppu_recipe.validate_policy_structure(
            spoof_pol, context="spoof_pol", base_dir=tmp_path
        )
    with pytest.raises(
        ppu_recipe.ValidationError,
        match="policy model_id 'zai-org/GLM-5.3' does not match recipe .* model.model_id 'Qwen/Qwen3.8-27B'",
    ):
        ppu_recipe.resolve_recipe(
            tmp_path / "tests/ppu/recipes/Qwen/Qwen3.8-27B.yaml",
            policy_path=spoof_pol_path,
            tier="fast_sanity",
        )
    with pytest.raises(
        ppu_recipe.ValidationError,
        match="policy model_id 'zai-org/GLM-5.3' does not match recipe model.model_id 'Qwen/Qwen3.8-27B'",
    ):
        ppu_recipe._resolve_recipe_from_doc(
            qwen_doc,
            recipe_label="qwen_doc",
            policy=spoof_pol,
            policy_label="spoof_pol",
            tier="fast_sanity",
        )
    with pytest.raises(
        ppu_recipe.ValidationError,
        match="policy model_id 'zai-org/GLM-5.3' does not match recipe",
    ):
        ppu_e2e_select.build_recipe_matrix("", "fast_sanity")
    spoof_pol_path.unlink()

    # Positive control: policy using canonical model_id with a variant overriding hf_model_id passes
    glm_fp8_pol = yaml.safe_load((POLICY_DIR / "models" / "glm-5.3.yaml").read_text())
    glm_fp8_pol["variant"] = "fp8_channel"
    glm_fp8_pol["tiers"] = {"fast_sanity": glm_fp8_pol["tiers"]["fast_sanity"]}
    glm_fp8_pol.pop("replaces_legacy_cases", None)
    ppu_recipe.validate_policy_structure(
        glm_fp8_pol, context="glm_fp8_variant_control", base_dir=tmp_path
    )
    glm_fp8_pol_path = tmp_path / "tests/ppu/policy/models/glm-5.3-fp8-ctrl.yaml"
    glm_fp8_pol_path.write_text(yaml.safe_dump(glm_fp8_pol), encoding="utf-8")
    glm_fp8_res = ppu_recipe.resolve_recipe(
        tmp_path / "tests/ppu/recipes/zai-org/GLM-5.3.yaml",
        policy_path=glm_fp8_pol_path,
        tier="fast_sanity",
    )
    assert glm_fp8_res["model_id"] == "zai-org/GLM-5.3"
    assert (
        glm_fp8_res["hf_model_id"]
        == "T-HEAD/GLM-5.3-Quant-W-FP8-PerChannel-A-FP8-PerChannel"
    )


def test_topology_flag_invariant_blocks_multinode_and_disagg_bypass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    glm_doc = yaml.safe_load((RECIPES_DIR / "zai-org" / "GLM-5.3.yaml").read_text())

    # 1. Counterexample (R1 P1-02): single_node preset smuggling multi-node launch flags
    #    (--nnodes 4 --node-rank 0 --master-addr localhost) must fail closed across
    #    structure validation, validate_paths, resolve_recipe, and CI selector.
    bypass_doc = copy.deepcopy(glm_doc)
    bypass_doc["presets"]["decode"]["args"] = [
        "--nnodes",
        "4",
        "--node-rank",
        "0",
        "--master-addr",
        "localhost",
    ]
    with pytest.raises(
        ppu_recipe.ValidationError,
        match="single_node topology must not set multi-node flag",
    ):
        ppu_recipe.validate_serving_recipe_structure(bypass_doc, context="bypass_mn")

    shutil.copytree(ROOT / "tests/ppu", tmp_path / "tests/ppu")
    monkeypatch.setattr(ppu_e2e_select, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ppu_recipe, "REPO_ROOT", tmp_path)
    glm_copy = tmp_path / "tests/ppu/recipes/zai-org/GLM-5.3.yaml"
    glm_copy.write_text(yaml.safe_dump(bypass_doc), encoding="utf-8")

    with pytest.raises(
        ppu_recipe.ValidationError,
        match="single_node topology must not set multi-node flag",
    ):
        ppu_recipe.validate_paths(
            [glm_copy], vllm_parser_mode="required", parser=_sample_vllm_parser()
        )
    with pytest.raises(
        ppu_recipe.ValidationError,
        match="single_node topology must not set multi-node flag",
    ):
        ppu_recipe.resolve_recipe(
            glm_copy,
            policy_path=tmp_path / "tests/ppu/policy/models/glm-5.3-decode.yaml",
            tier="fast_sanity",
        )
    with pytest.raises(
        ppu_recipe.ValidationError,
        match="single_node topology must not set multi-node flag",
    ):
        ppu_e2e_select.build_recipe_matrix("", "fast_sanity")
    with pytest.raises(
        ppu_recipe.ValidationError,
        match="single_node topology must not set multi-node flag",
    ):
        ppu_e2e_select.build_recipe_matrix("glm-5.3-decode", "fast_sanity")

    # 2. single_node preset injecting --kv-transfer-config or exceeding declared devices
    bad_kv_single = copy.deepcopy(glm_doc)
    bad_kv_single["presets"]["decode"]["args"] = [
        "--kv-transfer-config",
        '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}',
    ]
    with pytest.raises(
        ppu_recipe.ValidationError,
        match="single_node topology must not set --kv-transfer-config",
    ):
        ppu_recipe.validate_serving_recipe_structure(bad_kv_single, context="kv_single")

    bad_cap_single = copy.deepcopy(glm_doc)
    bad_cap_single["presets"]["decode"]["remove_args"] = ["--tensor-parallel-size"]
    bad_cap_single["presets"]["decode"].pop("tp", None)
    bad_cap_single["presets"]["decode"]["args"] = ["--data-parallel-size", "32"]
    with pytest.raises(
        ppu_recipe.ValidationError, match="exceeding declared topology capacity"
    ):
        ppu_recipe.validate_serving_recipe_structure(
            bad_cap_single, context="cap_single"
        )

    # 3. single-node pd_disagg preset smuggling --nnodes 4 fails closed
    bad_pd_mn = copy.deepcopy(glm_doc)
    bad_pd_mn["presets"]["pd_prefill"]["args"].extend(["--nnodes", "4"])
    with pytest.raises(
        ppu_recipe.ValidationError,
        match="single-node pd_disagg topology must not set multi-node flag",
    ):
        ppu_recipe.validate_serving_recipe_structure(bad_pd_mn, context="pd_mn")

    # 4. Positive control: restoring clean GLM-5.3.yaml passes full validation and selector
    glm_copy.write_text(yaml.safe_dump(glm_doc), encoding="utf-8")
    report = ppu_recipe.validate_paths(
        [glm_copy], vllm_parser_mode="required", parser=_sample_vllm_parser()
    )
    assert report["checked_combinations"] == 14
    assert report["parser_checked"] is True
    decode_entry = next(
        item
        for item in ppu_e2e_select.build_recipe_matrix("glm-5.3-decode", "fast_sanity")[
            "include"
        ]
        if item["key"] == "glm-5.3-decode"
    )
    assert decode_entry["ppu_count"] == 8
    assert decode_entry["preset"] == "decode"


def test_mooncake_pd_endpoint_wiring_nondefault_and_counterexamples(
    tmp_path: Path,
) -> None:
    recipe_path = RECIPES_DIR / "zai-org" / "GLM-5.3.yaml"
    glm_doc = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))

    # 1. Non-default address, port, and IB device list resolve to Mooncake's real runtime inputs
    prefill = ppu_recipe.resolve_recipe(recipe_path, preset="pd_prefill")
    p_flags, p_env = ppu_recipe.substitute_validation_placeholders(
        prefill["flags"],
        prefill["server_env"],
        values={
            "kv_ip": "198.51.100.10",
            "kv_port": 18998,
            "ib_devices": "rdma2,rdma3",
        },
    )
    p_kv = json.loads(p_flags[p_flags.index("--kv-transfer-config") + 1])
    assert p_env["VLLM_HOST_IP"] == "198.51.100.10"
    assert int(p_env["VLLM_MOONCAKE_BOOTSTRAP_PORT"]) == 18998
    assert p_kv["kv_role"] == "kv_producer"
    assert p_kv["kv_connector_extra_config"]["device_name"] == "rdma2,rdma3"
    assert "kv_port" not in p_kv
    assert "kv_ip" not in p_kv["kv_connector_extra_config"]
    ppu_recipe.validate_flags_with_parser(
        _sample_vllm_parser(), p_flags, context="pd_prefill_nondefault"
    )

    decode = ppu_recipe.resolve_recipe(recipe_path, preset="pd_decode")
    d_flags, d_env = ppu_recipe.substitute_validation_placeholders(
        decode["flags"],
        decode["server_env"],
        values={"kv_ip": "198.51.100.20", "ib_devices": "rdma4,rdma5"},
    )
    d_kv = json.loads(d_flags[d_flags.index("--kv-transfer-config") + 1])
    assert d_env["VLLM_HOST_IP"] == "198.51.100.20"
    assert "VLLM_MOONCAKE_BOOTSTRAP_PORT" not in d_env
    assert d_kv["kv_role"] == "kv_consumer"
    assert d_kv["kv_connector_extra_config"]["device_name"] == "rdma4,rdma5"
    assert "kv_port" not in d_kv
    assert "kv_ip" not in d_kv["kv_connector_extra_config"]
    ppu_recipe.validate_flags_with_parser(
        _sample_vllm_parser(), d_flags, context="pd_decode_nondefault"
    )

    # 2. External bootstrap_server_address mode on prefill also works with non-default host:port
    ext_doc = copy.deepcopy(glm_doc)
    ext_doc["presets"]["pd_prefill"]["env"].pop("VLLM_MOONCAKE_BOOTSTRAP_PORT")
    ext_args = ext_doc["presets"]["pd_prefill"]["args"]
    ext_kv = json.loads(ext_args[ext_args.index("--kv-transfer-config") + 1])
    ext_kv["kv_connector_extra_config"]["bootstrap_server_address"] = (
        "{kv_ip}:{kv_port}"
    )
    ext_args[ext_args.index("--kv-transfer-config") + 1] = json.dumps(
        ext_kv, separators=(",", ":")
    )
    ext_path = tmp_path / "ext_bootstrap.yaml"
    ext_path.write_text(yaml.safe_dump(ext_doc), encoding="utf-8")
    ext_res = ppu_recipe.resolve_recipe(ext_path, preset="pd_prefill")
    ext_flags, ext_env = ppu_recipe.substitute_validation_placeholders(
        ext_res["flags"],
        ext_res["server_env"],
        values={"kv_ip": "198.51.100.30", "kv_port": 19001, "ib_devices": "rdma1"},
    )
    ext_sub_kv = json.loads(ext_flags[ext_flags.index("--kv-transfer-config") + 1])
    assert ext_env["VLLM_HOST_IP"] == "198.51.100.30"
    assert (
        ext_sub_kv["kv_connector_extra_config"]["bootstrap_server_address"]
        == "198.51.100.30:19001"
    )
    assert ppu_recipe._parse_mooncake_host_port(
        ext_sub_kv["kv_connector_extra_config"]["bootstrap_server_address"],
        context="ext_ipv4",
    ) == ("198.51.100.30", 19001)

    # Bracketed IPv6 host:port control passes
    ext_v6 = copy.deepcopy(ext_doc)
    v6_args = ext_v6["presets"]["pd_prefill"]["args"]
    v6_kv = json.loads(v6_args[v6_args.index("--kv-transfer-config") + 1])
    v6_kv["kv_connector_extra_config"]["bootstrap_server_address"] = (
        "[{kv_ip}]:{kv_port}"
    )
    v6_args[v6_args.index("--kv-transfer-config") + 1] = json.dumps(
        v6_kv, separators=(",", ":")
    )
    ppu_recipe.validate_serving_recipe_structure(ext_v6, context="ext_ipv6_control")
    assert ppu_recipe._parse_mooncake_host_port("[::1]:19001", context="v6") == (
        "::1",
        19001,
    )

    # Counterexamples: scheme/path/extra-colon/empty-host bootstrap_server_address rejected
    for bad_addr in (
        "http://{kv_ip}:{kv_port}",
        "{kv_ip}:0:{kv_port}",
        ":{kv_port}",
        "{kv_ip}:{kv_port}/query",
        "[{kv_ip}]:0:{kv_port}",
        "[{kv_ip}]{kv_port}",
    ):
        bad_ext = copy.deepcopy(ext_doc)
        b_args = bad_ext["presets"]["pd_prefill"]["args"]
        b_kv = json.loads(b_args[b_args.index("--kv-transfer-config") + 1])
        b_kv["kv_connector_extra_config"]["bootstrap_server_address"] = bad_addr
        b_args[b_args.index("--kv-transfer-config") + 1] = json.dumps(
            b_kv, separators=(",", ":")
        )
        with pytest.raises(
            ppu_recipe.ValidationError, match="bootstrap_server_address"
        ):
            ppu_recipe.validate_serving_recipe_structure(bad_ext, context="bad_ext")
        bad_ext_path = tmp_path / f"bad_ext_{abs(hash(bad_addr))}.yaml"
        bad_ext_path.write_text(yaml.safe_dump(bad_ext), encoding="utf-8")
        with pytest.raises(
            ppu_recipe.ValidationError, match="bootstrap_server_address"
        ):
            ppu_recipe.validate_paths(
                [bad_ext_path],
                vllm_parser_mode="required",
                parser=_sample_vllm_parser(),
            )

    # 3. Counterexamples: dead Mooncake fields and missing/ambiguous runtime wiring fail closed
    dead_port = copy.deepcopy(glm_doc)
    dp_args = dead_port["presets"]["pd_prefill"]["args"]
    dp_kv = json.loads(dp_args[dp_args.index("--kv-transfer-config") + 1])
    dp_kv["kv_port"] = "{kv_port}"
    dp_args[dp_args.index("--kv-transfer-config") + 1] = json.dumps(dp_kv)
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"does not read KVTransferConfig\.kv_port",
    ):
        ppu_recipe.validate_serving_recipe_structure(dead_port, context="dead_port")

    dead_ip = copy.deepcopy(glm_doc)
    di_args = dead_ip["presets"]["pd_prefill"]["args"]
    di_kv = json.loads(di_args[di_args.index("--kv-transfer-config") + 1])
    di_kv["kv_connector_extra_config"]["kv_ip"] = "{kv_ip}"
    di_args[di_args.index("--kv-transfer-config") + 1] = json.dumps(di_kv)
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"does not read kv_connector_extra_config\.kv_ip",
    ):
        ppu_recipe.validate_serving_recipe_structure(dead_ip, context="dead_ip")

    no_host_ip = copy.deepcopy(glm_doc)
    no_host_ip["presets"]["pd_decode"]["env"].pop("VLLM_HOST_IP")
    with pytest.raises(
        ppu_recipe.ValidationError, match="missing required placeholder.*kv_ip"
    ):
        ppu_recipe.validate_serving_recipe_structure(no_host_ip, context="no_host_ip")

    both_bootstrap = copy.deepcopy(ext_doc)
    both_bootstrap["presets"]["pd_prefill"]["env"]["VLLM_MOONCAKE_BOOTSTRAP_PORT"] = (
        "{kv_port}"
    )
    with pytest.raises(
        ppu_recipe.ValidationError,
        match="must not set both bootstrap_server_address and VLLM_MOONCAKE_BOOTSTRAP_PORT",
    ):
        ppu_recipe.validate_serving_recipe_structure(
            both_bootstrap, context="both_bootstrap"
        )

    decode_bootstrap = copy.deepcopy(glm_doc)
    decode_bootstrap["presets"]["pd_decode"]["env"]["VLLM_MOONCAKE_BOOTSTRAP_PORT"] = (
        "{kv_port}"
    )
    with pytest.raises(
        ppu_recipe.ValidationError,
        match="decode .* does not read bootstrap_server_address or VLLM_MOONCAKE_BOOTSTRAP_PORT",
    ):
        ppu_recipe.validate_serving_recipe_structure(
            decode_bootstrap, context="decode_bootstrap"
        )


def test_effective_vllm_world_size_pcp_defaults_and_runner_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    glm_doc = yaml.safe_load((RECIPES_DIR / "zai-org" / "GLM-5.3.yaml").read_text())

    # 1. TP8 + PCP2 demands 16 devices (not 8): fails against 1x8 topology, resolves to 16 on 1x16
    pcp_oversub = copy.deepcopy(glm_doc)
    pcp_oversub["presets"]["decode"]["args"] = [
        "--prefill-context-parallel-size",
        "2",
    ]
    with pytest.raises(
        ppu_recipe.ValidationError,
        match="requires 16 devices, exceeding declared topology capacity",
    ):
        ppu_recipe.validate_serving_recipe_structure(pcp_oversub, context="pcp_1x8")

    pcp_16 = copy.deepcopy(pcp_oversub)
    pcp_16["presets"]["decode"]["topology"]["devices_per_node"] = 16
    pcp_path = tmp_path / "glm_pcp16.yaml"
    pcp_path.write_text(yaml.safe_dump(pcp_16), encoding="utf-8")
    resolved_pcp = ppu_recipe.resolve_recipe(pcp_path, preset="decode")
    assert resolved_pcp["ppu_count"] == 16
    report = ppu_recipe.validate_paths(
        [pcp_path], vllm_parser_mode="required", parser=_sample_vllm_parser()
    )
    assert report["parser_checked"] is True

    # 2. Removing --data-parallel-size 32 from 4-node decode_ep yields vLLM default world size 1,
    #    which must fail (not silently copy topology capacity 32)
    no_dp_ep = copy.deepcopy(glm_doc)
    ep_args = no_dp_ep["presets"]["decode_ep"]["args"]
    idx_dp = ep_args.index("--data-parallel-size")
    del ep_args[idx_dp : idx_dp + 2]
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"topology\.nodes \(4\) must evenly divide effective vLLM world size \(1\)",
    ):
        ppu_recipe.validate_serving_recipe_structure(no_dp_ep, context="no_dp_ep")

    # 3. Single-node recipe omitting all parallel flags resolves to vLLM default world_size=1
    one_card = copy.deepcopy(glm_doc)
    one_card["ppu"]["topology"]["devices_per_node"] = 1
    one_card["model"]["base_args"] = [
        tok
        for i, tok in enumerate(one_card["model"]["base_args"])
        if tok != "--tensor-parallel-size"
        and (
            i == 0 or one_card["model"]["base_args"][i - 1] != "--tensor-parallel-size"
        )
    ]
    one_card["variants"] = {"default": copy.deepcopy(glm_doc["variants"]["default"])}
    one_card["presets"] = {}
    one_path = tmp_path / "one_card.yaml"
    one_path.write_text(yaml.safe_dump(one_card), encoding="utf-8")
    assert ppu_recipe.resolve_recipe(one_path)["ppu_count"] == 1

    # 4. Single-node M890 selector enforces the 8-card runner cap for both TP16 and TP8xPCP2
    shutil.copytree(ROOT / "tests/ppu", tmp_path / "tests/ppu")
    monkeypatch.setattr(ppu_e2e_select, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ppu_recipe, "REPO_ROOT", tmp_path)
    glm_tp16_doc = copy.deepcopy(glm_doc)
    glm_tp16_doc["presets"]["decode"]["topology"]["devices_per_node"] = 16
    glm_tp16_doc["presets"]["decode"]["tp"] = 16
    (tmp_path / "tests/ppu/recipes/zai-org/GLM-5.3.yaml").write_text(
        yaml.safe_dump(glm_tp16_doc), encoding="utf-8"
    )
    # Static validation still passes (preset stays available in static catalog)
    assert (
        ppu_recipe.validate_paths(
            [tmp_path / "tests/ppu/recipes/zai-org/GLM-5.3.yaml"],
            vllm_parser_mode="required",
            parser=_sample_vllm_parser(),
        )["checked_combinations"]
        == 14
    )
    default_keys = {
        item["key"]
        for item in ppu_e2e_select.build_recipe_matrix("", "fast_sanity")["include"]
    }
    assert "glm-5.3" in default_keys
    assert "glm-5.3-decode" not in default_keys
    with pytest.raises(
        ValueError,
        match="requires 16 devices, exceeding single-node 'zhenwu_m890' CI runner limit \\(8\\)",
    ):
        ppu_e2e_select.build_recipe_matrix("glm-5.3-decode", "fast_sanity")
    # Positive control: higher runner cap admits the 16-card entry cleanly
    admitted = ppu_e2e_select.build_recipe_matrix(
        "glm-5.3-decode", "fast_sanity", max_devices=16
    )["include"]
    assert len(admitted) == 1 and admitted[0]["ppu_count"] == 16

    # Same 8-card cap catches TP8 + PCP2 (16 devices) in selector
    (tmp_path / "tests/ppu/recipes/zai-org/GLM-5.3.yaml").write_text(
        yaml.safe_dump(pcp_16), encoding="utf-8"
    )
    default_keys_pcp = {
        item["key"]
        for item in ppu_e2e_select.build_recipe_matrix("", "fast_sanity")["include"]
    }
    assert "glm-5.3-decode" not in default_keys_pcp
    with pytest.raises(
        ValueError,
        match="requires 16 devices, exceeding single-node 'zhenwu_m890' CI runner limit \\(8\\)",
    ):
        ppu_e2e_select.build_recipe_matrix("glm-5.3-decode", "fast_sanity")


def test_vllm_recipe_rejects_missing_or_empty_variants(tmp_path: Path) -> None:
    glm_doc = yaml.safe_load((RECIPES_DIR / "zai-org" / "GLM-5.3.yaml").read_text())

    for name, mutator in (
        ("empty_variants", lambda d: d.update({"variants": {}, "presets": {}})),
        (
            "missing_variants",
            lambda d: (d.pop("variants", None), d.update({"presets": {}})),
        ),
    ):
        bad = copy.deepcopy(glm_doc)
        mutator(bad)
        with pytest.raises(
            ppu_recipe.ValidationError,
            match=r"variants must be a non-empty mapping for vllm serving recipes",
        ):
            ppu_recipe.validate_serving_recipe_structure(bad, context=name)

        bad_path = tmp_path / f"{name}.yaml"
        bad_path.write_text(yaml.safe_dump(bad), encoding="utf-8")
        with pytest.raises(
            ppu_recipe.ValidationError,
            match=r"variants must be a non-empty mapping for vllm serving recipes",
        ):
            ppu_recipe.validate_paths(
                [bad_path], vllm_parser_mode="required", parser=_sample_vllm_parser()
            )

    # Positive control: non-empty single variant expands and passes parser check
    good = copy.deepcopy(glm_doc)
    good["variants"] = {"default": copy.deepcopy(glm_doc["variants"]["default"])}
    good["presets"] = {}
    good_path = tmp_path / "nonempty_variants.yaml"
    good_path.write_text(yaml.safe_dump(good), encoding="utf-8")
    report = ppu_recipe.validate_paths(
        [good_path], vllm_parser_mode="required", parser=_sample_vllm_parser()
    )
    assert report["checked_combinations"] == 1
    assert report["parser_checked"] is True


def test_single_node_dp_requires_all_ranks_local_and_admits_dp8_local8_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copytree(ROOT / "tests/ppu", tmp_path / "tests/ppu")
    monkeypatch.setattr(ppu_e2e_select, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ppu_recipe, "REPO_ROOT", tmp_path)

    m3_rec_path = tmp_path / "tests/ppu/recipes/MiniMaxAI/MiniMax-M3.yaml"
    m3_pol_path = tmp_path / "tests/ppu/policy/models/minimax-m3.yaml"
    m3_doc_orig = yaml.safe_load(m3_rec_path.read_text(encoding="utf-8"))
    m3_pol_doc = yaml.safe_load(m3_pol_path.read_text(encoding="utf-8"))
    m3_pol_doc["tiers"]["fast_sanity"]["preset"] = "dp_ep_low_latency"
    m3_pol_path.write_text(yaml.safe_dump(m3_pol_doc), encoding="utf-8")

    # 1. Counterexample (P1 r4694520699): DP8 with --data-parallel-size-local 4 on single_node
    #    fails recipe validation, policy resolution, automatic selection, and explicit selection.
    bad_dpl4 = copy.deepcopy(m3_doc_orig)
    bad_dpl4["presets"]["dp_ep_low_latency"]["args"] = [
        *bad_dpl4["presets"]["dp_ep_low_latency"]["args"],
        "--data-parallel-size-local",
        "4",
    ]
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"effective local DP size \(4\) to equal global DP size \(8\)",
    ):
        ppu_recipe.validate_serving_recipe_structure(bad_dpl4, context="bad_dpl4")

    m3_rec_path.write_text(yaml.safe_dump(bad_dpl4), encoding="utf-8")
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"effective local DP size \(4\) to equal global DP size \(8\)",
    ):
        ppu_recipe.validate_paths(
            [m3_rec_path], vllm_parser_mode="required", parser=_sample_vllm_parser()
        )
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"effective local DP size \(4\) to equal global DP size \(8\)",
    ):
        ppu_recipe.resolve_recipe(
            m3_rec_path, policy_path=m3_pol_path, tier="fast_sanity"
        )
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"effective local DP size \(4\) to equal global DP size \(8\)",
    ):
        ppu_e2e_select.build_recipe_matrix("", "fast_sanity")
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"effective local DP size \(4\) to equal global DP size \(8\)",
    ):
        ppu_e2e_select.build_recipe_matrix("minimax-m3", "fast_sanity")

    # Additional partial-local-DP counterexamples on single_node (external rank & ray span)
    for extra_args, extra_env, expected in (
        (
            ["--data-parallel-rank", "0"],
            {},
            r"effective local DP size \(1\) to equal global DP size \(8\)",
        ),
        (
            ["--data-parallel-external-lb"],
            {},
            r"--data-parallel-external-lb requires --data-parallel-rank",
        ),
        (
            ["--data-parallel-size-local", "16"],
            {},
            r"--data-parallel-size-local \(16\) cannot exceed --data-parallel-size \(8\)",
        ),
        (
            ["--data-parallel-backend", "ray"],
            {"VLLM_RAY_DP_PACK_STRATEGY": "span"},
            r"effective local DP size \(1\) to equal global DP size \(8\)",
        ),
    ):
        bad_extra = copy.deepcopy(m3_doc_orig)
        bad_extra["presets"]["dp_ep_low_latency"]["args"] = [
            *bad_extra["presets"]["dp_ep_low_latency"]["args"],
            *extra_args,
        ]
        if extra_env:
            bad_extra["presets"]["dp_ep_low_latency"]["env"] = {
                **bad_extra["presets"]["dp_ep_low_latency"].get("env", {}),
                **extra_env,
            }
        with pytest.raises(ppu_recipe.ValidationError, match=expected):
            ppu_recipe.validate_serving_recipe_structure(bad_extra, context="bad_extra")

    # 2. Positive control: DP8 with explicit --data-parallel-size-local 8 passes validation
    #    and is admitted by both automatic and explicit single-node CI selection.
    good_dpl8 = copy.deepcopy(m3_doc_orig)
    good_dpl8["presets"]["dp_ep_low_latency"]["args"] = [
        *good_dpl8["presets"]["dp_ep_low_latency"]["args"],
        "--data-parallel-size-local",
        "8",
    ]
    m3_rec_path.write_text(yaml.safe_dump(good_dpl8), encoding="utf-8")
    report = ppu_recipe.validate_paths(
        [m3_rec_path], vllm_parser_mode="required", parser=_sample_vllm_parser()
    )
    assert report["parser_checked"] is True
    resolved_ctrl = ppu_recipe.resolve_recipe(
        m3_rec_path, policy_path=m3_pol_path, tier="fast_sanity"
    )
    assert resolved_ctrl["ppu_count"] == 8
    assert resolved_ctrl["dp_size"] == 8
    assert resolved_ctrl["dp_size_local"] == 8

    auto_item = next(
        item
        for item in ppu_e2e_select.build_recipe_matrix("", "fast_sanity")["include"]
        if item["key"] == "minimax-m3"
    )
    explicit_items = ppu_e2e_select.build_recipe_matrix("minimax-m3", "fast_sanity")[
        "include"
    ]
    assert auto_item["ppu_count"] == 8
    assert auto_item["preset"] == "dp_ep_low_latency"
    assert len(explicit_items) == 1 and explicit_items[0]["ppu_count"] == 8

    # 3. Inherited os.environ VLLM_RAY_DP_PACK_STRATEGY=span with --data-parallel-backend ray
    #    and omitted --data-parallel-size-local is consistently rejected across validation,
    #    resolve_recipe, automatic/explicit selector, and _launch_resolved_server, whereas
    #    overriding VLLM_RAY_DP_PACK_STRATEGY=fill in recipe env passes.
    ray_doc = copy.deepcopy(m3_doc_orig)
    ray_doc["presets"]["dp_ep_low_latency"]["args"] = [
        *ray_doc["presets"]["dp_ep_low_latency"]["args"],
        "--data-parallel-backend",
        "ray",
    ]
    m3_rec_path.write_text(yaml.safe_dump(ray_doc), encoding="utf-8")
    pre_resolved_ray = ppu_recipe.resolve_recipe(
        m3_rec_path, policy_path=m3_pol_path, tier="fast_sanity"
    )
    assert pre_resolved_ray["dp_size_local"] == 8

    monkeypatch.setenv("VLLM_RAY_DP_PACK_STRATEGY", "span")
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"effective local DP size \(1\) to equal global DP size \(8\)",
    ):
        ppu_recipe.validate_serving_recipe_structure(ray_doc, context="ray_env_span")
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"effective local DP size \(1\) to equal global DP size \(8\)",
    ):
        ppu_recipe.resolve_recipe(
            m3_rec_path, policy_path=m3_pol_path, tier="fast_sanity"
        )
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"effective local DP size \(1\) to equal global DP size \(8\)",
    ):
        ppu_e2e_select.build_recipe_matrix("", "fast_sanity")
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"effective local DP size \(1\) to equal global DP size \(8\)",
    ):
        ppu_e2e_select.build_recipe_matrix("minimax-m3", "fast_sanity")
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"effective local DP size \(1\) to equal global DP size \(8\)",
    ):
        ppu_recipe_runner._launch_resolved_server(
            pre_resolved_ray, log_path=tmp_path / "never_launched.log"
        )

    ray_fill = copy.deepcopy(ray_doc)
    ray_fill["presets"]["dp_ep_low_latency"]["env"] = {
        **ray_fill["presets"]["dp_ep_low_latency"].get("env", {}),
        "VLLM_RAY_DP_PACK_STRATEGY": "fill",
    }
    m3_rec_path.write_text(yaml.safe_dump(ray_fill), encoding="utf-8")
    resolved_fill = ppu_recipe.resolve_recipe(
        m3_rec_path, policy_path=m3_pol_path, tier="fast_sanity"
    )
    assert resolved_fill["dp_size_local"] == 8
    monkeypatch.delenv("VLLM_RAY_DP_PACK_STRATEGY")

    # 4. Multi-node DP32+EP with --data-parallel-hybrid-lb and omitted --data-parallel-size-local
    #    infers local DP = 8 per node first (matching pinned vllm arg_utils.py:2476-2520) and passes.
    glm_path = tmp_path / "tests/ppu/recipes/zai-org/GLM-5.3.yaml"
    glm_hybrid = yaml.safe_load(glm_path.read_text(encoding="utf-8"))
    glm_hybrid["presets"]["decode_ep"]["args"] = [
        *glm_hybrid["presets"]["decode_ep"]["args"],
        "--data-parallel-hybrid-lb",
    ]
    glm_path.write_text(yaml.safe_dump(glm_hybrid), encoding="utf-8")
    glm_hybrid_report = ppu_recipe.validate_paths(
        [glm_path], vllm_parser_mode="required", parser=_sample_vllm_parser()
    )
    assert glm_hybrid_report["checked_combinations"] == 14
    glm_hybrid_ep = ppu_recipe.resolve_recipe(glm_path, preset="decode_ep")
    assert glm_hybrid_ep["ppu_count"] == 32
    assert glm_hybrid_ep["dp_size"] == 32
    assert glm_hybrid_ep["dp_size_local"] == 8


def test_multi_node_ep_requires_effective_expert_parallel(tmp_path: Path) -> None:
    glm_path = RECIPES_DIR / "zai-org" / "GLM-5.3.yaml"
    glm_doc = yaml.safe_load(glm_path.read_text(encoding="utf-8"))

    # Positive control: checked-in GLM-5.3 decode_ep (DP32 + --enable-expert-parallel) passes
    ctrl_report = ppu_recipe.validate_paths(
        [glm_path], vllm_parser_mode="required", parser=_sample_vllm_parser()
    )
    assert ctrl_report["checked_combinations"] == 14
    ctrl_ep = ppu_recipe.resolve_recipe(glm_path, preset="decode_ep")
    assert ctrl_ep["ppu_count"] == 32
    assert "--enable-expert-parallel" in ctrl_ep["flags"]

    # Counterexample 1 (P2 r4694520719): removing --enable-expert-parallel from decode_ep fails
    no_ep = copy.deepcopy(glm_doc)
    no_ep["presets"]["decode_ep"]["args"] = [
        tok
        for tok in no_ep["presets"]["decode_ep"]["args"]
        if tok != "--enable-expert-parallel"
    ]
    with pytest.raises(
        ppu_recipe.ValidationError,
        match="multi_node_ep topology requires effective --enable-expert-parallel",
    ):
        ppu_recipe.validate_serving_recipe_structure(no_ep, context="no_ep")

    no_ep_path = tmp_path / "glm_no_ep.yaml"
    no_ep_path.write_text(yaml.safe_dump(no_ep), encoding="utf-8")
    with pytest.raises(
        ppu_recipe.ValidationError,
        match="multi_node_ep topology requires effective --enable-expert-parallel",
    ):
        ppu_recipe.validate_paths(
            [no_ep_path], vllm_parser_mode="required", parser=_sample_vllm_parser()
        )

    # Counterexample 2: disabling EP via --no-enable-expert-parallel or --disable-expert-parallel fails
    for neg_flag in ("--no-enable-expert-parallel", "--disable-expert-parallel"):
        neg_ep = copy.deepcopy(glm_doc)
        neg_ep["presets"]["decode_ep"]["args"] = [
            *neg_ep["presets"]["decode_ep"]["args"],
            neg_flag,
        ]
        with pytest.raises(
            ppu_recipe.ValidationError,
            match="multi_node_ep topology requires effective --enable-expert-parallel",
        ):
            ppu_recipe.validate_serving_recipe_structure(neg_ep, context=neg_flag)


def test_vllm_dp_env_fallback_crossnode_tp_and_hybrid_headless_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copytree(ROOT / "tests/ppu", tmp_path / "tests/ppu")
    monkeypatch.setattr(ppu_e2e_select, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ppu_recipe, "REPO_ROOT", tmp_path)

    glm_rec_path = tmp_path / "tests/ppu/recipes/zai-org/GLM-5.3.yaml"
    glm_pol_path = tmp_path / "tests/ppu/policy/models/glm-5.3.yaml"
    glm_orig = yaml.safe_load(glm_rec_path.read_text(encoding="utf-8"))

    # 1. P1 (r4695877063): GLM-5.3 TP4 prefill with no --data-parallel-size flag and
    #    VLLM_DP_SIZE=2 (in preset env or inherited os.environ) activates ParallelConfig
    #    env fallback (global DP=2, local DP=1) and must be rejected everywhere.
    bad_env_dp2 = copy.deepcopy(glm_orig)
    bad_env_dp2["presets"]["prefill"]["env"] = {
        **bad_env_dp2["presets"]["prefill"].get("env", {}),
        "VLLM_DP_SIZE": "2",
    }
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"effective local DP size \(1\) to equal global DP size \(2\)",
    ):
        ppu_recipe.validate_serving_recipe_structure(bad_env_dp2, context="env_dp2")

    glm_rec_path.write_text(yaml.safe_dump(bad_env_dp2), encoding="utf-8")
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"effective local DP size \(1\) to equal global DP size \(2\)",
    ):
        ppu_recipe.validate_paths(
            [glm_rec_path], vllm_parser_mode="required", parser=_sample_vllm_parser()
        )
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"effective local DP size \(1\) to equal global DP size \(2\)",
    ):
        ppu_recipe.resolve_recipe(
            glm_rec_path, policy_path=glm_pol_path, tier="fast_sanity"
        )
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"effective local DP size \(1\) to equal global DP size \(2\)",
    ):
        ppu_e2e_select.build_recipe_matrix("", "fast_sanity")
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"effective local DP size \(1\) to equal global DP size \(2\)",
    ):
        ppu_e2e_select.build_recipe_matrix("glm-5.3", "fast_sanity")

    # Inherited os.environ VLLM_DP_SIZE=2 on clean recipe is also rejected across
    # resolve_recipe, selector, and _launch_resolved_server.
    glm_rec_path.write_text(yaml.safe_dump(glm_orig), encoding="utf-8")
    clean_resolved = ppu_recipe.resolve_recipe(
        glm_rec_path, policy_path=glm_pol_path, tier="fast_sanity"
    )
    monkeypatch.setenv("VLLM_DP_SIZE", "2")
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"effective local DP size \(1\) to equal global DP size \(2\)",
    ):
        ppu_recipe.resolve_recipe(
            glm_rec_path, policy_path=glm_pol_path, tier="fast_sanity"
        )
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"effective local DP size \(1\) to equal global DP size \(2\)",
    ):
        ppu_e2e_select.build_recipe_matrix("glm-5.3", "fast_sanity")
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"effective local DP size \(1\) to equal global DP size \(2\)",
    ):
        ppu_recipe_runner._launch_resolved_server(
            clean_resolved, log_path=tmp_path / "never_started.log"
        )
    monkeypatch.delenv("VLLM_DP_SIZE")

    # Conflicting recipe VLLM_DP_SIZE alongside CLI --data-parallel-size > 1 fails closed
    conflict_dp = copy.deepcopy(glm_orig)
    conflict_dp["presets"]["pd_decode"]["env"] = {
        **conflict_dp["presets"]["pd_decode"]["env"],
        "VLLM_DP_SIZE": "2",
    }
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"recipe env VLLM_DP_SIZE \(2\) conflicts with --data-parallel-size \(8\)",
    ):
        ppu_recipe.validate_serving_recipe_structure(
            conflict_dp, context="conflict_env_dp"
        )

    # Positive control: GLM-5.3 TP4 prefill with VLLM_DP_SIZE=1 agrees on DP=1 / local DP=1 (4 cards)
    good_env_dp1 = copy.deepcopy(glm_orig)
    good_env_dp1["presets"]["prefill"]["env"] = {
        **good_env_dp1["presets"]["prefill"].get("env", {}),
        "VLLM_DP_SIZE": "1",
    }
    glm_rec_path.write_text(yaml.safe_dump(good_env_dp1), encoding="utf-8")
    res_dp1 = ppu_recipe.resolve_recipe(
        glm_rec_path, policy_path=glm_pol_path, tier="fast_sanity"
    )
    assert res_dp1["ppu_count"] == 4
    assert res_dp1["dp_size"] == 1
    assert res_dp1["dp_size_local"] == 1
    sel_dp1 = ppu_e2e_select.build_recipe_matrix("glm-5.3", "fast_sanity")["include"]
    assert len(sel_dp1) == 1 and sel_dp1[0]["ppu_count"] == 4

    # 2. P2 (r4695877083): 4x8 multi_node_ep with DP2 x TP16 (each TP16 replica spans 2 nodes)
    #    resolves to 32 devices and local DP=1 both when omitted and when explicitly set to 1.
    for explicit_dpl in (None, "1"):
        dp2_tp16 = copy.deepcopy(glm_orig)
        ep_args = [
            tok
            for tok in dp2_tp16["presets"]["decode_ep"]["args"]
            if tok not in {"--data-parallel-size", "32"}
        ]
        ep_args.extend(["--data-parallel-size", "2", "--tensor-parallel-size", "16"])
        if explicit_dpl is not None:
            ep_args.extend(["--data-parallel-size-local", explicit_dpl])
        dp2_tp16["presets"]["decode_ep"]["args"] = ep_args
        glm_rec_path.write_text(yaml.safe_dump(dp2_tp16), encoding="utf-8")
        rep = ppu_recipe.validate_paths(
            [glm_rec_path], vllm_parser_mode="required", parser=_sample_vllm_parser()
        )
        assert rep["checked_combinations"] == 14
        res_ep = ppu_recipe.resolve_recipe(glm_rec_path, preset="decode_ep")
        assert res_ep["ppu_count"] == 32
        assert res_ep["dp_size"] == 2
        assert res_ep["dp_size_local"] == 1

    bad_dp2_tp16_dpl2 = copy.deepcopy(dp2_tp16)
    bad_dp2_tp16_dpl2["presets"]["decode_ep"]["args"] = [
        *dp2_tp16["presets"]["decode_ep"]["args"][:-2],
        "--data-parallel-size-local",
        "2",
    ]
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"multi_node_ep --data-parallel-size-local \(2\) must match per-node DP layout \(1\)",
    ):
        ppu_recipe.validate_serving_recipe_structure(
            bad_dp2_tp16_dpl2, context="bad_dp2_tp16_dpl2"
        )

    # 3. P2 (r4695877100): AUTHORING.md qualifies --headless vs --data-parallel-hybrid-lb,
    #    and combining --data-parallel-hybrid-lb with --headless is rejected.
    authoring_text = (ROOT / "tests/ppu/AUTHORING.md").read_text(encoding="utf-8")
    assert "--data-parallel-hybrid-lb" in authoring_text
    assert "without `--headless`" in authoring_text

    bad_hybrid_headless = copy.deepcopy(glm_orig)
    bad_hybrid_headless["presets"]["decode_ep"]["args"] = [
        *bad_hybrid_headless["presets"]["decode_ep"]["args"],
        "--data-parallel-hybrid-lb",
        "--headless",
    ]
    with pytest.raises(
        ppu_recipe.ValidationError,
        match="--data-parallel-hybrid-lb is not applicable in --headless mode",
    ):
        ppu_recipe.validate_serving_recipe_structure(
            bad_hybrid_headless, context="hybrid_headless"
        )

    # 4. R1 P1-1: Single-node out-of-bounds VLLM_DP_RANK_LOCAL=1 (with VLLM_DP_SIZE=1)
    #    and multi-node env-only VLLM_DP_SIZE=2 without CLI --data-parallel-size fail closed;
    #    VLLM_DP_RANK_LOCAL=0 and CLI --data-parallel-size 2 controls pass.
    bad_rank_local = copy.deepcopy(glm_orig)
    bad_rank_local["presets"]["prefill"]["env"] = {
        **bad_rank_local["presets"]["prefill"].get("env", {}),
        "VLLM_DP_SIZE": "1",
        "VLLM_DP_RANK_LOCAL": "1",
    }
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"VLLM_DP_RANK_LOCAL \(1\) must be in \[0, 0\]",
    ):
        ppu_recipe.validate_serving_recipe_structure(
            bad_rank_local, context="bad_rank_local"
        )

    glm_rec_path.write_text(yaml.safe_dump(glm_orig), encoding="utf-8")
    monkeypatch.setenv("VLLM_DP_SIZE", "1")
    monkeypatch.setenv("VLLM_DP_RANK_LOCAL", "1")
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"VLLM_DP_RANK_LOCAL \(1\) must be in \[0, 0\]",
    ):
        ppu_recipe.resolve_recipe(
            glm_rec_path, policy_path=glm_pol_path, tier="fast_sanity"
        )
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"VLLM_DP_RANK_LOCAL \(1\) must be in \[0, 0\]",
    ):
        ppu_e2e_select.build_recipe_matrix("glm-5.3", "fast_sanity")
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"VLLM_DP_RANK_LOCAL \(1\) must be in \[0, 0\]",
    ):
        ppu_recipe_runner._launch_resolved_server(
            clean_resolved, log_path=tmp_path / "never_started_rank_local.log"
        )
    monkeypatch.setenv("VLLM_DP_RANK_LOCAL", "0")
    ok_rank_local = ppu_recipe.resolve_recipe(
        glm_rec_path, policy_path=glm_pol_path, tier="fast_sanity"
    )
    assert ok_rank_local["ppu_count"] == 4
    monkeypatch.delenv("VLLM_DP_SIZE")
    monkeypatch.delenv("VLLM_DP_RANK_LOCAL")

    bad_mn_env_dp = copy.deepcopy(glm_orig)
    ep_no_cli_dp = [
        tok
        for tok in bad_mn_env_dp["presets"]["decode_ep"]["args"]
        if tok not in {"--data-parallel-size", "32"}
    ]
    ep_no_cli_dp.extend(["--tensor-parallel-size", "16"])
    bad_mn_env_dp["presets"]["decode_ep"]["args"] = ep_no_cli_dp
    bad_mn_env_dp["presets"]["decode_ep"]["env"] = {
        **bad_mn_env_dp["presets"]["decode_ep"]["env"],
        "VLLM_DP_SIZE": "2",
    }
    with pytest.raises(
        ppu_recipe.ValidationError,
        match=r"multi-node topology \(nodes=4\) requires CLI --data-parallel-size rather than VLLM_DP_SIZE=2",
    ):
        ppu_recipe.validate_serving_recipe_structure(
            bad_mn_env_dp, context="bad_mn_env_dp"
        )

    # 5. R1 P1-2: ExecutorWithExternalLauncher class path, external_launcher, and
    #    underscore CLI flag spellings fail closed across validation, resolve, selector, and runner.
    for exec_val in (
        "external_launcher",
        "vllm.v1.executor.uniproc_executor.ExecutorWithExternalLauncher",
        "uni",
    ):
        bad_exec = copy.deepcopy(glm_orig)
        bad_exec["presets"]["prefill"]["args"] = [
            *bad_exec["presets"]["prefill"].get("args", []),
            "--distributed-executor-backend",
            exec_val,
        ]
        with pytest.raises(
            ppu_recipe.ValidationError, match="--distributed-executor-backend"
        ):
            ppu_recipe.validate_serving_recipe_structure(bad_exec, context=exec_val)
        glm_rec_path.write_text(yaml.safe_dump(bad_exec), encoding="utf-8")
        with pytest.raises(
            ppu_recipe.ValidationError, match="--distributed-executor-backend"
        ):
            ppu_e2e_select.build_recipe_matrix("glm-5.3", "fast_sanity")

    good_exec_mp = copy.deepcopy(glm_orig)
    good_exec_mp["presets"]["prefill"]["args"] = [
        *good_exec_mp["presets"]["prefill"].get("args", []),
        "--distributed-executor-backend",
        "mp",
    ]
    glm_rec_path.write_text(yaml.safe_dump(good_exec_mp), encoding="utf-8")
    assert (
        ppu_recipe.resolve_recipe(
            glm_rec_path, policy_path=glm_pol_path, tier="fast_sanity"
        )["ppu_count"]
        == 4
    )

    # Multi-node (nodes > 1) rejects --distributed-executor-backend ray / external_launcher /
    # ExecutorWithExternalLauncher (matching pinned parallel.py:1199-1207), rejects 'uni' when
    # replica world > 1 (DP2 x TP16), and accepts 'mp' or 'uni' when replica world == 1 (DP32).
    for bad_mn_exec in (
        "ray",
        "external_launcher",
        "vllm.v1.executor.uniproc_executor.ExecutorWithExternalLauncher",
    ):
        bad_mn_doc = copy.deepcopy(glm_orig)
        bad_mn_doc["presets"]["decode_ep"]["args"] = [
            *bad_mn_doc["presets"]["decode_ep"]["args"],
            "--distributed-executor-backend",
            bad_mn_exec,
        ]
        with pytest.raises(
            ppu_recipe.ValidationError,
            match=r"--distributed-executor-backend must be one of \['mp', 'uni'\] for nodes=4",
        ):
            ppu_recipe.validate_serving_recipe_structure(
                bad_mn_doc, context=f"mn_{bad_mn_exec}"
            )
        tampered_mn = ppu_recipe.resolve_recipe(
            ROOT / "tests/ppu/recipes/zai-org/GLM-5.3.yaml", preset="decode_ep"
        )
        tampered_mn["flags"] = [
            *tampered_mn["flags"],
            "--distributed-executor-backend",
            bad_mn_exec,
        ]
        with pytest.raises(
            ppu_recipe.ValidationError,
            match=r"--distributed-executor-backend must be one of \['mp', 'uni'\] for nodes=4",
        ):
            ppu_recipe_runner._launch_resolved_server(
                tampered_mn, log_path=tmp_path / "never_started_mn_exec.log"
            )

    bad_mn_uni_tp16 = copy.deepcopy(dp2_tp16)
    bad_mn_uni_tp16["presets"]["decode_ep"]["args"] = [
        *dp2_tp16["presets"]["decode_ep"]["args"],
        "--distributed-executor-backend",
        "uni",
    ]
    with pytest.raises(
        ppu_recipe.ValidationError,
        match="--distributed-executor-backend 'uni' requires per-replica world size 1, got 16",
    ):
        ppu_recipe.validate_serving_recipe_structure(
            bad_mn_uni_tp16, context="mn_uni_tp16"
        )

    for good_mn_exec in ("mp", "uni"):
        good_mn_doc = copy.deepcopy(glm_orig)
        good_mn_doc["presets"]["decode_ep"]["args"] = [
            *good_mn_doc["presets"]["decode_ep"]["args"],
            "--distributed-executor-backend",
            good_mn_exec,
        ]
        glm_rec_path.write_text(yaml.safe_dump(good_mn_doc), encoding="utf-8")
        assert (
            ppu_recipe.resolve_recipe(glm_rec_path, preset="decode_ep")["ppu_count"]
            == 32
        )

    for bad_us_args in (
        ["--data_parallel_size", "2"],
        ["--distributed_executor_backend", "external_launcher"],
        ["--enable_prefix_caching"],
    ):
        bad_us = copy.deepcopy(glm_orig)
        bad_us["presets"]["prefill"]["args"] = [
            *bad_us["presets"]["prefill"].get("args", []),
            *bad_us_args,
        ]
        with pytest.raises(
            ppu_recipe.ValidationError, match="canonical hyphenated spelling"
        ):
            ppu_recipe.validate_serving_recipe_structure(bad_us, context="bad_us")
        glm_rec_path.write_text(yaml.safe_dump(bad_us), encoding="utf-8")
        with pytest.raises(
            ppu_recipe.ValidationError, match="canonical hyphenated spelling"
        ):
            ppu_recipe.resolve_recipe(
                glm_rec_path, policy_path=glm_pol_path, tier="fast_sanity"
            )
        with pytest.raises(
            ppu_recipe.ValidationError, match="canonical hyphenated spelling"
        ):
            ppu_e2e_select.build_recipe_matrix("glm-5.3", "fast_sanity")
        tampered = copy.deepcopy(clean_resolved)
        tampered["flags"] = [*tampered["flags"], *bad_us_args]
        with pytest.raises(
            ppu_recipe.ValidationError, match="canonical hyphenated spelling"
        ):
            ppu_recipe_runner._launch_resolved_server(
                tampered, log_path=tmp_path / "never_started_us.log"
            )

    # Review #6066321892: external-LB multi_node_ep accepts both omitted and explicit
    # --data-parallel-size-local 1 on the same external rank (matching pinned arg_utils.py:2501-2535).
    for ext_args in (
        ["--data-parallel-external-lb", "--data-parallel-rank", "0"],
        [
            "--data-parallel-external-lb",
            "--data-parallel-rank",
            "0",
            "--data-parallel-size-local",
            "1",
        ],
        ["--data-parallel-rank", "0", "--data-parallel-size-local", "1"],
        ["--data-parallel-hybrid-lb", "--data-parallel-size-local", "1"],
        ["--data-parallel-start-rank", "0", "--data-parallel-size-local", "1"],
    ):
        ext_doc = copy.deepcopy(glm_orig)
        ext_doc["presets"]["decode_ep"]["args"] = [
            *ext_doc["presets"]["decode_ep"]["args"],
            *ext_args,
        ]
        ppu_recipe.validate_serving_recipe_structure(ext_doc, context="ext_lb_mn")
        glm_rec_path.write_text(yaml.safe_dump(ext_doc), encoding="utf-8")
        report = ppu_recipe.validate_paths([glm_rec_path], vllm_parser_mode="skip")
        assert report["checked_combinations"] > 0
        ext_res = ppu_recipe.resolve_recipe(glm_rec_path, preset="decode_ep")
        assert (ext_res["ppu_count"], ext_res["dp_size"], ext_res["dp_size_local"]) == (
            32,
            32,
            1,
        )

    for bad_ext_args, err_pat in (
        (
            [
                "--data-parallel-external-lb",
                "--data-parallel-rank",
                "0",
                "--data-parallel-size-local",
                "8",
            ],
            "external data-parallel mode requires --data-parallel-size-local 1 or omitted",
        ),
        (
            ["--data-parallel-rank", "0"],
            "external data-parallel mode requires --data-parallel-size-local 1 or omitted",
        ),
        (
            ["--data-parallel-start-rank", "32", "--data-parallel-size-local", "1"],
            "--data-parallel-start-rank must be in",
        ),
        (
            ["--data-parallel-start-rank", "bad", "--data-parallel-size-local", "8"],
            "--data-parallel-start-rank must be an integer",
        ),
    ):
        bad_ext_doc = copy.deepcopy(glm_orig)
        bad_ext_doc["presets"]["decode_ep"]["args"] = [
            *bad_ext_doc["presets"]["decode_ep"]["args"],
            *bad_ext_args,
        ]
        with pytest.raises(ppu_recipe.ValidationError, match=err_pat):
            ppu_recipe.validate_serving_recipe_structure(
                bad_ext_doc, context="bad_ext_lb_mn"
            )
        glm_rec_path.write_text(yaml.safe_dump(bad_ext_doc), encoding="utf-8")
        with pytest.raises(ppu_recipe.ValidationError, match=err_pat):
            ppu_recipe.resolve_recipe(glm_rec_path, preset="decode_ep")
