#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validates Step5 TP4/PP2 with the current SAIL sources and installed native extensions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import sysconfig
from functools import partial
from pathlib import Path

# Spawned subprocesses also execute this, so not only the main process uses the new sources.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import vllm_sail  # noqa: E402

NATIVE = Path(sysconfig.get_path("platlib")) / "vllm_sail"
if str(NATIVE) not in vllm_sail.__path__:
    vllm_sail.__path__.append(str(NATIVE))


def serialize_logprobs(values):
    if values is None:
        return None
    return {
        str(token): {
            "logprob": value.logprob,
            "rank": value.rank,
            "decoded_token": value.decoded_token,
        }
        for token, value in values.items()
    }


def first_difference(first, second):
    for index, (left, right) in enumerate(zip(first, second, strict=False)):
        if left != right:
            return index
    return min(len(first), len(second)) if len(first) != len(second) else None


def report_difference(prompt, first, second, round_id):
    left, right = first.outputs[0], second.outputs[0]
    index = first_difference(left.token_ids, right.token_ids)
    if index is None:
        return
    print(
        json.dumps(
            {
                "event": "first_difference",
                "prompt": prompt,
                "rounds": [1, round_id],
                "token_index": index,
                "generated_token_number": index + 1,
                "common_prefix_token_ids": left.token_ids[:index],
                "token_ids": [
                    ids[index] if index < len(ids) else None
                    for ids in (left.token_ids, right.token_ids)
                ],
                "logprobs": [
                    serialize_logprobs(output.logprobs[index])
                    if output.logprobs is not None and index < len(output.logprobs)
                    else None
                    for output in (left, right)
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def summarize_log(path):
    """Reads preserved evidence without importing models or running device computation."""
    if path.is_dir():
        seen = set()
        counts = {
            field: {"checked": 0, "changed": 0}
            for field in ("replay", "input_mutation")
        }
        for log in sorted(path.glob("rank-*.jsonl")):
            for line in log.read_text().splitlines():
                record = json.loads(line)
                for field, count in counts.items():
                    if field in record:
                        count["checked"] += 1
                        count["changed"] += any(
                            not row["equal"] for row in record[field].values()
                        )
                fields = ("inputs", "outputs", "replay")
                changed = [
                    field
                    for field in fields
                    if any(not row["equal"] for row in record.get(field, {}).values())
                ]
                if not changed:
                    continue
                key = (
                    record["rank"],
                    record["round"],
                    record["step"],
                    "replay" in changed,
                )
                if key not in seen:
                    seen.add(key)
                    print(
                        json.dumps(
                            {**record, "changed": list(changed)}, ensure_ascii=False
                        )
                    )
        print(json.dumps({"event": "replay_summary", "directory": str(path), **counts}))
        return
    results = {}
    passed = False
    for line in path.read_text().splitlines():
        passed |= "STEP5_MEGAMOE_PP2_PASS" in line
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        if "round" in record:
            results.setdefault(record["prompt"], {})[record["round"]] = record
    print(json.dumps({"log": str(path), "passed": passed, "prompts": len(results)}))
    for prompt, rounds in results.items():
        first = rounds[min(rounds)]
        for round_id, second in sorted(rounds.items()):
            if round_id == min(rounds):
                continue
            index = first_difference(first["token_ids"], second["token_ids"])
            left, right = first.get("logprobs"), second.get("logprobs")
            drift = None
            if left is not None and right is not None:
                for step, (a, b) in enumerate(zip(left, right, strict=False)):
                    if index is not None and step > index:
                        break
                    delta = max(
                        (
                            abs(a[token]["logprob"] - b[token]["logprob"])
                            for token in a.keys() & b.keys()
                        ),
                        default=0,
                    )
                    if delta:
                        drift = {"token_index": step, "max_common_logprob_delta": delta}
                        break
            print(
                json.dumps(
                    {
                        "prompt": prompt,
                        "rounds": [min(rounds), round_id],
                        "same_prompt_tokens": first.get("prompt_token_ids")
                        == second.get("prompt_token_ids"),
                        "token_index": index,
                        "first_logprob_drift": drift,
                        "candidates": [row[index] for row in (left, right)]
                        if index is not None
                        and left
                        and right
                        and index < min(len(left), len(right))
                        else None,
                    },
                    ensure_ascii=False,
                )
            )


def configure_model_diagnostics(model, directory, replay, steps):
    """Compares per-layer results only in standalone eager diagnostic runs; does not change the normal inference path."""
    import types

    import torch
    import torch.distributed as dist

    rank = dist.get_rank()
    log = Path(directory) / f"rank-{rank}.jsonl"
    selected = {"Step5DecoderLayer", "PPUStep5MegaMoEExperts", "PPUStep5BF16MoERunner"}

    def snapshot(value, prefix=""):
        if isinstance(value, torch.Tensor):
            return {prefix: value.detach().cpu().clone()}
        if isinstance(value, tuple | list):
            value = dict(enumerate(value))
        if isinstance(value, dict):
            return {
                name: tensor
                for key, item in value.items()
                for name, tensor in snapshot(item, f"{prefix}.{key}").items()
            }
        return {}

    def compare(first, second):
        result = {}
        for key in first.keys() | second.keys():
            a, b = first.get(key), second.get(key)
            same_shape = a is not None and b is not None and a.shape == b.shape
            result[key] = {
                "equal": same_shape and torch.equal(a, b),
                "first_shape": list(a.shape) if a is not None else None,
                "current_shape": list(b.shape) if b is not None else None,
            }
            if same_shape and a.numel():
                result[key].update(
                    max_abs=(a.double() - b.double()).abs().max().item(),
                    changed=int((a != b).sum()),
                )
        return result

    def wrap(module, name, original):
        def forward(self, *args, **kwargs):
            round_id = self._step5_diag_round
            step = self._step5_diag_step
            self._step5_diag_step += 1
            if round_id == 0 or step >= steps:
                return original(*args, **kwargs)
            inputs = snapshot({"args": args, "kwargs": kwargs})
            output = original(*args, **kwargs)
            outputs = snapshot(output)
            record = {
                "rank": rank,
                "round": round_id,
                "step": step,
                "module": name,
                "class": type(self).__name__,
                "input_shapes": {
                    key: list(value.shape) for key, value in inputs.items()
                },
            }
            if type(self).__name__ == "Step5DecoderLayer" and self.layer_idx in (0, 46):
                record["positions"] = inputs[".args.0"].tolist()
            if round_id == 1:
                self._step5_diag_refs[step] = (inputs, outputs)
            elif step in self._step5_diag_refs:
                first_inputs, first_outputs = self._step5_diag_refs[step]
                record.update(
                    inputs=compare(first_inputs, inputs),
                    outputs=compare(first_outputs, outputs),
                )
            if replay and type(self).__name__ != "Step5DecoderLayer":
                output = output.clone()
                repeated = original(*args, **kwargs)
                record["replay"] = compare(outputs, snapshot(repeated))
                record["input_mutation"] = compare(
                    inputs, snapshot({"args": args, "kwargs": kwargs})
                )
            with log.open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            return output

        return types.MethodType(forward, module)

    count = 0
    for name, module in model.named_modules():
        if type(module).__name__ not in selected:
            continue
        module._step5_diag_round = 0
        module._step5_diag_step = 0
        module._step5_diag_refs = {}
        module.forward = wrap(module, name, module.forward)
        count += 1
    return {"rank": rank, "instrumented_modules": count}


def set_diagnostic_round(model, round_id):
    for module in model.modules():
        if hasattr(module, "_step5_diag_round"):
            module._step5_diag_round = round_id
            module._step5_diag_step = 0


def generate_fixed_batch(llm, prompts, params):
    """Pauses background scheduling and resumes after everything is enqueued, isolating frontend arrival timing."""
    client = llm.llm_engine.engine_core
    client.call_utility("pause_generation", "keep", False)
    try:
        llm.enqueue(prompts, params, use_tqdm=False)
    finally:
        client.call_utility("resume_scheduler")
    return llm.wait_for_completion()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--summarize-log", type=Path, action="append")
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--trace-steps", type=int, default=4)
    parser.add_argument("--replay-moe", action="store_true")
    parser.add_argument("--backend", default="deep_gemm_mega_moe")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--scheduling", choices=("async", "sync"), default="async")
    parser.add_argument("--admission", choices=("live", "paused"), default="live")
    parser.add_argument(
        "--prompt",
        action="append",
        help="repeatable; defaults to the three original inputs",
    )
    parser.add_argument("--logprobs", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    if args.summarize_log:
        for path in args.summarize_log:
            summarize_log(path)
        return
    if not args.model:
        parser.error("either model or summarize-log is required")
    if args.replay_moe and args.trace_dir is None:
        parser.error("replay-moe also requires trace-dir")
    if args.trace_dir is not None:
        if not args.eager or args.scheduling != "sync" or args.trace_steps < 1:
            parser.error(
                "per-layer diagnostics support eager + sync only; trace-steps must be positive"
            )
        args.trace_dir = args.trace_dir.resolve()
        if not args.trace_dir.is_relative_to(ROOT / "build"):
            parser.error("trace-dir must live under the SAIL build directory")
        args.trace_dir.mkdir(parents=True, exist_ok=False)
    if args.rounds < 2 or not 0 <= args.logprobs <= 20:
        parser.error("rounds must be at least 2; logprobs must be between 0 and 20")
    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        stream = os.open(args.log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(stream, 1)
        os.dup2(stream, 2)
        os.close(stream)
    for name, value in {
        "VLLM_SAIL_STEP5_FUSED_NORM": "1",
        "VLLM_SAIL_STEP5_DSA_SCORES": "v1",
        "VLLM_SAIL_STEP5_DSA_PREFILL_BACKEND": "native",
        "VLLM_SAIL_STEP5_DSA_PREFILL_BATCH_ATTENTION": "1",
        "VLLM_SAIL_STEP5_DSA_SELECTOR_PPU": "1",
        "DG_USE_MOE_DYNAMIC_TILE": "1",
        "NCCL_CUMEM_ENABLE": "1",
        "VLLM_ENGINE_READY_TIMEOUT_S": "60000",
        "VLLM_PROCESS_GROUP_TIMEOUT_M": "60",
    }.items():
        os.environ.setdefault(name, value)

    from vllm import LLM, SamplingParams

    print(f"SAIL source: {vllm_sail.__file__}; native: {NATIVE}", flush=True)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
        data_parallel_size=1,
        distributed_executor_backend="mp",
        enable_expert_parallel=True,
        moe_backend=args.backend,
        block_size=64,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=0.90,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        async_scheduling=args.scheduling == "async",
        disable_custom_all_reduce=True,
        enable_prefix_caching=False,
        sparse_indexer_topk_backend="deep_select",
        enforce_eager=args.eager,
        seed=0,
    )
    params = SamplingParams(
        temperature=0, max_tokens=64, logprobs=args.logprobs or None
    )
    prompts = args.prompt or [
        "Hello.",
        "Compute 17 + 25.",
        "Explain pipeline parallelism in one sentence.",
    ]
    config = llm.llm_engine.vllm_config
    print(
        json.dumps(
            {
                "event": "validation_config",
                "args": {
                    name: str(value) if isinstance(value, Path) else value
                    for name, value in vars(args).items()
                },
                "sampling_params": repr(params),
                "input_mode": "raw_completion",
                "resolved_async_scheduling": config.scheduler_config.async_scheduling,
                "resolved_moe_backend": config.kernel_config.moe_backend,
                "env": {
                    name: value
                    for name, value in sorted(os.environ.items())
                    if name.startswith(
                        (
                            "VLLM_SAIL_STEP5_",
                            "VLLM_PPU_STEP5_",
                            "VLLM_SAIL_MEGA_MOE_",
                            "VLLM_PPU_MEGA_MOE_",
                            "DG_",
                        )
                    )
                    or name
                    in {
                        "VLLM_USE_V2_MODEL_RUNNER",
                        "NCCL_CUMEM_ENABLE",
                        "CUDA_VISIBLE_DEVICES",
                        "VLLM_BATCH_INVARIANT",
                    }
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.trace_dir is not None:
        print(
            json.dumps(
                {
                    "event": "diagnostics_installed",
                    "workers": llm.apply_model(
                        partial(
                            configure_model_diagnostics,
                            directory=str(args.trace_dir),
                            replay=args.replay_moe,
                            steps=args.trace_steps,
                        )
                    ),
                }
            ),
            flush=True,
        )
    rounds = []
    for round_id in range(1, args.rounds + 1):
        if args.trace_dir is not None:
            llm.apply_model(partial(set_diagnostic_round, round_id=round_id))
        results = (
            generate_fixed_batch(llm, prompts, params)
            if args.admission == "paused"
            else llm.generate(prompts, params)
        )
        rounds.append(results)
        for prompt, result in zip(prompts, results, strict=True):
            assert result.outputs[0].token_ids, "generation output is empty"
            print(
                json.dumps(
                    {
                        "round": round_id,
                        "prompt": prompt,
                        "text": result.outputs[0].text,
                        "token_ids": result.outputs[0].token_ids,
                        "request_id": result.request_id,
                        "prompt_token_ids": result.prompt_token_ids,
                        "finish_reason": result.outputs[0].finish_reason,
                        "logprobs": [
                            serialize_logprobs(row)
                            for row in result.outputs[0].logprobs
                        ]
                        if result.outputs[0].logprobs is not None
                        else None,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    first = rounds[0]
    for round_id, second in enumerate(rounds[1:], start=2):
        for prompt, left, right in zip(prompts, first, second, strict=True):
            report_difference(prompt, left, right, round_id)
    for second in rounds[1:]:
        assert [x.outputs[0].token_ids for x in first] == [
            x.outputs[0].token_ids for x in second
        ], "repeated greedy generations disagree"
    print("STEP5_MEGAMOE_PP2_PASS", flush=True)


if __name__ == "__main__":
    main()
