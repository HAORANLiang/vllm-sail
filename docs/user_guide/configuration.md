# Configuration

Set environment variables before starting vLLM so the server and worker processes
inherit the same configuration. Runtime defaults are defined in
[`vllm_sail/envs.py`](../../vllm_sail/envs.py).

Every `VLLM_SAIL_*` runtime variable accepts the corresponding `VLLM_PPU_*`
name for compatibility. The SAIL name takes precedence whenever it is set,
including `0` or an empty string. An empty boolean is false; an empty backend
selection uses its default. Boolean truthy values are `true`, `1`, `yes` and
`on`, case-insensitively. `is_set()` recognizes either name, and `snapshot()`
reports effective values under SAIL names. `VLLM_DEEPEPLL_RECV_HOOK` keeps its
shared upstream name.

## Plugin discovery

vLLM discovers the installed plugin through package entry points. If you use
vLLM's plugin allowlist, include `ppu`:

```bash
export VLLM_PLUGINS=ppu
```

The same name enables platform discovery and the general hook that installs
registrations and model integrations. Include any other plugins needed by your
application in the allowlist.

## Backend selection

| Variable | Default | Values / behavior |
| --- | --- | --- |
| `VLLM_SAIL_MOE_BACKEND` | Automatic | `deepgemm`, `acext`, `triton` |
| `VLLM_SAIL_DENSE_BACKEND` | Automatic | `deepgemm`, `acext`, `triton` |
| `VLLM_SAIL_DENSE_BF16_DEEPGEMM` | `0` | Opt into the PPU 1.5 BF16 dense DeepGEMM path |
| `VLLM_SAIL_DEEPGEMM_MOE_TP_FUSED` | `0` | Enable the supported fused tensor-parallel MoE path |
| `VLLM_SAIL_DEEPGEMM_MOE_FUSED_ACT_QUANT` | `0` | Fuse the activation and the second MoE GEMM's input quantization into DeepGEMM's SiLU/SiTU kernel for the FP8 and MXFP4 experts; the BF16 (w4a16 or unquantized) experts use its values-only kernel, which also replaces the standalone `vllm::silu_and_mul` / `vllm::situ_and_mul` that unquantized BF16 dense MLPs and shared experts run on |
| `VLLM_SAIL_USE_PLA` | `1` | Allow PLA linear-attention paths where available |
| `VLLM_GDN_DECODE_KERNEL` | `auto` on PPU | `auto`, `cuda`, `pla`, `triton`; selects GDN decode independently of prefill and KDA |
| `VLLM_SAIL_FUSED_RMSNORM_QUANT` | `0` | Enable supported fused RMSNorm and quantization paths |
| `VLLM_SAIL_USE_OPT_TOKEN_GROUP_QUANT` | `0` | Enable the optimized token-group quantization path |
| `VLLM_SAIL_USE_TRITON_INT8_QUANT` | `1` | Use Triton dynamic per-token INT8 quantization |
| `VLLM_DEEPEPLL_RECV_HOOK` | `1` | Use the deferred receive hook; `0` selects the event-based fallback |

For example, to request the MoE DeepGEMM backend:

```bash
VLLM_SAIL_MOE_BACKEND=deepgemm vllm serve /path/to/model
```

Leave backend variables unset for normal capability-based selection.
An explicit choice still requires the corresponding library, device support and
compatible tensor layout. Avoid setting overlapping backend controls while
diagnosing a selection issue.

INT8 dense `deepgemm` is used for BF16 output without bias. Layers with bias
use ACEXT when selected and available, otherwise Triton fuses bias in its GEMM.
Symmetric INT8 dense layers fall back to Triton when neither DeepGEMM nor ACEXT
is selected; `VLLM_SAIL_DENSE_BACKEND=triton` selects that path directly. Triton
input quantization is controlled separately by `VLLM_SAIL_USE_TRITON_INT8_QUANT`.
Asymmetric INT8 activation quantization still requires the unsupported CUTLASS
zero-point path.

On PPU, GDN decode `auto` prefers PLA when the legacy `VLLM_SAIL_USE_PLA`
switch permits it and the model uses supported head dimensions and FP32 SSM
state. Otherwise it tries the registered native fused kernel, then Triton.
An explicit `pla`, `cuda` or `triton` choice overrides that legacy switch for
GDN decode only. Prefill and KDA retain their existing selection.

PLA decode requires `--mamba-ssm-cache-dtype float32` and K=V=128; the resolver
does not change an existing cache dtype. Missing dependencies or unsupported
static configurations fail early for explicit choices. Native `cuda` requires
a real CUDA dispatch implementation, not just an operation schema or an
unsupported-op stub. Its fused MTP kernel applies only to eligible batches;
other batches use the existing Triton path. The new HGGC port still requires
native build and device correctness verification before deployment.

## Step5 configuration lifetime

All `VLLM_SAIL_STEP5_*` settings and `VLLM_STEP5_INDEXER_FP8_AS_FP16` use the
same lazy access as other model settings: each `envs` attribute read resolves
and parses the current process environment, including aliases. The general
plugin startup hook does not freeze these values. `is_set()` checks the current
environment, and `snapshot()` parses the current values on each call.

Set configuration before starting vLLM to keep workers consistent. Updating
`os.environ` affects subsequent reads in that process, not other processes,
already-created model/cache state or captured CUDA graphs. Restart the server
and workers when changing configuration that affects those objects.

## Step5 Prefill Attention

`VLLM_SAIL_STEP5_DSA_PREFILL_BACKEND` defaults to `flashinfer`. Selecting
`native` enables the PPU native path for supported inputs. If the backend
variable is unset, the compatibility switch
`VLLM_SAIL_STEP5_DSA_PREFILL_PPU=1` also selects `native`.

`VLLM_SAIL_STEP5_DSA_PREFILL_BATCH_ATTENTION` defaults to `1` (`True`) and
only affects native prefill: indexer selection still consumes chunks, while attention
runs once after all chunks have supplied their selections. Set it to `0` to
run attention per chunk. It does not select the native backend, affect decode,
or change FlashInfer or non-PPU execution. Unsupported native inputs still
fall back to FlashInfer before launch.

## Step5 joint preparation

`VLLM_SAIL_STEP5_JOINT_PREPARE=1` enables experimental eight-warps launch
configuration for the unchanged joint indexer/main-attention preparation kernel.
It defaults to `0`; the `VLLM_PPU_STEP5_JOINT_PREPARE` alias remains supported,
with the SAIL name taking precedence. The switch is read lazily when selecting
the launch configuration; set it before startup and CUDA graph capture.
Changing it does not alter an already-captured graph.

Only PPU SM89, 32/64 tokens, TP4 local main QH16/KH1 with dimension 192 and
RoPE64, indexer H4 padded to H8 with dimension 256 and RoPE32, BF16 activations,
FP32 main norm weights, FP8 query/cache storage and no fused CSA update qualify.
Other shapes, dtypes, layouts or launch options retain the original configuration.
Set the switch to `0` to restore the original four-warps launch. This does not
change the direct scorer switches, kernel math, output allocation or cache layout.

The measured benefit is a single-launch warm-cache graph-event diagnostic on
configuration-derived shapes, not captured serving inputs. ACU counters,
cold-cache performance, whole-model accuracy and HTTP throughput are not accepted
by that comparison. The switch remains opt-in.

## Step5 Score

`VLLM_SAIL_STEP5_DSA_SCORES` selects the **prefill** scorer:

| Mode | Behavior |
| --- | --- |
| `v0` (default) | Keep the upstream call chain. |
| `v1` | Resident-block scoring; attempt one launch per eligible layer forward. |
| `v2` | Prefer whole-layer compact FP8 scoring, otherwise whole-layer resident; ordinary H4 chunks keep their existing path. |
| `v3` | Force group/chunk scoring with compact FP8 and grouped ordinary H4 enabled; never consolidate whole-layer scores. |

Step5 variables are read lazily and retain their `VLLM_PPU_*` aliases;
SAIL takes precedence. Prefill and decode modes are independent.

One launch covers all actual queries and all local KV groups on one device,
not one launch per group or chunk. Original chunk metadata, selection, causal
lengths, paired-head mapping, and attention prepare/finish remain unchanged.
Summary K is packed once after the original summary update and shared between
all groups within that forward. Q, packed K, and scores are never cached across
forwards or layers; only validated step-local metadata may be reused.

`VLLM_SAIL_STEP5_DSA_SCORE_MAX_BYTES` is a lazily read nonnegative integer, default
`1073741824` (1 GiB), with `VLLM_PPU_STEP5_DSA_SCORE_MAX_BYTES` as its alias.
The exact FP32 score-output demand is `actual_tokens * local_groups * regions * 4`;
equality with the budget is allowed. This limits scores only, not total temporary
or peak memory. It applies to `v1/v2`; set `0` for same-scorer chunked A/B.
`v3` always uses chunks regardless of this budget, so no extra variable is needed.
Exceeding the budget rejects consolidation before new score metadata/output
allocation or launch; all groups retain the original 128 MiB chunk behavior,
with a deduplicated fallback log.

`v3` replaces the former `v2 + SCORE_MAX_BYTES=0 + SCORE_GROUPED=1`
combination. Ordinary H4 prefill chunks reuse the grouped scorer with up to
16 queries per group. H8, paired H4 and H16 retain their existing chunk
contracts and fallbacks; selector, attention backend and tolerances are unchanged.

For isolated A/B, compare `v2 + SCORE_MAX_BYTES=0` against `v3`. The grouped
candidate still requires SM89, FP8 queries, FP32 weights, dimension 256,
regions > 512, at least 256 chunk queries and a full request-local group.
Unsupported contracts fall back before launch.

```bash
.venv/bin/python tools/step5_score_harness.py --mode layer-diagnostic-bench \
  --grouped-ab --large --layer-case 1 --layer-case 5 \
  --attention-backend flashinfer --groups 6 --iterations 8
```

`--grouped-ab` is diagnostic only. It adds a `v3-grouped` arm to the original
five arms, sharing the input and chunk behavior of `v2-chunked`. Historical
`v2-grouped` reports remain readable. Fixed FP32 oracle failures are retained;
these reports are not production accuracy acceptance.

Whole-layer resident scoring retains the existing PPU hardware E4M3 eligibility
(device capability >= 8.9), with FP8 queries, uint8 shared packed K, FP32 weights,
dimension 256, valid H4/H8/H16 views and complete runtime contracts. Compact is
restricted to SM89 and further requires effective H4/H8, regions > 512 and at least
256 queries containing a full request-local query group. H16 does not enable
compact and uses whole-layer resident when eligible. Tails stay in the selected
whole-layer scorer, with original masking and paired lengths, instead of
silently launching another scorer.

`VLLM_SAIL_STEP5_DSA_SCORE_DECODE` selects only decode/MTP score computation,
with alias `VLLM_PPU_STEP5_DSA_SCORE_DECODE`:

| Mode | Former configuration | Behavior |
| --- | --- | --- |
| `v0` (default) | `SCORE_DIRECT=0` | Original direct call chain. |
| `v1` | `SCORE_DIRECT=1`, `SCORE_DIRECT_MTP=0` | Experimental query/weights-resident scorer. |
| `v2` | `SCORE_DIRECT=1`, `SCORE_DIRECT_MTP=1` | Experimental guarded MTP scorer. |

Both candidates retain the four FP32 FMA chains and addition order. Admission
requires PPU SM89, H8, FP8 queries, uint8 summaries, FP32 weights, dimension 256
and the original 128-region tile. FP16 indexers and unsupported contracts fall
back before launch. Decode mode does not change prefill, selector, attention,
output allocation or metadata, and is independent of prefill's score budget.

`v1` includes the measured  launch configuration (`workers=3`, `num_warps=4`,
`num_stages=1`, `HOIST=True`, `SPLIT=True`) only for the recorded 64-token,
query_len=4, groups=1 layout and its exact shape/stride/capacity contract.
Other admitted layouts retain the existing resident launch parameters.

`v2` does not apply that launch tuning. It additionally requires query_len=4,
groups=1, 8 or 16 local requests and score capacity >= 8192 regions. Within the
guarded kernel, K is shared only when actual sequence length is 65536–69632
and all four slots of the request are active. Other requests use original direct
score computation, not the `v1` resident candidate. Device-side fallback still
has branch overhead. Local requests are not a measured HTTP batch-size mapping.

```bash
export VLLM_SAIL_STEP5_DSA_SCORES=v3       # chunked prefill with grouped H4
export VLLM_SAIL_STEP5_DSA_SCORE_DECODE=v1 # resident decode; v2 selects guarded MTP
```

The old `SCORE_GROUPED`, `SCORE_DIRECT` and `SCORE_DIRECT_MTP` variables, including
their `VLLM_PPU_*` spellings, are removed and no longer select a path. Migrate
existing launch scripts to the mode variables. Configure modes before process
startup and graph capture. Subsequent environment changes affect later `envs`
reads in that process, but do not change already-captured graphs.

Matching the old implementation does not establish FP32 oracle acceptance.
The candidates share known failures at `atol=rtol=1e-4`; they remain explicit
experiments, without accepted whole-model accuracy or HTTP throughput results.

Contexts that need no scoring still launch zero score kernels. Inputs eligible for resident continue to use resident. Increasing
the budget does not override kernel contracts: consolidation falls back if the
retained resident kernel cannot represent the whole-layer output row offset.
Compact uses int64 group/row offsets and is not subject to that resident limit.
Real JIT or device errors propagate directly instead of silently recomputing.
The selector and attention backends are selected independently; explicit
DeepSelect still requires the FP32-capable SAIL DeepSelect library. The grouped
scorer is an experimental optimization that is off by default; device
correctness and performance must be accepted separately, and neither implies
whole-model accuracy or HTTP throughput has passed.

Standalone device acceptance uses the existing `tools/step5_score_harness.py`:
it compares `reference/resident/compact` by default, `--route` selects a route
explicitly, and the legacy `mqa2/4/8` run only when requested explicitly.
`--selector per_row|deep_select` exercises the real selector; a missing
explicit dependency is reported as blocked rather than faking a fallback pass.
`--mode kernel` covers low-level scales, tail groups, K updates, and graph
replay; `--mode tune` performs an offline configuration sweep and never
autotunes in the production hot path. `--large` includes the 64K context;
`--mode ab` reports score-only, the pack/select/metadata pipeline, and the full
DSA forward separately, with the original resident as the primary baseline.
`--mode layer-numerical` validates real `Step5DSAAttentionImpl.forward` with
shared summary K, all group/chunk slices, unchanged `atol=rtol=1e-4` score
oracles, TopK, attention outputs, and summary/ring state. `--mode layer-ab`
compares budget `0` versus 1 GiB within each production scorer in A/B and B/A
orders, separately timing preallocated score-only kernels, pack/normalize/select with and
without new score metadata, and the full DSA forward. Independent profiling
checks each actual forward's device kernel hits, host submissions and slice
consumptions; two forwards must each recompute, including after changing K.
`--large` extends this matrix to multiple groups, H16 and paired boundaries.
If the fixed CPU oracle fails, `--mode diagnostic --case layer_forward` runs
only real outer-forward launch diagnostics with numerical failure markers;
it never enables formal A/B timing or reports production acceptance.
These single-layer results do not establish whole-model accuracy or HTTP throughput.
Acceptance is refused while other compute processes occupy the device; user
services are never stopped. Run Python commands inside the prepared SAIL
container.

`--group-rows/--tile/--grid-cap/--warps` override the default configuration in
the low-level numerical mode and bound the sweep in tuning mode; A/B rejects
custom configurations and must measure the actual production defaults. In the
formal numerical and A/B modes, both the CPU FP32 oracle and the resident
comparison must pass the fixed `atol=rtol=1e-4`; failures never proceed to
performance timing or recommended dispatch. Every measured attention backend
in the full forward must satisfy the fallback gate, and candidate routes must
have real submissions.

`--mode diagnostic-bench --large --groups 6 --iterations 32` is a separate FP8
kernel performance investigation: it first checks equivalence with resident
and repeat determinism, then runs A/B and B/A warm-cache graph replays on
preallocated outputs. Each side sweeps configurations offline, and both default
and per-side tuned results are reported. Inputs are fixed-seed synthetic data;
the 64K context is measured in 4096-query chunks, and each side's FP32 output
stays within 128 MiB. This mode excludes packing, selector, metadata, and
attention, and does not represent cold-cache HBM performance. Fixed CPU oracle
metrics are retained even when failing; `numerical_valid` and
`performance_valid` stay false, and no production dispatch recommendation is
produced.

To restore both original score paths:

```bash
export VLLM_SAIL_STEP5_DSA_SCORES=v0
export VLLM_SAIL_STEP5_DSA_SCORE_DECODE=v0
```


## DeepSeek V4.1

The main integration retains upstream V4.1 target/DSpark execution, shared KV
and index metadata, and adds PPU attention, delayed mHC, channelwise dense/Engram
quantization and checkpoint-name mappings. These source adaptations still need
device, model-load and inference verification; see the
[verification guide](verification_guide.md).

### MegaMoE

On Zhenwu M890, select MegaMoE with expert parallelism:

```bash
vllm serve /path/to/deepseek-v41 \
  --enable-expert-parallel --moe-backend deep_gemm_mega_moe
```

The mode selects the activation precision for MXFP4 routed expert weights.
All three modes return BF16 outputs. Install a
[DeepGEMM-for-sail](https://github.com/t-head/DeepGEMM-for-sail) build exposing
the selected kernel and its weight transform, where applicable.

| Mode | `VLLM_SAIL_MEGA_MOE_MMA_TYPE` | Activation | Expert dimensions |
| --- | --- | --- | --- |
| W4A4 | `fp4xfp4` | SwiGLU, SiTU | Hidden ≥ 2048 and divisible by 512. Intermediate divisible by 64. |
| W4A8 (default) | `fp8xfp4` | SwiGLU, SiTU | Hidden divisible by 256. Intermediate divisible by 128. |
| W4A16 | `bf16xfp4` | SiTU | Hidden divisible by 256. Intermediate divisible by 128. |

DeepSeek V4/V4.1 uses SwiGLU and can select W4A4 or W4A8. Kimi K3's native
latent MoE uses SiTU and requires an eligible MXFP4 checkpoint. The platform
gate checks hardware and DeepGEMM APIs without a model-name allow-list and
accepts EPLB configuration; the expert implementation checks the actual geometry,
activation and EPLB compatibility. Models without `moe_intermediate_size` defer
the shape check to expert construction. W4A8 still requires EPLB to be disabled
because rebalancing can change the inverse weight scale captured by the kernel.
W4A16 refreshes its transformed weight banks each step when EPLB is active.
Restart the server to replace checkpoint weights
with W4A8 or W4A16, whose transformed banks are bound by CUDA graphs. Shared
experts run as a separate MLP.

Upstream DeepSeek uses GateLinear with fused top-k for small local batches.
Larger local batches require a DeepGEMM build exposing `bf16_mega_gate`.

`VLLM_SAIL_MOE_BACKEND=deepgemm` continues to select the ordinary PPU DeepGEMM
expert backend. Use the CLI selection above for MegaMoE.

Indexer precision is independent of the expert mode. On M890, DeepSeek V4
and V4.1 default to MXFP4 indexer Q/K with BF16 scoring weights. GLM-5 and
GLM-5-Next retain their FP8 indexers with FP32 scoring weights; GLM-5-Next
uses the legacy FP8 DeepGEMM API and pads its query heads to at least 32.
On Zhenwu 810E, the GLM indexer uses INT8. Selecting MXFP4 indexer cache
explicitly is supported only for DeepSeek V4/V4.1 on M890. These format
rules do not enable MegaMoE for GLM's ordinary FusedMoE path.

The integration retains upstream's `SparseAttnIndexer` for dense
scoring, including candidate masking. V4/V4.1 query producers use the selected
operator's `weights_dtype` contract. The separate candidate-only
`SparseMQAIndexer` path can be enabled for DeepSeek V4.1 on M890 with
`--attention-config '{"indexer_sparse_logits":true,"indexer_kv_dtype":"mxfp4"}'`.
It requires the SAIL DeepGEMM sparse APIs, the `deep_select==1.0.0` wheel,
`VLLM_SAIL_USE_DEEPSELECT=1` (default), and top-k backend `auto` or
`deep_select`. The candidate-source layer still uses dense scoring; later
indexers score only its candidate blocks and return request-local token indices.

This path requires 32 indexer heads, head dimension 128, candidate blocks of
8/16/32 positions, and a power-of-two candidate count yielding 4–4096 sparse
blocks divisible by four. Top-k must fit the candidate width and be at most
4096; upstream DeepSelect row-alignment constraints still apply. Context
parallelism and non-MXFP4 indexer caches are rejected. Missing or disabled
dependencies fail at startup instead of silently selecting dense scoring.
Keep the option disabled to use the dense indexer. PPU inference correctness
and performance for the candidate-only path still require device validation.

The SAIL DeepSelect entry accepts BF16 and FP32 scores and passes them to the
external `deep_select` library in their original dtype, never downcasting FP32
to BF16. When the entry conditions are met, Step5 selects the library for both
`auto` and `deep_select`. FP32 execution requires upgrading to the FP32-capable
PPU DeepSelect library; the current BF16-only `deep_select==1.0.0` raises a
dtype error directly and no longer falls back automatically on FP32 input. Use
`--sparse-indexer-topk-backend per_row` until the library is upgraded.

Prefill chunking retains upstream's chunker and BF16 budget conversion for the
sparse backend. The PPU sparse builder includes SAIL's physical row padding in
that budget; the dense path keeps the conservative upstream FP32 budget.

## Compatibility controls

For FP8/BF16 conversion and recipe selection, see [Online quantization](online_quantization.md).
`VLLM_SAIL_FP8_LOAD_POLICY=native|bf16|int8` applies only when no explicit
quantization argument was supplied. Leaving it unset preserves native behavior;
the guide describes legacy opt-in variable precedence and source-format limits.

| Variable | Default | Behavior |
| --- | --- | --- |
| `VLLM_SAIL_DISABLE_MOE_WNA16_CUDA` | `0` | Disable the native WNA16 MoE path; takes precedence over `VLLM_SAIL_FORCE_MOE_WNA16_CUDA` |
| `VLLM_SAIL_FORCE_MOE_WNA16_CUDA` | `0` | Request the native WNA16 path; raises if the kernel is unavailable |
| `VLLM_SAIL_ENABLE_MOE_MARLIN` | `0` | Legacy selection gate; does not enable excluded native Marlin kernels |
| `VLLM_SAIL_FUSED_GDN_DECODE` | `1` | Legacy getter; dispatch now uses `VLLM_SAIL_USE_PLA` |

## Profiling

| Variable | Default | Behavior |
| --- | --- | --- |
| `VLLM_SAIL_NVTX_PROFILE` | `0` | Enable optional NVTX instrumentation |
| `VLLM_SAIL_NVTX_DUMP_TOPK` | `0` | Include top-k details in NVTX labels |
| `VLLM_SAIL_NVTX_VFA_DUMP_SEQLEN` | `0` | Include sequence lengths in FlashAttention NVTX labels |

`SAIL_NVTX_PROFILE` is also accepted, after `VLLM_PPU_NVTX_PROFILE` in precedence.
The profiling integration requires the optional `nvtx` package; install it with
`python -m pip install --no-build-isolation -e '.[profiling]'` from the repository
root. The SAIL SDK's optional `model_prof` package adds iteration hooks.


The same opt-in switch adds versioned `[FW_SPARSE] op:score|topk|attention`
ranges in the plugin's common dispatch and actual launch wrappers. Existing
whole-forward labels remain available. No new runtime switch is required.
An `op` range includes the operator's device preparation; a nested `kernel`
range identifies an observed launch. `backend:dispatch` is not execution
proof: use the actual leaf backend, not `requested_backend:auto/native`.
Tensor fields contain shape, dtype, stride and storage offset only; optional
inputs are `None`. No tensor values or pointers are read. `capture:1` records
host graph capture, not the number of subsequent graph replays.

Export one complete label per TXT line (an arbitrary text prefix is accepted):

```text
[FW_SPARSE] op:topk,v:1,fw:vllm,model:step5,pd_type:decode,scope:op,impl:select_regions,backend:dispatch,capture:0,scores:(2, 64),scores_dtype:torch.float32,scores_stride:(64, 1),scores_offset:0,current:(2,),current_dtype:torch.int64,current_stride:(1,),current_offset:0,topk:4,sort:1
```

The standalone UT does not start a model/server and is not part of the default
`tests/ut` performance gate. Extraction and dry-run require only the standard
library; malformed candidate labels report the source file and line. Sources,
SHA256, line numbers and observation counts survive deduplication. Old forward
labels cannot reconstruct these operator contracts.

```bash
python nvx_ut/benchmark_step5_sparse_nvtx.py extract \
  --nvtx trace-a.txt trace-b.txt --output cases.json
python nvx_ut/benchmark_step5_sparse_nvtx.py compare \
  --nvtx trace-a.txt --op topk --workload-config workload.json \
  --dry-run --output-dir sparse-dry-run
VLLM_SAIL_NVTX_PROFILE=0 VLLM_PPU_NVTX_PROFILE=0 SAIL_NVTX_PROFILE=0 \
python nvx_ut/benchmark_step5_sparse_nvtx.py compare \
  --nvtx trace-a.txt --op topk --backends per_row deep_select selector_v2 \
  --workload-config workload.json --scope prepared-op \
  --timer graph-events --output-dir sparse-comparison
```

For per-case GPU UTs, copy `nvx_ut/` to an environment with matching vLLM 0.33.x
and vLLM SAIL, then run its self-contained entry point:

```bash
python nvx_ut/run_step5_dsa_nvtx_ut.py \
  --nvtx trace-a.txt trace-b.txt --op topk --backends deep_select
```

This entry point requires `--op` and `--backends`, discovers installed native
extensions automatically, and checks vLLM's 0.33 series. `--output` defaults to
`./output`; an existing root is preserved and each run uses one start-time
subdirectory with flat per-case files. Only GPU ops are executed and timed;
CPU references only validate correctness outside timing. `--dry-only` performs
metadata preflight. See [Verification](verification_guide.md#step5-sparse-operator-replay)
for portability requirements.

The lower-level `compare` output directory must be new or empty unless a unique
`--output-prefix` is provided. Choose case IDs with repeatable `--case-id`;
choose operators separately or with `--op all`. A minimal explicit choice of
synthetic strategy is:

```json
{"default": {"policy": "capacity-causal", "scale": 0.1}, "cases": {}}
```

Shape does not encode content. `capacity-causal` explicitly chooses capacity-
bounded lengths, page mappings, positions, active rows and fixed selected
regions; these values are saved as synthetic workload data. For `policy:explicit`,
provide the content-dependent fields reported by dry-run in each case mapping:
`current` for topk; `seq_lens`, `slots`, `block_table` for direct score;
`lengths`, `block_table` for packed score; `lengths`, `owners`, `spans` for grouped
score; `seq_lens`, `positions`, `active`, `block_table`, `indices` and prefill
`owners` for attention. Quantized KV also needs explicit positive `k_scale` and
`v_scale`. Packed score uses explicit `summary_scale` (default synthetic 1.0).
Actual tensor contents are never recovered from a trace. Named comparison arms
and their actual leaf backend are recorded separately: the original paged
baseline on PPU may execute the existing `software_paged` compatibility kernel.
This is reported explicitly, not presented as an upstream CUDA kernel launch.
Unsafe/unsupported layouts are rejected instead of silently made contiguous. Some raw kernel labels
lack the complete public contract and are therefore not replayable on their own.

Defaults are seed 0, 5 warmups, 12 alternating-order samples, 20 graph iterations
and a conservative 2048 MiB budget. CUDA defaults to FlashInfer CUPTI with graph
and cold L2; unavailable timing fails. PPU uses graph replay/device events with
**warm cache**, not CUPTI/cold-cache acceptance. Disable NVTX and other profiling
in a fresh timing process. Host planning, compilation, allocation, oracle, capture
and H2D are outside timing; necessary device staging remains in `prepared-op`.
`--scope kernel` rejects arms without identical raw input/output boundaries.
Score's independent FP32 tolerance is fixed at `atol=rtol=1e-4`; attention uses
`0.025` on the same fixed selections. Topk checks exact complete stable semantics
or the explicitly reported raw cutoff-tie contract. Eager and graph correctness
are checked before timing, with another check afterward. Numerical failures still
report device timing but retain `correctness_failed` and a failed exit code;
`timing.valid_for_comparison` is false and no throughput or speedup is reported.
Execution/timer failures do not fabricate timing. Failed/unsupported arms never
substitute another backend.
Raw JSON samples and CSV are retained; range/median above 5% is unstable and does
not yield a stable speedup. Comparisons describe existing configurations, not
tuned backend optima, online workload frequency or end-to-end serving latency.

## Version metadata

Commit-pinned installations must report the full source SHA recorded in the
packaged kernel manifest. Leave `VLLM_VERSION` and `VLLM_VERSION_OVERRIDE` unset
so the plugin can verify that identity.

For release-tag metadata, `VLLM_VERSION` can state the actual release base when a
source build reports an ambiguous version. Use `VLLM_VERSION=0.34.0` only for a
checkout verified against that release. Commit-pinned installations still
require their full source SHA.

## Inspect the active environment

```bash
python -c "from vllm_sail import collect_env; collect_env()"
```

The report includes versions, device information, SDK package availability and
the PPU environment-variable snapshot. Package presence is a discovery check;
successful loading and execution require the
[verification steps](verification_guide.md).

`PPU_SDK` is also read at runtime and defaults to `/usr/local/PPU_SDK`. When
importing ACEXT fails on an unresolved cuBLAS symbol, the plugin preloads
`CUDA_SDK/lib64/libcublas.so` from that root and retries.

Build-time settings such as `PPU_SDK`, `HGCC`, `CMAKE_HGCC_DIR` and
`PYTORCH_SAIL_ARCH` are documented in [HGGC kernels](../developer_guide/kernels.md#build-settings).
