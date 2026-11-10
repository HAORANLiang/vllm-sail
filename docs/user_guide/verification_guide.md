# Verification Guide

Validate the same package installation, SDK and hardware that will run the
workload. Keep the results of each check separate:

| Check | What it establishes |
| --- | --- |
| CPU unit tests | Python behavior, registration contracts and packaging invariants |
| Source regeneration checks | Generated upstream kernels match their declared inputs |
| HGGC build | Selected sources compile and link for the requested targets |
| Extension imports and strict preflight | Libraries load and representative operations are registered |
| Device correctness tests | Tested kernels match their numerical references on that device |
| Model run | The tested checkpoint and configuration complete inference |
| Benchmark | Performance under the recorded workload and environment |

Passing one check does not imply the later checks pass. A skipped test provides
no execution evidence for that feature.

## 1. Record the environment

After following [Installation](../getting_started/installation.md), run:

```bash
git rev-parse HEAD
python -c "from vllm_sail import collect_env; collect_env()"
python -c "import vllm_sail; print(vllm_sail.__file__)"
```

Record the PPU model and count, SDK version, SAIL PyTorch version, vLLM revision
and plugin revision. Confirm the printed package path points to the installation
being tested. A source checkout can shadow a separately installed native wheel.

Run the four-library import and strict preflight commands in
[Check the installation](../getting_started/installation.md#4-check-the-installation).
Resolve shared-library errors before running a model.

## 2. Run unit tests

From the plugin checkout, using its test environment:

```bash
python -m pytest tests/ut -q
```

The merge gate runs with only `requirements/dev.txt` installed, without torch,
vLLM or a device. Additional integration tests become available in a PPU
environment with their dependencies installed. Investigate failures and report
skip reasons; do not compare test counts from different environments as if they
cover the same paths.

Compatibility UT may intentionally import the plugin checkout beside installed
vLLM and torch. Record those import paths and set `VLLM_SOURCE_ROOT` to the same
exact upstream revision even when source-marked cases are deselected. This UT
mode does not verify the installed plugin wheel: separately compare its full
package contents with the wheel and run native imports, device tests and models
from that verified installation.

### Check upstream source compatibility

Tests live with the feature they cover, independently of the vLLM release that
introduced them. Tests marked `upstream_source` read a selected vLLM checkout
without importing vLLM, torch or the SDK:

```bash
VLLM_SOURCE_ROOT=/path/to/vllm python -m pytest tests/ut -q -rs \
  -m upstream_source --require-upstream-source
```

`VLLM_SOURCE_ROOT` names the checkout root containing `vllm/__init__.py`.
Without it, ordinary CPU runs skip source-dependent checks. The required-source
option makes a missing or invalid checkout an error, so CI cannot silently
skip these checks. The source tests cover selected upstream call sites, keyword
compatibility, quantization behavior and imported consumer inventories. They do
not establish that module imports, patch installation or device inference work.

### CI jobs

The [unit-test workflow](../../.github/workflows/ut.yaml) runs for pull requests,
pushes to `main` or `master`, and manual dispatch. Its jobs run independently:

| Job | Environment and checks | Failure handling |
| --- | --- | --- |
| CPU UT | Python 3.10–3.13; only `requirements/dev.txt`; UT and branch coverage | Test failure fails the workflow; coverage-upload failure does not |
| CI control contracts | Python 3.12; standalone stdlib tests of the Preview control tools, the existing operator-test definition check and shell syntax checks | Allocates no PPU; touches no Registry or release assets; any failed check fails the workflow |
| vLLM source (main-verified) | Python 3.12; checks out the exact verified main commit; runs `upstream_source` tests with a required source path | Checkout or test failure fails the workflow |

The source job reads the full SHA in
[verified main commit](../../.github/vllm-main-verified.commit) and logs the
resolved commit. Update this pin only after qualifying that exact revision.
Keep feature tests named after their behavior. The workflow does not install
vLLM or run PPU workloads; installed compatibility and device checks require
the prepared environment below. Repository branch protection determines which
job results are required for merging.

The separate [upstream canary](../../.github/workflows/upstream-canary.yaml) runs
weekly on Tuesday at 03:17 UTC and on manual dispatch. It installs upstream
`main` and checks patch drift. A drift failure opens or updates the tracking
issue and fails the job. Installation failure stops before that report step.
This scheduled check does not replace PR checks, and none of these public CPU
jobs runs PPU inference or `tests/e2e`.

### Preview images and minimum regression

Preview is a separate image-verification and distribution channel. It does not
replace the official `PPU CI`, model accuracy, performance, dual-chip operator
verification or `release.yaml`. It does not modify the
[verified main commit](../../.github/vllm-main-verified.commit), and it does not
promote a minimum-regression result into a full software-stack certification;
KMD and firmware stay marked `not_measured`.

| Workflow | Inputs and outputs |
| --- | --- |
| [Nightly regression (minimum)](../../.github/workflows/nightly-tests.yaml) | Daily at 00:17 UTC or manual dispatch on the default branch; downloads the successful `ppu-accuracy-nightly.yaml` artifacts of `vllm-project/vllm-sail` `main`, assembles and tests a candidate image |
| [Distribute nightly previews](../../.github/workflows/promote-nightly.yaml) | `workflow_run` of a successful nightly, or a manual `test_run_id` on the default branch; rechecks the passed bundle of the current attempt and distributes the original wheels and the identical image digest |
| [Dockerfile CI](../../.github/workflows/dockerfile-ci.yaml) | PRs run the hosted static checks automatically; a trusted maintainer requests an image build and a single-device smoke test from the default branch, reported separately as `dockerfile-ppu` |

The source must be a `schedule` or `workflow_dispatch` run in which the whole
workflow succeeded, not merely the wheel-build job. The source run must have been
created within 48 hours and its `ppu-wheels-<run_id>` artifact must be unexpired.
A missing qualifying artifact fails the run; there is no fallback to a source
build, a PR preview or a failed run. The accuracy lane is long, so the schedule
does not guarantee that the previous round finished; select a qualifying run
manually when needed, and never treat a missing or expired artifact as a success.

Three commit identities are recorded separately: the control SHA of the
default-branch control scripts, the SAIL SHA of the wheels and the upstream vLLM
SHA of the wheels. Tests check out the source at the wheel's SAIL SHA, and the
upstream vLLM SHA must match the reviewed pin of the control checkout. Testing an
old artifact does not mean the latest `main` is covered.

[Dockerfile.ci](../../docker/Dockerfile.ci) only installs the two original wheels
into the configured runtime image, with `--no-deps --force-reinstall` followed by
`pip check`; it downloads no model and compiles no vLLM. The test stage first
verifies the installed Python/native bytes and import paths against the original
wheels, then runs the 2 patch UTs and the 9 `silu_and_mul` numerical tests of the
[fixed catalog](../../.github/ci/tests/catalog.json) on one Zhenwu M890. The JUnit
and node reports must prove that all 11 items actually executed; a skip is not a
pass. Extending the catalog also requires changing the runner and the evidence
checks; adding JSON entries alone is not enough.

#### Administrator configuration and first enablement

No real image address or credential is provided by default. Until the enablement
variables are configured, the hosted preflight fails and no downstream
self-hosted job is allocated. Merge the control files into the protected default
branch through a PR first, then configure:

| Actions Variable | Requirement |
| --- | --- |
| `PPU_TRANSITION_ENABLED` | `true` once the runner owner approves using persistent hosts for this purpose |
| `PREVIEW_REGISTRY_HOST` | Registry host name, optionally with a port; no scheme or namespace |
| `PREVIEW_IMAGE_REPOSITORY` | A dedicated untagged Preview image path on that registry |
| `RUNTIME_BASE_IMAGE` | A readable `repository@sha256:<64 hex>`; must differ from the Preview output repository |
| `TRUSTED_PPU_TESTS_ENABLED` | `true` only after the first nightly/distribution verification and approval of trusted PR execution |

`RUNTIME_BASE_IMAGE` must already contain the matching SAIL SDK, SAIL PyTorch,
CUDA-compatible runtime dependencies, Triton, and the dependencies of the tested
source's `requirements/dev.txt` (including pytest). The existing runners'
`PPU_CI_RUNTIME_IMAGE` is not propagated to this variable. The workflows log in
only to `PREVIEW_REGISTRY_HOST`; base images from other private registries must be
arranged separately.

Create the following Environments and restrict them to the default branch;
declaring `environment:` does not install approval rules by itself:

| Environment | Secrets and protection |
| --- | --- |
| `nightly-registry` | `REGISTRY_USER`, `REGISTRY_TOKEN`: Preview push permission; `UPSTREAM_ACTIONS_TOKEN` may be configured for cross-repository reads |
| `nightly-test` | `REGISTRY_PULL_USER`, `REGISTRY_PULL_TOKEN`: read-only pull permission |
| `preview-distribution` | `REGISTRY_USER`, `REGISTRY_TOKEN`: Preview tagging/push permission only |
| `dockerfile-ci` | `REGISTRY_PULL_USER`, `REGISTRY_PULL_TOKEN`; configure required reviewers and prevent self-approval where supported |

Consuming this repository's artifacts from this repository usually only needs
`GITHUB_TOKEN`. A cross-repository token must be able to read the source Actions
artifacts and the local Preview Release. Registry write credentials stay in their
own Environment and are never passed to PR device jobs or test containers.
Protect the workflow files, the runner group and the default branch; Environment
approval is not a malicious-code isolation mechanism and must not be used to let
untrusted external PRs through.

Device runners must carry the label `[self-hosted, Linux, X64, 8-ppu-m890p]` and
satisfy:

- Linux x86-64, Bash, Git, Python 3.11+, Docker/runc, flock, coreutils.
- Connectivity to GitHub/API/Actions artifacts and the Registry, plus enough
  disk space.
- A working PPU driver and `/dev/alixpu`, `/dev/alixpu_ctl`, `/dev/alixpu_sep`,
  `/dev/alixpu_ppu0`, with device access for the runner user and the container
  user.
- Coordination with every job that shares the devices; the label alone is not an
  exclusive device reservation.

The new scripts reuse `/tmp/vllm-sail-ppu-ci.lock` and wait at most 3600 seconds;
the existing PR runtime still uses `flock -n` and fails when the lock is taken.
Before going live, arrange a dedicated runner or a conflict-free window; unifying
the lock policy requires adjusting every caller and timeout separately. This route
is not a Kubernetes isolation solution.

Verify the first enablement in this order:

1. Run the hosted control tests, then configure the base image, Registry,
   Environments and runners.
2. Approve and enable `PPU_TRANSITION_ENABLED`, then run the nightly manually; a
   qualifying source run ID may be supplied.
3. Confirm the candidate image, the 11 actually executed results and the passed
   bundle of the current attempt.
4. Confirm that the automatically distributed image digest and wheel SHA256 match
   the tested values.
5. Only then enable the trusted Dockerfile PR tests and watch the following
   scheduled runs.

A manual Dockerfile test requires an open, non-draft, same-repository PR targeting
the default branch; both the triggerer and the author must have write permission.
Tick `approve_host_execution` when requesting, and expect the head/base SHA to be
revalidated after approval; any change requires a new request. It tests the PR's
`docker/` directory together with existing Preview wheels, not the SAIL sources
modified by that PR. `dockerfile-static` success does not imply `dockerfile-ppu`
success and cannot replace the official `PPU CI`. If a cancellation leaves a
pending status, treat it as not passed and verify again manually. A
path-filtered check should not be made a required status for every PR.

#### Distribution, retries and retention

The publisher maintains only these Preview namespaces:

- Image tag: `preview-<upstream_run_id>-test-<local_test_run_id>`, without
  rebuilding the tested image.
- Immutable prerelease:
  `preview-nightly-<upstream_run_id>-test-<local_test_run_id>`.
- Fixed prerelease: `preview-baseline`, holding `preview-wheels.zip` and
  `preview-baseline.json` and preserving the original wheel versions and metadata.

The release push trigger and the `setuptools-scm` version derivation both exclude
`preview-*`, so a Preview run ID is never interpreted as a package version. The
minimum regression does not update the stable pin, the BOM, `ci-baseline` or an
official Release. Artifacts distributed within the last 48 hours may be retested
daily, but an identical or older source run never advances the Preview pointer;
retesting old wheels after changing only the Dockerfile does not advance it
either.

The two assets of the fixed Release are not updated atomically. The publisher
runs serially: it uploads the wheels and then writes the JSON marker; readers
refuse a missing asset or a hash mismatch. An interruption is recovered from the
verified immutable Preview assets, and the old assets are kept under an archived
name. Do not run another publisher against the same pointer concurrently. To roll
back, a maintainer pauses the publisher and selects the wheels and digest of a
historical immutable Preview.

Actions artifacts are retained for 14 days; nothing cleans the Registry, the
Releases or the persistent work directories automatically. Configure a separate
retention and disk-monitoring policy when going live.

Local control verification needs no torch, vLLM, device or network. In the project
root with Python 3.11+:

```bash
python3.12 -B -m unittest discover -s tests -p 'test_nightly_assets.py' -q
python3.12 -B -m unittest discover -s tests -p 'test_deployment.py' -q
python3.12 -B -m unittest discover -s tests -p 'test_verify_installed.py' -q
python3.12 -B -m unittest discover -s tests -p 'test_workflow_contracts.py' -q
python3.12 -B scripts/ci/deployment.py validate-test-source --source .
bash -n scripts/ci/run_existing_tests.sh
bash -n scripts/ci/ppu_smoke.sh
```

## 3. Run device correctness tests

In the PPU environment, install `requirements/dev.txt`. Use an editable native
plugin build in the checkout, or ensure the test process imports the installed
wheel as described above. Start with activation and the affected feature:

```bash
pytest tests/e2e/test_native_activation.py -q -rs
pytest tests/e2e/test_deepseek_v4_native.py -q -rs
pytest tests/e2e/test_mxfp4_prepare.py -q -rs
pytest tests/e2e/tier_b/test_norm_quant.py -q -rs
```

The [`tests/e2e`](../../tests/e2e) directory also contains linear-attention,
indexer, cache and model-specific kernel tests. Select the relevant tests for
the change; run the full suite when qualifying an environment:

```bash
pytest tests/e2e -q -rs
```

These tests require a real PPU and the applicable SDK libraries. Verify that the
intended tests actually ran. Include dtype, shape, tolerance and device in a
correctness report, and test both architecture targets when claiming both.

Capability markers restrict tests to one chip: `cap80` runs only on compute
capability 8.0 (Zhenwu 810E) and `cap89` only on 8.9 (Zhenwu M890); other
devices skip them. `perf` marks performance benchmarks and `model_e2e` marks
offline model end-to-end tests. Select or exclude them with `-m`, for example
`pytest tests/e2e -q -rs -m "not perf"`.

### GDN fused MTP decode

Run the native GDN MTP accuracy matrix with a native plugin build:

```bash
python -m pytest tests/e2e/gdn/test_gdn_fused_mtp.py -q -rs \
  -o junit_family=xunit1 --junitxml=gdn-mtp-accuracy.xml
```

The tests call the native fused op directly and force the unfused reference onto
Triton, bypassing PLA. Selected cases also compare with an independent eager
PyTorch reference. On PPU, a missing native dispatch registration fails the
suite rather than skipping it.

Coverage includes all supported head ratios, BF16/FP32 recurrent state and norm
weights, FP16/BF16/FP32 gate bias, SiLU/sigmoid output gates, MTP lengths 1–8,
strided rows and state slots, padding and NULL slots, and repeated accepted-token
rewind with changing request lengths. Every round checks outputs, candidate
state writes, unused state storage and output-buffer guards. The suite also
tests numerical boundaries and the layer's post-convolution entry point.

The native-versus-Triton output keeps upstream's relative-L2 limit of `5e-4`;
pointwise output/state comparisons retain the upstream model-path tolerances
`atol=rtol=3e-2`. Each written slot additionally has a relative-L2 limit of `3e-2`.
Untouched slots and buffer guards must remain bitwise identical. The JUnit file
records relative-L2 and maximum absolute errors per round. These limits are
source-derived, not calibrated from a PPU run; investigate failures before
changing them.

The independent oracle has worked-example CPU tests that run when torch is
available:

```bash
python -m pytest tests/ut/test_gdn_mtp_reference.py -q
```

CPU oracle tests and device-test collection do not establish native accuracy.
Record actual execution counts and skips from the PPU command above.

## 4. Run a model

### Quick serving sanity

The standalone sanity runner launches one installed model, checks readiness
against a unique served-model identity, sends a deterministic arithmetic request,
checks its complete final answer and stops its owned server process group:

```bash
python scripts/eval/ppu_model_sanity.py --list
python scripts/eval/ppu_model_sanity.py \
  --profile deepseek-v4-community-tp8 \
  --checkpoint-root /path/to/datasets \
  --output /absolute/new-sanity-attempt \
  --with-stream
```

Profiles cover community DeepSeek V4 Flash 0731 TP8, its MXFP4/FP8-channel TP8
and DP8/EP8 low-latency variants, GLM 5.2 MXFP4/FP8 TP8, and MiniMax M3
FP8-channel TP8. Checkpoint suffixes and serving arguments are recorded in
[`model_sanity_profiles.json`](../../scripts/eval/model_sanity_profiles.json).
The dataset root defaults to `/ppusw/datasets`; set it to the actual mounted
root. Each attempt requires a new output directory. Run profiles sequentially
on free devices in the selected environment, preserving existing work.

The default sends one ordinary request; `--with-stream` adds a streamed request.
Both must return the final answer `42` and finish naturally. Truncated output,
reasoning without a final answer, a wrong answer, HTTP failure, serving or
shutdown ERROR logs, and failed cleanup fail the attempt. The process exits
nonzero on failure and retains its invocation, installed package metadata,
responses, metrics, serving log and result. Adjust readiness, request and token
budgets explicitly with `--ready-timeout`, `--request-timeout`, and `--max-tokens`.
Each request budget covers connection setup, response headers and the complete
body, including a streamed response that never finishes a line. Expiration
fails the attempt and enters service cleanup; cleanup has its own bounded wait.

This is a short inference check for the selected configuration. It does not
replace the accuracy evaluation below, kernel correctness checks, package-byte
verification or evidence that every configured optimization was active.

### Model verification

Follow the [quickstart](../getting_started/quickstart.md) with a checkpoint
appropriate for the available memory and supported precision. Confirm that:

- The PPU platform and intended compute backend are selected.
- Model loading and warm-up complete.
- The readiness endpoint succeeds and generation returns sensible output.
- Repeated generation and the intended context lengths complete successfully.

For MoE, quantized or multi-device workloads, repeat the check using the intended
expert backend, checkpoint layout and parallelism. A small dense-model smoke
test does not cover those paths.

## 5. Report correctness and performance

Include the exact command, checkpoint identifier and revision, dtype,
quantization, context length, device count and parallelism. For benchmarks, also
include the workload or dataset, input/output lengths, concurrency, warm-up,
throughput and latency measurements.

Useful tools include vLLM's `vllm bench --help` and the repository's
[`benchmarks/kernels`](../../benchmarks/kernels). Compare baseline and candidate
with the same software, hardware and workload. Explain the numerical reference
and tolerance used for kernel comparisons.

Report which checks passed, failed, skipped or were not run. Attach concise
logs and a minimal reproducer through the repository's issue or pull request
template; remove credentials and private model or host details before posting.
