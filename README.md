<h1 align="center">vLLM SAIL</h1>

<p align="center">
  <strong>LLM inference and serving on T-Head Zhenwu AI accelerators</strong>
</p>

<p align="center">
  <a href="docs/README.md">Documentation</a> |
  <a href="docs/getting_started/installation.md">Installation</a> |
  <a href="docs/user_guide/supported_features.md">Supported Features</a> |
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

## Overview

vLLM SAIL is a hardware plugin for [vLLM](https://github.com/vllm-project/vllm).
It connects vLLM's inference engine to the SAIL software stack, bringing PPU
kernels, attention backends, quantization and model integration to the standard
vLLM Python API and OpenAI-compatible server.

**SAIL** is the software brand and **Zhenwu (真武)** is the hardware product
name. The T-Head AI chips are **Zhenwu 810E** and **Zhenwu M890**. The Python
distribution is `vllm-sail`, the import package is `vllm_sail`, and the vLLM
plugin name is `ppu`.

The plugin builds native kernels with **HGGC**, using the `hgcc` compiler and
[cmake-hgcc](https://github.com/t-head/cmake-hgcc). PPU libraries and
[Triton for SAIL](https://github.com/t-head/triton-for-sail) provide specialized
compute paths. vLLM discovers the installed plugin through its platform and
general plugin entry points.

## Features

- **Native PPU kernels:** activation, normalization, quantization, cache
  operations, sampling and MoE routing compiled with HGGC.
- **Attention:** PPU FlashAttention integration, MLA and sparse attention,
  plus GDN and KDA linear attention paths.
- **Dense and MoE compute:** DeepGEMM, ACEXT and Triton backends, with BF16/FP16
  and selected INT8, FP8, MXFP4 and INT4 quantization paths.
- **Model integration:** PPU model registrations and adaptations for dense,
  mixture-of-experts and hybrid architectures.
- **Developer tooling:** tests that run without a PPU, reproducible kernel
  generation for upstream sources, patch checks and device correctness tests.

Availability depends on the device, SDK, backend and checkpoint format. See the
[feature matrix](docs/user_guide/supported_features.md) for implementation scope
and constraints, and the [verification guide](docs/user_guide/verification_guide.md)
for validating a deployment.

## Requirements

| Component | Requirement |
| --- | --- |
| Hardware | Zhenwu 810E and Zhenwu M890 AI chips; build targets `ppu_10` and `ppu_15` |
| Operating system | Linux for PPU builds and inference |
| Python | 3.10–3.13; choose a version supported by your SDK packages |
| vLLM | `>=0.33.0,<0.34.0` release line, exact source pin, built with `VLLM_TARGET_DEVICE=empty` |
| SAIL software | SAIL SDK, SAIL PyTorch and matching PPU compute libraries |
| Build tools | `hgcc`, CMake, Ninja and a C++ compiler; CMake fetches pinned cmake-hgcc sources when needed |

Build the exact upstream commit recorded in
[the verified pin](.github/vllm-main-verified.commit) and
[kernel manifest](csrc/upstream/manifest.toml), using **vLLM v0.33.0** as the
release baseline. Both pin records must agree. The
[installation guide](docs/getting_started/installation.md) describes dependency
setup and native extension checks.

## Getting Started

Prepare the [SAIL environment](docs/getting_started/installation.md), then clone
and install the plugin:

```bash
git clone https://github.com/t-head/vllm-sail.git
cd vllm-sail
python -m pip install -r requirements/build.txt -r requirements/ppu.txt
python -m pip install --no-build-isolation --no-deps -e .
python -c "from vllm_sail import collect_env; collect_env()"
```

After the installation checks pass, start a server with a local model:

```bash
vllm serve /path/to/model \
  --served-model-name ppu-model \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --host 127.0.0.1 --port 8000
```

Follow the [quickstart](docs/getting_started/quickstart.md) for a client request,
offline inference and model configuration.

## Documentation

| Guide | Contents |
| --- | --- |
| [Installation](docs/getting_started/installation.md) | SDK prerequisites, source installation and troubleshooting |
| [Quickstart](docs/getting_started/quickstart.md) | Online serving and offline inference |
| [Supported features](docs/user_guide/supported_features.md) | Backends, quantization, model integration and limitations |
| [Configuration](docs/user_guide/configuration.md) | Backend selection and PPU environment variables |
| [Verification](docs/user_guide/verification_guide.md) | Environment, kernel and model validation |
| [Architecture](docs/developer_guide/architecture.md) | Plugin lifecycle and source layout |
| [HGGC kernels](docs/developer_guide/kernels.md) | Native builds, source ownership and regeneration |
| [Kernel benchmarks](benchmarks/README.md) | Device benchmarks and MoE tuning tools |

## Contributing

Contributions are welcome: report a reproducible issue, improve documentation,
add a test, tune a PPU workload or implement a kernel. Many development tasks
can be done without PPU hardware.

Start with [CONTRIBUTING.md](CONTRIBUTING.md) for the development environment,
test commands and pull request workflow. Use this repository's Issues to discuss
bugs and proposed features, and include the environment and validation details
requested by the templates.

## License

vLLM SAIL is licensed under the [Apache License 2.0](LICENSE). Third-party source
files retain their original copyright and license notices.
