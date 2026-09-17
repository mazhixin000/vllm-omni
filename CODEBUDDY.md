# CODEBUDDY.md This file provides guidance to CodeBuddy when working with code in this repository.

## Environment and installation

Run commands from the repository root unless noted otherwise.

### Create the recommended development environment

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```

The package supports Python 3.10–3.13, but project documentation recommends Python 3.12 to match CI closely.

### Install for development

```bash
uv pip install vllm==0.26.0 --torch-backend=auto
uv pip install -e ".[dev]"
```

Dependencies are selected by hardware in `setup.py` from `requirements/<device>.txt`. On non-CUDA systems, set `VLLM_OMNI_TARGET_DEVICE` to `npu`, `rocm`, `xpu`, `musa`, or `cpu` when auto-detection is unreliable; use `--no-build-isolation` when the build must detect an already installed platform-specific PyTorch.

### Build wheel and source distribution

```bash
bash scripts/build_wheel.sh --python python
```

This cleans previous build artifacts and runs `python -m build`. Use `--skip-clean` to preserve artifacts, or `--create-venv --run-quality` for a clean build that first installs development dependencies, runs all pre-commit hooks, and executes non-slow tests.

## Formatting and linting

### Install hooks

```bash
uv pip install pre-commit
pre-commit install
```

### Check staged files

```bash
pre-commit run
```

### Check the entire repository

```bash
pre-commit run --show-diff-on-failure --color=always --all-files
```

The hooks run YAML and whitespace checks, Ruff lint/format, typos, actionlint, test-marker validation, and pickle-import validation. Ruff uses a 120-character line limit. Follow Google Python/C++ style. Do not introduce `librosa` or the banned `torch.cuda.*` APIs listed in `pyproject.toml`; use vLLM multimodal helpers and `torch.accelerator.*` equivalents.

## Tests

Many integration tests require model weights and specific accelerators. Install test dependencies with `uv pip install -e ".[dev]"`; audio tests may also require `apt-get install -y espeak-ng jq`.

### Quick non-slow suite

```bash
python -m pytest tests/ -v -m "not slow"
```

This is the repository build script's standard quality-test command. Some tests do not run successfully on CPU-only systems.

### L1 CPU tests

```bash
cd tests && pytest -s -v -m "core_model and cpu"
```

### Run one test file or one test

```bash
python -m pytest -s -v tests/path/test_file.py
python -m pytest -s -v tests/path/test_file.py::TestClass::test_name
```

For level-aware model tests, add the declared level, for example `--run-level=core_model`, and select matching markers when needed.

### Run CI-aligned L2, L3, or nightly jobs

```bash
bash tools/run_ready_jobs.sh
bash tools/run_merge_jobs.sh
bash tools/nightly/run_nightly_jobs.sh
```

These scripts extract pytest jobs from Buildkite YAML and store logs/timing summaries under `logs/`. Use `--dry-run` to inspect commands, `--skip-simple` to omit L1-style jobs, and filters such as `--model-type diffusion`; nightly also supports `--test-type function`, `perf`, `acc`, `stability`, or `local`.

Test modules must use registered markers from `pyproject.toml`. Choose an execution level (`core_model`, `advanced_model`, or `full_model`), a model area (`omni`, `tts`, or `diffusion`) for model-centric tests, and accurate platform/resource markers. Prefer `tests/helpers/mark.py` helpers for hardware-aware tests. See `.claude/skills/vllm-omni-test/` for the full routing and naming rules.

## Documentation

```bash
uv pip install -e ".[docs]"
API_AUTONAV_EXCLUDE=vllm_omni mkdocs serve
```

The excluded API build starts quickly. Run `mkdocs serve` without the environment variable when generated API references are required.

## Architecture

### Public entrypoints and vLLM integration

vLLM-Omni extends vLLM instead of replacing it. `pyproject.toml` registers `vllm-omni` at `vllm_omni.entrypoints.cli.main:main` and a `vllm.general_plugins` hook that registers Omni model architectures in every vLLM worker subprocess. The CLI delegates ordinary invocations to vLLM; when `--omni` is present it installs Omni `serve` and benchmark commands. Online serving is the OpenAI-compatible FastAPI stack in `vllm_omni/entrypoints/openai/`; offline callers use synchronous `Omni` or asynchronous `AsyncOmni` in `vllm_omni/entrypoints/`.

### Configuration and heterogeneous pipelines

A model is represented as one or more stages rather than one monolithic engine. Deploy definitions live in `vllm_omni/deploy/*.yaml`; `vllm_omni/config/` and the stage configuration factory combine model/pipeline defaults, deploy YAML, platform overlays, and explicit CLI overrides. Stage metadata describes stage type, devices, output modality, sampling defaults, connectors, and engine/parallel settings. Some current deploy files use the `pipeline`/`stages`/`platforms` schema; do not copy a `stage_args` YAML from another branch without confirming that this checkout's parser supports it. Platform overlays can replace nested stage settings, so preserve complete required nested configs when editing them.

### Runtime and orchestration

`OmniBase` resolves configuration and constructs `AsyncOmniEngine`. The engine creates stage processes/clients and an `Orchestrator`; the main implementation is under `vllm_omni/engine/`. Requests become typed queue messages, are dispatched to the first stage, and are forwarded between stages until configured final outputs are produced. `StageEngineCoreClient` handles vLLM autoregressive stages, while diffusion stages use `StageDiffusionClient`/the diffusion process. `OmniConnector` implementations under `vllm_omni/distributed/` transfer tensors, multimodal outputs, and KV-related data between disaggregated stages. Request routing, stage replicas, asynchronous chunks, CFG companion requests, and prefill/decode disaggregation are coordinated here.

### Autoregressive and diffusion execution

Autoregressive model implementations, loaders, layers, and registries are primarily under `vllm_omni/model_executor/`, building on vLLM's scheduler, model runner, and KV-cache machinery. `OmniModelConfig` adds stage identity, architecture, worker/output types, connector configuration, and stage-specific Hugging Face/quantization handling.

The non-autoregressive stack is under `vllm_omni/diffusion/`. `diffusion_engine.py`, stage clients/processes, schedulers, executors, and workers drive DiT pipelines. `diffusion/models/` contains model-specific pipelines; adjacent packages provide attention backends, distributed tensor/sequence/VAE execution, cache acceleration, quantization, offloading, LoRA, output formatting, and lightweight profiling. Keep model-specific behavior in its model package and reusable acceleration mechanisms in the corresponding shared diffusion subsystem.

### Platforms, outputs, and observability

`vllm_omni/platforms/` contains CUDA/ROCm/NPU/XPU/MUSA-specific registration and overrides; hardware-specific kernels or execution behavior should remain isolated there where possible. `inputs/`, `request.py`, `outputs/`, and `data_entry_keys.py` define cross-stage request/output contracts. `metrics/` aggregates stage, transfer, modality, and end-to-end statistics; `profiler/` and `diffusion/profiler/` provide optional instrumentation. Changes to queue messages, serialization, or output contracts usually require coordinated updates across entrypoints, the orchestrator, stage clients, and tests.

### Test organization

Tests mirror production areas (`engine`, `entrypoints`, `diffusion`, `model_executor`, `platforms`, `metrics`, and so on). `tests/e2e/offline_inference/` and `tests/e2e/online_serving/` cover real model flows; `tests/dfx/` contains performance, accuracy, stability, and reliability suites; `tests/helpers/` centralizes runners and hardware marking. CI levels progress from CPU-friendly L1 logic tests through L2/L3 model integration to L4 nightly and L5 reliability work.

## Repository-specific contribution rules

PR titles use the prefixes documented in `docs/contributing/README.md`, including `[Bugfix]`, `[CI/Build]`, `[Doc]`, `[Model]`, `[Frontend]`, `[Kernel]`, `[Core]`, and `[Hardware][Vendor]`. Commits require a DCO `Signed-off-by` line. Before submitting a PR, follow `.claude/skills/precheck-pr/SKILL.md`; use quick mode for showstoppers and full mode for maintainer-level review. Specialized repository workflows also exist under `.claude/skills/` for diffusion models/performance, TTS models, quantization, testing, and NPU upgrades.
