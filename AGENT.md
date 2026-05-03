# 🤖 ModelTainer AI Agent Guidelines

Welcome, Agent! You are working on the **ModelTainer** repository.
Strictly adhere to the following engineering principles.

---

## 1. Architectural Philosophy

- **Component Isolation**: ModelTainer separates the API routing layer (`compose.yaml` gateway) from the backend engine execution layer (`scripts/run_profile.sh`). Do not tightly couple them.
- **No Custom Images for Engines**: ModelTainer uses **official Docker images** for LLM backends (`vllm/vllm-openai`, `lmsysorg/sglang`, `ghcr.io/ggerganov/llama.cpp:server`). Never write `Dockerfile`s for these engines.
- **Declarative YAML Profiles**: Every model deployment is a `.yaml` file in `profiles/`. All fields are Pydantic-validated by `scripts/profile_schema.py` before any Docker command runs.
- **Volume-Mounted Caching**: Containers never download models into ephemeral storage. `run_profile.sh` maps a host directory (`~/.cache/modeltainer`) into the container.
- **Hardware Tiers**: The `hardware:` field (`cpu` / `gpu` / `dgx_spark`) controls GPU flags, image selection, and engine-specific injections. Never hard-code `--gpus all` outside of this logic.

---

## 2. Profile System

### YAML Schema
All profiles are validated by `scripts/profile_schema.py` via Pydantic v2.
- **Add new `model_params` fields** in `ModelParams` with a docstring and the equivalent CLI flag.
- **Add new hardware tiers** by extending the `HardwareTier` enum and updating `gpu_flags()`, `engine_image()`, and `build_engine_cmd()`.

### Running & Validating
```bash
# Validate only (no Docker)
bash scripts/validate_profile.sh profiles/my-model.yaml

# Dry-run (print docker command)
bash scripts/run_profile.sh --dry-run profiles/my-model.yaml

# Launch
bash scripts/run_profile.sh profiles/my-model.yaml
```

### Official Model Tiers
Always maintain these 4 tiers in `config/models.yaml` and testing:
- **XS (Extra Small):** Port `8000`. Target: CPU/llama.cpp (e.g. LFM 1.2B).
- **S (Small):** Port `8010`. Target: Single GPU/vLLM (e.g. Gemma 4B).
- **M (Medium):** Port `8020`. Target: Standard GPU/vLLM (e.g. OSS 20B).
  - *M-alt:* Port `8021`. Target: SGLang variant.
- **L (Large):** Port `8030`. Target: DGX Spark/vLLM (e.g. OSS 120B).

---

## 3. Coding Standards

### Bash (`scripts/`)
- Always use `set -euo pipefail`.
- Validate all inputs before executing.
- Use `log()` / `err()` / `die()` helpers.
- Never parse JSON with Bash string ops — use Python for JSON.

### Python (`api/`, `scripts/`, `tests/`)
- PEP 8 style with type hints everywhere.
- Pydantic v2 for all data models (`model_config = {"extra": "forbid"}`).
- Async-first via `FastAPI`/`httpx` for the API gateway.
- No `requests` — use `httpx` only.

### Docker/Compose
- `compose.yaml` is for the gateway only. Engine containers are managed by `run_profile.sh`.
- No deprecated `version:` key.
- Always add `healthcheck:` and `restart: unless-stopped` to gateway services.

---

## 4. Extending Engines

When a user requests a new engine (e.g. `TensorRT-LLM`, `Ollama`):

1. **Find the official Docker image**.
2. **Add the enum value** to `EngineType` in `scripts/profile_schema.py`.
3. **Add an `engine_image()` branch** returning the correct image per hardware tier.
4. **Add a `build_engine_cmd()` branch** mapping `model_params` fields to CLI flags.
5. **Update `run_profile.sh`** if any special pre-steps are needed (e.g. GGUF download).
6. **Create an example profile** in `profiles/example-<engine>-<tier>.yaml`.
7. **Add tests** in `tests/test_profile_schema.py`.

---

## 5. Hardware Tiers

| Tier | `hardware:` | GPU Flag | Notes |
|------|-------------|----------|-------|
| CPU-only | `cpu` | none | vLLM uses `--device cpu`; llamacpp uses CPU threads |
| GPU | `gpu` | `--gpus all` | Standard NVIDIA/AMD workstation or server |
| DGX Spark | `dgx_spark` | `--gpus all` | ARM aarch64 + 128 GB unified memory; vLLM gets `--enforce-eager` automatically |

### DGX Spark Notes
- The GB10 Grace Blackwell Superchip is ARM (aarch64). Official vLLM and SGLang images ship multi-arch manifests — Docker pulls the correct architecture automatically.
- `--enforce-eager` is injected for vLLM on `dgx_spark` to bypass CUDA graph compilation on early Blackwell drivers. Do not remove this without testing.
- Dual-unit (2× DGX Spark via ConnectX-7) for 405B models is **out of scope** for the current runtime; use `tensor_parallel_size` within a single unit.

---

## 6. Agentic Workflows

- **Always fetch the model card** before creating a profile for a new model. Read the Hugging Face `README.md` to check for quantisation requirements, prompt templates, context window limits, and `--trust-remote-code` needs.
- **Validate before committing**: run `bash scripts/validate_profile.sh profiles/*.yaml` and `pytest tests/` before any PR.
- **Never commit secrets** — `config/secrets.env` and `.env` are git-ignored.

---

By maintaining this structure, ModelTainer remains agile, validated, and easy for users on any hardware tier to adopt.
