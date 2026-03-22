# 🤖 ModelTainer AI Agent Guidelines

Welcome, Agent! You are acting as an AI Engineer working on the **ModelTainer** repository. 
When generating code, modifying configurations, or redesigning the architecture, please strictly adhere to the following core engineering principles:

## 1. Architectural Philosophy
- **Component Isolation**: ModelTainer serves as a proxy/gateway. It separates the API routing layer (`compose.yaml` gateway) from the backend engine execution layer (`run_profile.sh`). Do not tightly couple them.
- **No Custom Images for Engines**: ModelTainer strictly uses **official Docker images** for state-of-the-art LLM backends (e.g., `vllm/vllm-openai`, `lmsysorg/sglang`, `ghcr.io/ggerganov/llama.cpp:server`). **Do not write `Dockerfile`s or build custom images for these engines.**
- **Delegated Execution via Custom Profiles**: The deployment logic for a specific model is abstracted into lightweight Bash scripts located in the `profiles/` directory (e.g., `profiles/example-vllm.sh`).
- **Volume Mounted Caching**: Containers must never download models directly into their ephemeral storage. The central `run_profile.sh` script maps a local host directory (e.g., `~/.cache/modeltainer`) into the container to persist Hugging Face models between container restarts. 

## 2. Coding Standards
- **Bash Scripts (`scripts/`)**: 
  - Always use `set -euo pipefail`.
  - Prefer explicit variable exports and validate required inputs prior to execution.
  - Implement comprehensive logging and error handling.
- **Python Code (`api/`, `tests/`)**: 
  - Adhere to **PEP8** standard conventions.
  - Heavily use Python Type Hints.
  - Assume an async-first context via `FastAPI`/`Starlette` for the API gateway.
- **Docker/Compose**: 
  - Keep `compose.yaml` configurations focused on the proxy/gateway and foundational services (Redis, etc).
  - Use `docker run` inside bash profiles to dynamically handle hardware attachment (`--gpus all`) and caching.

## 3. Extending Engines
When a user requests a new backend engine (e.g., `TensorRT-LLM`, `Ollama`, or `vLLM` forks):
1. **Find the target Official Docker Image**.
2. **Modify `scripts/run_profile.sh`**: Add a new `case "$ENGINE" in` block mapping the volume mounts and port forwarding appropriately.
3. **Create an Example Profile**: Add an `example-<engine>.sh` to the `profiles/` directory demonstrating basic use.

## 4. Agentic Workflows
- **Always Verify Model Instructions**: Before creating a running job or profile for a new model, you MUST always fetch and read the target model's Hugging Face `README.md` (Model Card). This ensures that you account for specific quantization details, prompt templates, context window constraints, or custom code requirements (`--trust-remote-code`) before attempting to start the engine.

By maintaining this structure, we ensure ModelTainer remains agile, performant, and incredibly easy for end-users to adopt.
