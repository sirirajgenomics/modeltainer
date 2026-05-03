# 🚀 ModelTainer

<p align="center">
  <em>A unified, high-performance API Gateway for Large Language Models.</em>
</p>

**ModelTainer** is an enterprise-ready, declarative deployment system that orchestrates state-of-the-art LLM inference engines — **vLLM**, **SGLang**, and **llama.cpp** — behind a single **OpenAI-compatible API endpoint**.

Each deployment is described by a simple **YAML profile file** that specifies the engine, hardware tier, model, and tuning parameters. The runner validates the profile and builds the exact `docker run` command — no Bash knowledge required.

---

## ✨ Key Capabilities

- **Unified OpenAI-Compatible Gateway**: One `/v1/chat/completions` endpoint regardless of backend.
- **Engine Agnostic**: [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [llama.cpp](https://github.com/ggerganov/llama.cpp) via their official Docker images.
- **Three Hardware Tiers**:
  - `cpu` — Pure CPU inference, no GPU needed.
  - `gpu` — Standard NVIDIA / AMD GPU node.
  - `dgx_spark` — NVIDIA DGX Spark (Grace Blackwell, 128 GB unified memory).
- **Declarative YAML Profiles**: Every model deployment lives in a single `.yaml` file. Validated before any container starts.
- **Structured Model Parameters**: `max_model_len`, `kv_cache_dtype`, `gpu_memory_utilization`, `quantization`, `tensor_parallel_size`, and more — all typed, all validated.
- **Raw Engine Passthrough**: `engine_args` list appends any flag the engine supports verbatim.
- **Zero-Copy Host Caching**: Model weights are cached to `~/.cache/modeltainer` on the host and reused across container restarts.
- **Dry-Run Mode**: Print the exact `docker run` command without launching anything.

---

## 🛠️ Prerequisites

- [Docker Engine](https://docs.docker.com/get-docker/) ≥ 24 with Compose V2.
- Python 3.8+ with `pyyaml` and `pydantic` (for profile validation).
- NVIDIA Container Toolkit (only if using GPU tiers).
- *(Optional)* [Hugging Face token](https://huggingface.co/settings/tokens) for gated models.

---

## ⚡ Quickstart

### 1. Clone & Enter Repository
```bash
git clone https://github.com/sirirajgenomics/modeltainer
cd modeltainer
```

### 2. Choose a Profile and Launch

#### CPU-Only Machine
```bash
# llama.cpp — best CPU inference engine
bash scripts/run_profile.sh profiles/example-llamacpp-cpu.yaml
```

#### CPU + GPU Machine
```bash
# vLLM — highest throughput on GPU
bash scripts/run_profile.sh profiles/example-vllm-gpu.yaml
```

#### NVIDIA DGX Spark
```bash
# vLLM with DGX Spark optimisations (--enforce-eager injected automatically)
bash scripts/run_profile.sh profiles/example-vllm-dgx-spark.yaml
```

### 3. Dry-Run (Preview docker command without launching)
```bash
bash scripts/run_profile.sh --dry-run profiles/example-vllm-gpu.yaml
```

### 4. Start the API Gateway
```bash
docker compose up -d gateway
```

### 5. Verify Inference
```bash
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Explain transformer attention."}]
  }'
```

---

## 📄 Creating Custom Profiles

Create a `.yaml` file anywhere in `profiles/`:

```yaml
# profiles/my-llama3-70b.yaml

engine: vllm
hardware: gpu
model: meta-llama/Llama-3.3-70B-Instruct
port: 8000

model_params:
  max_model_len: 8192
  gpu_memory_utilization: 0.90
  quantization: awq
  dtype: auto
  tensor_parallel_size: 2       # Spread across 2 GPUs
  trust_remote_code: false

# Pass any additional engine-specific flags
engine_args:
  - "--enable-prefix-caching"

# Point to the env var that holds your Hugging Face token
hf_token_env: HF_TOKEN
```

Launch it:
```bash
bash scripts/run_profile.sh profiles/my-llama3-70b.yaml
```

Validate only (no container):
```bash
bash scripts/validate_profile.sh profiles/my-llama3-70b.yaml
```

---

## 🖥️ Hardware Tiers

| Tier | `hardware:` value | GPU Flags | Use Case |
|------|-------------------|-----------|----------|
| CPU-only | `cpu` | None | Laptops, CI, development |
| CPU + GPU | `gpu` | `--gpus all` | Workstations, cloud VMs |
| DGX Spark | `dgx_spark` | `--gpus all` + Blackwell flags | DGX Spark (128 GB unified memory) |

---

## ⚙️ Profile Fields Summary

| Field | Description |
|-------|-------------|
| `engine` | `vllm` / `sglang` / `llamacpp` |
| `hardware` | `cpu` / `gpu` / `dgx_spark` |
| `model` | HF repo ID |
| `model_file` | GGUF filename (llamacpp only) |
| `port` | Host port |
| `model_params.max_model_len` | Context window length |
| `model_params.kv_cache_dtype` | KV cache precision |
| `model_params.gpu_memory_utilization` | VRAM fraction for KV cache |
| `model_params.quantization` | Quantisation method |
| `model_params.dtype` | Weights precision |
| `model_params.tensor_parallel_size` | Multi-GPU tensor parallelism |
| `model_params.n_threads` | CPU threads (llamacpp) |
| `model_params.n_gpu_layers` | GPU offload layers (llamacpp) |
| `model_params.trust_remote_code` | Allow HF custom code |
| `engine_args` | Raw engine CLI flags (list) |
| `hf_token_env` | Env var name for HF token |

See [docs/profile-reference.md](docs/profile-reference.md) for the full reference.

---

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| [Profile Reference](docs/profile-reference.md) | Every profile field documented |
| [DGX Spark Guide](docs/dgx-spark.md) | Blackwell-specific setup and tips |
| [vLLM Runbook](docs/vllm-runbook.md) | vLLM GPU backend details |
| [Security](docs/security.md) | Auth, TLS, container hardening |
| [Resource Sizing](docs/resource-sizing.md) | VRAM/RAM estimates |
| [Troubleshooting](docs/troubleshooting.md) | Common issues |
| [SLURM + HPC](docs/slurm-vllm.md) | Running on HPC clusters |
| [AGENT.md](AGENT.md) | Guidelines for AI agents working on this repo |

---

## ⚖️ License

ModelTainer is open-source software licensed under the [Apache 2.0 License](LICENSE).
