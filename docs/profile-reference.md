# Profile Reference

Complete reference for every field in a ModelTainer `.yaml` profile manifest.

A profile is a single YAML file that tells `run_profile.sh` everything it needs to launch one inference engine backend.

---

## Quick Example

```yaml
engine: vllm
hardware: gpu
model: Qwen/Qwen2.5-7B-Instruct
port: 8000

model_params:
  max_model_len: 8192
  gpu_memory_utilization: 0.90
  trust_remote_code: true

engine_args:
  - "--enable-prefix-caching"
```

```bash
bash scripts/run_profile.sh profiles/my-model.yaml
```

---

## Top-Level Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `engine` | string | ✅ | — | Inference engine: `vllm`, `sglang`, or `llamacpp` |
| `hardware` | string | | `gpu` | Hardware tier: `cpu`, `gpu`, or `dgx_spark` |
| `model` | string | ✅ | — | Hugging Face repo ID (e.g. `Qwen/Qwen2.5-7B-Instruct`) |
| `model_file` | string | ✅ for llamacpp | — | GGUF filename within the HF repo (llamacpp only) |
| `port` | integer | | `8000` | Host port to expose. Must be 1–65535. |
| `model_params` | object | | (all null) | Structured model tunables — see below |
| `engine_args` | list[string] | | `[]` | Raw args appended verbatim to the engine command |
| `host_cache_dir` | string | | `~/.cache/modeltainer` | Host directory for HF model cache |
| `container_name` | string | | `modeltainer-{engine}-{port}` | Override the Docker container name |
| `hf_token_env` | string | | — | Env var name holding your HF token, e.g. `HF_TOKEN` |

---

## Hardware Tiers

### `cpu`
- No GPU attached. Zero `--gpus` flags.
- **vLLM**: uses `vllm/vllm-openai:latest-cpu` image with `--device cpu`.
- **SGLang**: uses `lmsysorg/sglang:latest-cpu`.
- **llama.cpp**: same image as GPU, CPU threads only (no `-ngl`).

### `gpu`
- Standard NVIDIA or AMD GPU node. Adds `--gpus all` to `docker run`.
- All three engines are fully supported.

### `dgx_spark`
- NVIDIA DGX Spark (GB10 Grace Blackwell, ARM aarch64, 128 GB unified memory).
- Adds `--gpus all`.
- **vLLM**: automatically injects `--enforce-eager` to avoid CUDA graph issues on early Blackwell drivers.
- Set `gpu_memory_utilization` higher than usual (e.g. `0.92`) to leverage the large unified memory pool.

---

## `model_params` Fields

All fields are optional. Unknown fields are rejected (typos fail fast).

### Shared (vLLM + SGLang)

| Field | Type | Engine | Equivalent CLI Flag | Description |
|-------|------|--------|---------------------|-------------|
| `max_model_len` | integer ≥ 1 | vLLM, SGLang | `--max-model-len` | Maximum context window / sequence length |
| `kv_cache_dtype` | string | vLLM, SGLang | `--kv-cache-dtype` | KV cache precision: `auto`, `fp8`, `fp8_e5m2` |
| `gpu_memory_utilization` | float 0.0–1.0 | vLLM, SGLang | `--gpu-memory-utilization` | Fraction of GPU VRAM for KV cache |
| `quantization` | string | vLLM, SGLang | `--quantization` | `awq`, `gptq`, `fp8`, etc. |
| `dtype` | string | vLLM, SGLang | `--dtype` | Weights dtype: `auto`, `half`, `float16`, `bfloat16`, `float32` |
| `tensor_parallel_size` | integer ≥ 1 | vLLM, SGLang | `--tensor-parallel-size` | GPUs for tensor parallelism |
| `trust_remote_code` | boolean | vLLM, SGLang | `--trust-remote-code` | Allow custom model code from HF |
| `served_model_name` | string | vLLM, SGLang | `--served-model-name` | Name the API advertises for this model |

### vLLM Only

| Field | Type | CLI Flag | Description |
|-------|------|----------|-------------|
| `pipeline_parallel_size` | integer ≥ 1 | `--pipeline-parallel-size` | Number of pipeline stages |

### llama.cpp Only

| Field | Type | CLI Flag | Description |
|-------|------|----------|-------------|
| `max_model_len` | integer ≥ 1 | `-c` | Context window size |
| `n_threads` | integer ≥ 1 | `--threads` | CPU threads to use |
| `n_parallel` | integer ≥ 1 | `--parallel` | Parallel request slots (concurrent users) |
| `n_gpu_layers` | integer ≥ 0 | `--n-gpu-layers` | Layers to offload to GPU (`-1` = all) |

---

## `engine_args` — Raw Passthrough

Append any raw engine CLI flags not covered by `model_params`. These are injected **after** all structured flags, so they can override defaults.

```yaml
engine_args:
  - "--enable-prefix-caching"
  - "--speculative-model"
  - "Qwen/Qwen2.5-0.5B-Instruct"
  - "--num-speculative-tokens"
  - "5"
```

> [!WARNING]
> Arguments in `engine_args` are passed verbatim with no validation. Conflicting or duplicate flags may cause engine startup errors.

---

## HF Token for Gated Models

Set `hf_token_env` to the name of the environment variable that holds your Hugging Face token:

```yaml
hf_token_env: HF_TOKEN
```

Then export the token before running:

```bash
export HF_TOKEN=hf_xxx...
bash scripts/run_profile.sh profiles/my-gated-model.yaml
```

The runner injects the token as `HUGGING_FACE_HUB_TOKEN` and `HF_TOKEN` inside the container automatically.

---

## Validation

Validate a profile without launching anything:

```bash
bash scripts/validate_profile.sh profiles/my-model.yaml

# Or validate all profiles at once:
bash scripts/validate_profile.sh profiles/*.yaml
```

## Dry Run

Print the exact `docker run` command that would be executed:

```bash
bash scripts/run_profile.sh --dry-run profiles/my-model.yaml
```
