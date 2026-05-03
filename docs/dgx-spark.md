# NVIDIA DGX Spark Guide

ModelTainer has first-class support for the **NVIDIA DGX Spark** (GB10 Grace Blackwell Superchip).

## Why DGX Spark is Different

| Property | Standard GPU Node | DGX Spark |
|----------|-------------------|-----------|
| Architecture | x86-64 + discrete GPU | ARM aarch64 (Cortex-X925/A725) |
| Memory | System RAM + VRAM (separate) | 128 GB LPDDR5x **unified** (CPU+GPU share) |
| Model capacity | Limited by VRAM | ~200B params (single unit) |
| AI Performance | Varies | 1 PFLOP FP4 (with sparsity) |
| Docker image arch | amd64 | aarch64 (multi-arch pulls automatically) |

The key advantage is **unified memory** — there's no VRAM boundary, so models load directly without expensive CPU↔GPU transfers.

---

## Prerequisites

- DGX Spark with NVIDIA DGX OS (Ubuntu-based) installed.
- Docker Engine ≥ 24 with Compose V2.
- NVIDIA Container Toolkit installed (ships with DGX OS).
- *(Optional)* Hugging Face token for gated models.

---

## Quickstart

### 1. Clone & Enter Repository

```bash
git clone https://github.com/sirirajgenomics/modeltainer
cd modeltainer
```

### 2. Launch a Backend

Use the `dgx_spark` hardware tier in your profile. ModelTainer handles the rest.

```bash
# 7B model — fast and practical for most use cases
bash scripts/run_profile.sh profiles/example-vllm-dgx-spark.yaml
```

### 3. Start the API Gateway

```bash
docker compose up -d gateway
```

### 4. Verify Inference

```bash
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "my-model", "messages": [{"role": "user", "content": "Hello!"}]}'
```

---

## DGX Spark Profile Recommendations

```yaml
engine: vllm
hardware: dgx_spark        # ← This is the key setting

model: Qwen/Qwen2.5-7B-Instruct
port: 8000

model_params:
  # Leverage large unified memory with a long context
  max_model_len: 32768
  # Use most of 128 GB unified pool for KV cache
  gpu_memory_utilization: 0.92
  # Blackwell natively handles bfloat16
  dtype: auto
  trust_remote_code: true

# --enforce-eager is injected automatically by ModelTainer
# for vLLM on dgx_spark to avoid CUDA graph issues
engine_args: []
```

### What ModelTainer Does Automatically on `dgx_spark`

1. **Selects the correct image tag** — pulls the multi-arch manifest; Docker resolves to `aarch64` automatically.
2. **Adds `--gpus all`** — DGX Spark has a Blackwell GPU.
3. **Injects `--enforce-eager`** for vLLM — avoids CUDA graph compilation issues on early Blackwell driver releases. Once upstream fixes this, you can remove it via `engine_args` if needed.

---

## Running Large Models (70B+)

The DGX Spark's 128 GB unified memory can comfortably hold a 70B model at `bfloat16` (~140 GB) if combined with quantisation:

```yaml
engine: vllm
hardware: dgx_spark
model: meta-llama/Llama-3.3-70B-Instruct
port: 8000

model_params:
  max_model_len: 8192
  gpu_memory_utilization: 0.90
  quantization: awq           # AWQ brings 70B into ~35 GB
  dtype: auto
  trust_remote_code: false
```

---

## Dual-Unit Scaling (Preview)

Two DGX Spark units linked via ConnectX-7 (200 Gb/s) can handle models up to ~405B parameters. Multi-node tensor parallelism support is on the ModelTainer roadmap. For now, run each unit independently and route between them via `config/models.yaml`.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `exec format error` | Wrong arch image pulled | DGX Spark is aarch64; ensure your Docker pulls the multi-arch manifest (default behavior). |
| OOM on 7B model | KV cache too large | Lower `gpu_memory_utilization` to `0.85`. |
| Slow CUDA graph compilation at startup | Blackwell CUDA graph issue | `--enforce-eager` is injected automatically; if not, add it to `engine_args`. |
| Model not found | Gated model, no HF token | Set `hf_token_env: HF_TOKEN` in the profile and `export HF_TOKEN=hf_xxx`. |
