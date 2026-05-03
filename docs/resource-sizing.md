# Resource Sizing Cheatsheet

Estimate hardware needs for common model configurations.

| Model | Backend | Hardware | Precision | Min VRAM / RAM | Notes |
|-------|---------|----------|-----------|----------------|-------|
| `Qwen/Qwen2.5-0.5B-Instruct` | vLLM | CPU | float32 | ~2 GB RAM | CPU-only, dev/test |
| `unsloth/Qwen2.5-0.5B-Instruct-GGUF` (Q4) | llama.cpp | CPU | Q4_K_M | ~1 GB RAM | Ideal for CPU-only machines |
| `NousResearch/Meta-Llama-3-8B-Instruct` | vLLM | GPU | fp16 | ~16 GB VRAM | Single GPU |
| `openai/gpt-oss-20b` | vLLM | GPU | mxfp4 | ~24 GB VRAM | Single GPU |
| `gpt-oss-20b-mxfp4.gguf` | llama.cpp | GPU (partial) | mxfp4 | ~10 GB RAM + partial VRAM | `-ngl` offloads layers |
| `Qwen/Qwen2.5-7B-Instruct` | vLLM | DGX Spark | bfloat16 | ~14 GB unified | 128 GB pool; long context feasible |
| `meta-llama/Llama-3.3-70B-Instruct` (AWQ) | vLLM | DGX Spark | AWQ ~4-bit | ~35 GB unified | AWQ brings 70B into DGX Spark budget |
| `meta-llama/Llama-3.3-70B-Instruct` | vLLM | DGX Spark | bfloat16 | ~140 GB unified | Max single-unit capacity |

Adjust `GPU_COUNT` to spread models across multiple GPUs if available.

## Notes
- VRAM estimates assume no other processes share the GPU.
- Higher precision (fp16, fp32) increases memory requirements.
- For custom models, consult their documentation for exact sizing.
