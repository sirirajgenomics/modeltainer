# 🚀 ModelTainer

<p align="center">
  <em>A unified, high-performance API Gateway for Large Language Models.</em>
</p>

**ModelTainer** is an enterprise-ready, declarative deployment system that orchestrates state-of-the-art LLM inference engines—including **vLLM**, **SGLang**, and **llama.cpp**—behind a single **OpenAI-compatible API endpoint**. 

By leveraging official Docker images and a flexible profile-based runner, ModelTainer eliminates the friction of managing complex container builds, manual weight caching, and multi-engine routing. You can hot-swap models, perform A/B testing on different engines, and serve GPU and CPU models side-by-side with minimal effort.

---

## ✨ Key Capabilities

- **Unified OpenAI-Compatible Gateway**: Interact with multiple disparate models (running on diverse hardware) through one seamless `/v1/chat/completions` REST endpoint.
- **Engine Agnostic**: Natively supports [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), and [llama.cpp](https://github.com/ggerganov/llama.cpp) utilizing their official, highly optimized Docker distributions.
- **Declarative Custom Profiles**: Spin up models dynamically using transparent `.sh` profiles rather than wrestling with monolithic Docker Compose files or manual build contexts. 
- **Zero-Copy Host Caching**: Model weights are seamlessly cached to your host machine (default: `~/.cache/modeltainer`), entirely bypassing redundant downloads and saving critical SSD space.
- **Seamless Scalability**: Runs efficiently on a single laptop edge-node or scales elegantly into multi-GPU server environments.

## 🛠️ Prerequisites

- [Docker Engine](https://docs.docker.com/get-docker/) (v24+ recommended) with Compose V2.
- NVIDIA Container Toolkit (if utilizing GPUs).
- Git.
- *(Optional)* [Hugging Face token](https://huggingface.co/settings/tokens) exported for gated models.

## ⚡ Quickstart

### 1. Clone & Enter Repository
```bash
git clone https://github.com/sirirajgenomics/modeltainer
cd modeltainer
```

### 2. Launch LLM Backends (Custom Profiles)
ModelTainer uses **Custom Profiles** located in the `profiles/` directory. These profiles abstract the complexity of launching isolated engines.

Let's spin up a GPU-accelerated **vLLM** backend and a CPU-bound **llama.cpp** backend:

```bash
# Start a GPU-accelerated instance via vLLM
bash scripts/run_profile.sh profiles/example-vllm.sh

# Start a CPU instance via llama.cpp
bash scripts/run_profile.sh profiles/example-llamacpp.sh
```

*(Note: We also include a high-throughput SGLang profile in `profiles/example-sglang.sh`!)*

### 3. Start the API Gateway
The gateway proxies requests to your newly spun up backends. Its routes are defined dynamically via `config/models.yaml`.

```bash
docker compose up -d gateway
```

### 4. Verify Inference 
Send a standard chat completion request to your unified endpoint:

```bash
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-oss-20b-it", 
    "messages": [{"role": "user", "content": "Explain the architecture of a transformer model."}]
  }'
```
You will receive a blazing-fast, streaming token response!

---

## ⚙️ Creating Custom Profiles

Creating your own model deployment is extremely straightforward. Create a new file in `profiles/my-model.sh`:

```bash
#!/usr/bin/env bash

ENGINE="sglang"               # vllm, sglang, or llamacpp
MODEL="Qwen/Qwen2.5-7B"       # Hugging Face Repo ID
PORT="8000"                   # Port to expose
SGLANG_ARGS="--trust-remote-code" # Engine specific arguments
```

Execute it instantly via: `bash scripts/run_profile.sh profiles/my-model.sh`.

## 📚 Documentation
- See the `docs/` folder for deeper guides on security, model routing, and complex deployment topologies.
- For AI Agents and Contributors, please read our [AGENT.md](AGENT.md) for architectural guidelines.

## ⚖️ License
ModelTainer is open-source software licensed under the [Apache 2.0 License](LICENSE).
