# ModelTainer

ModelTainer delivers one‑command deployment for large language models on CPUs or GPUs. It exposes an OpenAI‑compatible API that can serve vLLM and llama.cpp models side‑by‑side, allowing you to hot‑swap models and compare results with minimal effort.

## Features

- **Unified API** – Interact with GPU (vLLM) and CPU/ARM (llama.cpp) models through the same endpoint.
- **Hot‑swappable models** – Change models via configuration or by switching Docker/Apptainer containers; no rebuilds required.
- **Portable** – Runs on a laptop or scales out to multi‑node clusters using Docker Compose.
- **Streaming responses** – Tokens stream as they are generated.
- **Apptainer support** – Package models into transferable Apptainer images.
- **NVIDIA Docker support** – Build GPU-enabled images with a user-defined
  NVIDIA Docker base version.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) Engine \>= 24 with Compose V2
- Git
- (Optional) [Hugging Face token](https://huggingface.co/settings/tokens) for gated models

## Quickstart

1. Clone the repository and enter it:
   ```bash
   git clone https://github.com/sirirajgenomics/modeltainer
   cd modeltainer
   ```
2. Start example LLM backends so the gateway has targets to proxy. The commands below launch a GPU vLLM service and a CPU llama.cpp service:
   ```bash
   # To start backends, use the included profile scripts. 
   # These profiles define how to run the official Docker images (vllm, sglang, llamacpp).
   # The scripts will automatically cache downloaded models locally.
   
   # Start a GPU vLLM service
   bash scripts/run_profile.sh profiles/example-vllm.sh
   
   # Start a CPU llama.cpp service
   bash scripts/run_profile.sh profiles/example-llamacpp.sh
   
   # Note: An example SGLang profile is also available in profiles/example-sglang.sh
   ```
   The scripts will spin up Docker containers natively and expose `http://localhost:8000` for vLLM and `http://localhost:8002` for llama.cpp by default, matching `config/models.yaml`.
3. Launch the API gateway:
   ```bash
   docker compose up -d gateway
   ```
   The gateway configures endpoints before composing the services.
4. Verify the stack with a chat completion request:
   ```bash
   curl -N -X POST http://localhost:8080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model": "gpt-oss-20b-it", "messages": [{"role": "user", "content": "Hello"}]}'
   ```
   A streaming response confirms everything is running.

### Configuration and Cleanup

- Ensure your LLM containers serve the models referenced in `config/models.yaml`.
- The `make up` command prints the configured models so you can verify endpoints before startup.
- Stop the gateway with `make down` and remove backend containers with `docker compose -f <file> down`.

## Documentation

See the [documentation index](docs/README.md) for guides on quickstart, security, model swapping, troubleshooting, and more.

## License

ModelTainer is licensed under the [Apache 2.0 License](LICENSE).

