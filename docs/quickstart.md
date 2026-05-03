# Quickstart

Bring up ModelTainer and serve an LLM behind an OpenAI-compatible API in minutes.

## Prerequisites
- Docker Engine \>= 24 with Compose V2
- Internet access for the first model download
- (Optional) [Hugging Face token](https://huggingface.co/settings/tokens) for gated models

## Steps
1. Clone the repository and change into the directory:
   ```bash
   git clone https://github.com/sirirajgenomics/modeltainer.git
   cd modeltainer
   ```
2. Start example backends so the gateway has models to proxy. Run these in separate terminal tabs, or run them in the background.

   **XS Tier (llama.cpp on CPU):**
   ```bash
   bash scripts/run_profile.sh profiles/example-llamacpp-cpu.yaml
   ```

   **M Tier (vLLM on GPU):**
   ```bash
   bash scripts/run_profile.sh profiles/example-vllm-gpu.yaml
   ```

3. Bring up the gateway:
   ```bash
   make up
   ```
4. Verify each tier:
   ```bash
   # XS — LFM2.5-1.2B-Thinking
   curl -s -X POST http://localhost:8080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model": "lfm2-5-1b", "messages": [{"role": "user", "content": "Hello"}]}'

   # S — Gemma 4 E4B
   curl -s -X POST http://localhost:8080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model": "gemma-4-e4b", "messages": [{"role": "user", "content": "Hello"}]}'

   # M — gpt-oss-20b
   curl -s -X POST http://localhost:8080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model": "gpt-oss-20b", "messages": [{"role": "user", "content": "Hello"}]}'

   # L — gpt-oss-120b
   curl -s -X POST http://localhost:8080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model": "gpt-oss-120b", "messages": [{"role": "user", "content": "Hello"}]}'
   ```
   A streaming response confirms the stack is running.

For advanced options such as model tuning or custom hardware configurations, see the [Profile Reference](profile-reference.md).

## Notes
- The first run downloads models into `~/.cache/modeltainer` (or `~/.cache/modeltainer_llamacpp`), which can take several minutes.
- Run `make down` to stop the gateway. To stop a backend container, use `docker stop modeltainer-<engine>-<port>`.
- Default ports are 8080 for the gateway, and backends run on 8000 (XS), 8010 (S), 8020 (M), 8021 (M-alt), and 8030 (L).
