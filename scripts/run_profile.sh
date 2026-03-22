#!/usr/bin/env bash

# Run an official Docker container based on a custom model profile
# Usage: ./scripts/run_profile.sh profiles/my-profile.sh

set -euo pipefail

usage() {
  echo "Usage: $0 <profile_script>"
  echo "Example: $0 profiles/example-vllm.sh"
  exit 1
}

if [ "$#" -ne 1 ]; then
  usage
fi

PROFILE_SCRIPT="$1"

if [ ! -f "$PROFILE_SCRIPT" ]; then
  echo "Error: Profile script '$PROFILE_SCRIPT' not found."
  exit 1
fi

echo "Loading profile: $PROFILE_SCRIPT"
source "$PROFILE_SCRIPT"

# Validate required variables
if [ -z "${ENGINE:-}" ] || [ -z "${MODEL:-}" ] || [ -z "${PORT:-}" ]; then
  echo "Error: Profile must define ENGINE, MODEL, and PORT."
  exit 1
fi

# Define default cache directory
HOST_CACHE_DIR="${HOST_CACHE_DIR:-$HOME/.cache/modeltainer}"
echo "Using host cache dir: $HOST_CACHE_DIR"

# Also support llama.cpp specific model file directly in host models directory
LLAMACPP_MODELS_DIR="${LLAMACPP_MODELS_DIR:-$HOME/.cache/modeltainer_llamacpp}"

case "$ENGINE" in
  vllm)
    echo "Starting vLLM server with model: $MODEL on port $PORT"
    mkdir -p "$HOST_CACHE_DIR"
    
    # Official vLLM Image: vllm/vllm-openai:latest
    docker run --gpus all -d --rm \
      -p "$PORT:$PORT" \
      -v "$HOST_CACHE_DIR:/root/.cache/huggingface" \
      --name "modeltainer-vllm-$PORT" \
      vllm/vllm-openai:latest \
      --model "$MODEL" --port "$PORT" ${VLLM_ARGS:-}
    ;;
    
  sglang)
    echo "Starting sglang server with model: $MODEL on port $PORT"
    mkdir -p "$HOST_CACHE_DIR"
    
    # Official SGLang Image: lmsysorg/sglang:latest
    docker run --gpus all -d --rm \
      -p "$PORT:$PORT" \
      -v "$HOST_CACHE_DIR:/root/.cache/huggingface" \
      --name "modeltainer-sglang-$PORT" \
      lmsysorg/sglang:latest \
      python3 -m sglang.launch_server --model-path "$MODEL" --port "$PORT" --host 0.0.0.0 ${SGLANG_ARGS:-}
    ;;
    
  llamacpp)
    echo "Starting llama.cpp server with model: $MODEL on port $PORT"
    mkdir -p "$LLAMACPP_MODELS_DIR"
    
    if [ -z "${MODEL_FILE:-}" ]; then
      echo "Error: llama.cpp requires MODEL_FILE to be specified in the profile (e.g. model.gguf)"
      exit 1
    fi
    
    # Download using huggingface_hub on the host side first, or inside a side-car if host doesn't have it.
    echo "Ensuring model file exists in local cache..."
    if [ ! -f "$LLAMACPP_MODELS_DIR/$MODEL_FILE" ]; then
        echo "Downloading $MODEL_FILE from $MODEL..."
        docker run --rm \
          -v "$LLAMACPP_MODELS_DIR:/models" \
          python:3.10-slim \
          bash -c "pip install huggingface_hub && huggingface-cli download $MODEL $MODEL_FILE --local-dir /models --local-dir-use-symlinks False"
    else
        echo "Model $MODEL_FILE already cached locally."
    fi

    # Official LLaMA.cpp Engine Image: ghcr.io/ggerganov/llama.cpp:server
    # It requires the model file explicitly (-m)
    docker run -d --rm \
      -p "$PORT:$PORT" \
      -v "$LLAMACPP_MODELS_DIR:/models" \
      --name "modeltainer-llamacpp-$PORT" \
      ghcr.io/ggerganov/llama.cpp:server \
      -m "/models/$MODEL_FILE" --port "$PORT" --host 0.0.0.0 ${LLAMACPP_ARGS:-}
    ;;
    
  *)
    echo "Unknown ENGINE: $ENGINE (Supported: vllm, sglang, llamacpp)"
    exit 1
    ;;
esac

echo "========================================="
echo "Container start command initiated."
echo "Use 'docker logs -f modeltainer-$ENGINE-$PORT' to view logs."
echo "Use 'docker stop modeltainer-$ENGINE-$PORT' to stop the container."
echo "========================================="
