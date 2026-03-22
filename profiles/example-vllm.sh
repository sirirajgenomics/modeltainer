#!/usr/bin/env bash

# Example Custom Profile for vLLM
# You can customize these variables to run different models.

# The engine to use (vllm, sglang, llamacpp)
ENGINE="vllm"

# The Hugging Face model repository
MODEL="Qwen/Qwen2.5-0.5B-Instruct"

# The port to expose on the host
PORT="8000"

# Any extra arguments to pass to the engine
# e.g., limiting max completion length or selecting specific quantizations
VLLM_ARGS="--max-model-len 4096"

# Where to cache models on the host machine.
# By default, this uses ~/.cache/modeltainer if not specified.
export HOST_CACHE_DIR="$HOME/.cache/modeltainer"
