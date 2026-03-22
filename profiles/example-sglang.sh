#!/usr/bin/env bash

# Example Custom Profile for sglang
# SGLang is optimized for throughput. It accepts similar arguments to vLLM.

# The engine to use
ENGINE="sglang"

# The Hugging Face model repository
MODEL="Qwen/Qwen2.5-0.5B-Instruct"

# The port to expose on the host
PORT="8001"

# Any extra arguments to pass to SGLang
SGLANG_ARGS="--trust-remote-code"

# Where to cache models on the host machine.
export HOST_CACHE_DIR="$HOME/.cache/modeltainer"
