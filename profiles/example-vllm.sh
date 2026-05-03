#!/usr/bin/env bash

# Example Custom Profile for vLLM — M tier (primary) or S tier
# Set MODEL and PORT to match the tier you want to run.

# The engine to use (vllm, sglang, llamacpp)
ENGINE="vllm"

# ---------------------------------------------------------------------------
# Choose a tier:
#   S  (Small)  — google/gemma-4-E4B-it         PORT=8010
#   M  (Medium) — openai/gpt-oss-20b             PORT=8020
#   L  (Large)  — openai/gpt-oss-120b            PORT=8030
# ---------------------------------------------------------------------------

# M tier default
MODEL="openai/gpt-oss-20b"

# M tier port
PORT="8020"

# Any extra arguments to pass to vLLM
VLLM_ARGS="--max-model-len 4096 --gpu-memory-utilization 0.90 --dtype auto"

# Where to cache models on the host machine.
export HOST_CACHE_DIR="$HOME/.cache/modeltainer"
