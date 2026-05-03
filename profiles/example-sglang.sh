#!/usr/bin/env bash

# Example Custom Profile for SGLang — M-alt tier
# SGLang is optimized for throughput. Serves the same model as the M vLLM
# backend but on a dedicated port for A/B testing.

# The engine to use
ENGINE="sglang"

# M-alt tier — same model as M, different engine
MODEL="openai/gpt-oss-20b"

# M-alt tier port (separate from vLLM M tier on 8020)
PORT="8021"

# Any extra arguments to pass to SGLang
SGLANG_ARGS="--trust-remote-code"

# Where to cache models on the host machine.
export HOST_CACHE_DIR="$HOME/.cache/modeltainer"
