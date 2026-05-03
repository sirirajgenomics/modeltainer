#!/usr/bin/env bash

# Example Custom Profile for llama.cpp — XS tier
# LiquidAI LFM2.5-1.2B-Thinking: great for CPU-only or low-VRAM GPU machines.

# The engine to use
ENGINE="llamacpp"

# XS tier — LiquidAI LFM2.5-1.2B-Thinking (GGUF)
MODEL="bartowski/LFM2.5-1.2B-Thinking-GGUF"

# The exact GGUF filename within the repository to download and serve
# (llama.cpp requires an explicit pointer to the gguf file)
MODEL_FILE="LFM2.5-1.2B-Thinking-Q4_K_M.gguf"

# XS tier port
PORT="8000"

# Extra arguments for llama.cpp (context window, threads)
LLAMACPP_ARGS="-c 8192 --threads 8 --parallel 2"

# Dedicated folder for llama.cpp model cache
export LLAMACPP_MODELS_DIR="$HOME/.cache/modeltainer_llamacpp"
