#!/usr/bin/env bash

# Example Custom Profile for llama.cpp
# llama.cpp is great for running models purely on CPU or low-end GPUs.

# The engine to use
ENGINE="llamacpp"

# The Hugging Face repository containing the GGUF file
MODEL="unsloth/Qwen2.5-0.5B-Instruct-GGUF"

# The exact filename within the repository to download and serve
# (llama.cpp requires explicit pointer to the gguf file)
MODEL_FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf"

# The port to expose on the host
PORT="8002"

# Extra arguments for llama.cpp, e.g. context window size
LLAMACPP_ARGS="-c 4096"

# Dedicated folder for llama configs cache
export LLAMACPP_MODELS_DIR="$HOME/.cache/modeltainer_llamacpp"
