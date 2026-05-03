#!/usr/bin/env bash

# =============================================================================
# ModelTainer — run_profile.sh
# =============================================================================
# Launch an official Docker inference engine from a declarative YAML profile.
#
# Usage:
#   bash scripts/run_profile.sh <profile.yaml>         # run
#   bash scripts/run_profile.sh --dry-run <profile.yaml>  # print docker command, don't run
#
# Profile format: see profiles/example-*.yaml and docs/profile-reference.md
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo "[modeltainer] $*"; }
err()  { echo "[modeltainer] ERROR: $*" >&2; }
die()  { err "$@"; exit 1; }

usage() {
  cat <<EOF
Usage: $0 [--dry-run] <profile.yaml>

  <profile.yaml>   Path to a ModelTainer YAML profile manifest.
  --dry-run        Print the resolved docker run command without executing it.

Examples:
  bash scripts/run_profile.sh profiles/example-vllm-gpu.yaml
  bash scripts/run_profile.sh --dry-run profiles/example-llamacpp-cpu.yaml
EOF
  exit 1
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
DRY_RUN=false
PROFILE_SCRIPT=""

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --help|-h) usage ;;
    -*) die "Unknown flag: $arg" ;;
    *)
      if [ -n "$PROFILE_SCRIPT" ]; then
        die "Only one profile can be specified at a time."
      fi
      PROFILE_SCRIPT="$arg"
      ;;
  esac
done

[ -z "$PROFILE_SCRIPT" ] && usage
[ ! -f "$PROFILE_SCRIPT" ] && die "Profile not found: '$PROFILE_SCRIPT'"

# ---------------------------------------------------------------------------
# Locate Python — prefer venv/system python3
# ---------------------------------------------------------------------------
PYTHON="${MODELTAINER_PYTHON:-$(command -v python3 2>/dev/null || command -v python || echo "")}"
[ -z "$PYTHON" ] && die "Python 3 is required but was not found. Install Python 3 or set MODELTAINER_PYTHON."
"$PYTHON" -c "import yaml, pydantic" 2>/dev/null \
  || die "Python dependencies missing. Run: pip install pyyaml pydantic"

# ---------------------------------------------------------------------------
# Validate and resolve the profile to a JSON spec
# ---------------------------------------------------------------------------
SCHEMA_PY="$(dirname "$(realpath "$0")")/profile_schema.py"
[ ! -f "$SCHEMA_PY" ] && die "profile_schema.py not found alongside run_profile.sh ($SCHEMA_PY)"

log "Validating profile: $PROFILE_SCRIPT"

SPEC_JSON=$("$PYTHON" "$SCHEMA_PY" --json "$PROFILE_SCRIPT") \
  || die "Profile validation failed. Fix the errors above and retry."

# ---------------------------------------------------------------------------
# Parse spec fields from JSON using Python (no jq dependency)
# ---------------------------------------------------------------------------
_jq() {
  "$PYTHON" -c "import json,sys; d=json.load(sys.stdin); print(d$1)" <<< "$SPEC_JSON"
}

CONTAINER_NAME=$(_jq "['container_name']")
IMAGE=$(_jq "['image']")
ENGINE=$(_jq "['engine']")
HARDWARE=$(_jq "['hardware']")
MODEL=$(_jq "['model']")
PORT=$(_jq "['port']")
CACHE_DIR=$(_jq "['cache_dir']")

# Build docker run flags from the spec
GPU_FLAGS=$("$PYTHON" -c "
import json, sys
d = json.loads(sys.stdin.read())
print(' '.join(d['gpu_flags']))
" <<< "$SPEC_JSON")

VOLUME_FLAGS=$("$PYTHON" -c "
import json, sys
d = json.loads(sys.stdin.read())
print(' '.join(f'-v {v}' for v in d['volumes']))
" <<< "$SPEC_JSON")

ENV_FLAGS=$("$PYTHON" -c "
import json, sys
d = json.loads(sys.stdin.read())
print(' '.join(f'-e {k}={v}' for k,v in d['env'].items()))
" <<< "$SPEC_JSON")

ENGINE_CMD=$("$PYTHON" -c "
import json, sys, shlex
d = json.loads(sys.stdin.read())
print(' '.join(shlex.quote(a) for a in d['engine_cmd']))
" <<< "$SPEC_JSON")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "========================================="
log "  Engine   : $ENGINE"
log "  Hardware : $HARDWARE"
log "  Model    : $MODEL"
log "  Port     : $PORT"
log "  Image    : $IMAGE"
log "  Cache    : $CACHE_DIR"
log "========================================="

# ---------------------------------------------------------------------------
# Special pre-step for llama.cpp: ensure GGUF file is downloaded
# ---------------------------------------------------------------------------
if [ "$ENGINE" = "llamacpp" ]; then
  MODEL_FILE=$(_jq "['model_file']")
  LLAMACPP_MODELS_DIR=$(_jq "['llamacpp_models_dir']")
  mkdir -p "$LLAMACPP_MODELS_DIR"

  if [ ! -f "$LLAMACPP_MODELS_DIR/$MODEL_FILE" ]; then
    log "Downloading '$MODEL_FILE' from '$MODEL'..."
    if $DRY_RUN; then
      log "[dry-run] Would download: $MODEL / $MODEL_FILE → $LLAMACPP_MODELS_DIR"
    else
      docker run --rm \
        -v "$LLAMACPP_MODELS_DIR:/models" \
        python:3.12-slim \
        bash -c "pip install -q huggingface_hub && \
                 huggingface-cli download '$MODEL' '$MODEL_FILE' \
                   --local-dir /models --local-dir-use-symlinks False"
    fi
  else
    log "Model file already cached: $LLAMACPP_MODELS_DIR/$MODEL_FILE"
  fi
else
  mkdir -p "$CACHE_DIR"
fi

# ---------------------------------------------------------------------------
# Build the full docker run command
# ---------------------------------------------------------------------------
DOCKER_CMD="docker run -d --rm \
  --name $CONTAINER_NAME \
  -p $PORT:$PORT \
  $GPU_FLAGS \
  $VOLUME_FLAGS \
  $ENV_FLAGS \
  $IMAGE \
  $ENGINE_CMD"

# Clean up excess whitespace for readability
DOCKER_CMD=$(echo "$DOCKER_CMD" | tr -s ' ')

# ---------------------------------------------------------------------------
# Execute (or print for --dry-run)
# ---------------------------------------------------------------------------
if $DRY_RUN; then
  log "[dry-run] Docker command:"
  echo ""
  echo "  $DOCKER_CMD"
  echo ""
  log "[dry-run] No container was started."
else
  log "Starting container: $CONTAINER_NAME"
  eval "$DOCKER_CMD"
  log "========================================="
  log "Container started successfully!"
  log "  Logs : docker logs -f $CONTAINER_NAME"
  log "  Stop : docker stop $CONTAINER_NAME"
  log "========================================="
fi
