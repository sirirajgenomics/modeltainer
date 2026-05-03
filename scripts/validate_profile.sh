#!/usr/bin/env bash
# =============================================================================
# ModelTainer — validate_profile.sh
# =============================================================================
# Dry-validate one or more YAML profile manifests without launching anything.
#
# Usage:
#   bash scripts/validate_profile.sh profiles/my-model.yaml
#   bash scripts/validate_profile.sh profiles/*.yaml
# =============================================================================

set -euo pipefail

die() { echo "[modeltainer] ERROR: $*" >&2; exit 1; }

[ "$#" -lt 1 ] && { echo "Usage: $0 <profile.yaml> [profile2.yaml ...]"; exit 1; }

PYTHON="${MODELTAINER_PYTHON:-$(command -v python3 2>/dev/null || command -v python || echo "")}"
[ -z "$PYTHON" ] && die "Python 3 is required."

SCHEMA_PY="$(dirname "$(realpath "$0")")/profile_schema.py"
[ ! -f "$SCHEMA_PY" ] && die "profile_schema.py not found: $SCHEMA_PY"

PASS=0
FAIL=0

for profile in "$@"; do
  if "$PYTHON" "$SCHEMA_PY" "$profile"; then
    PASS=$((PASS + 1))
  else
    FAIL=$((FAIL + 1))
  fi
done

echo ""
echo "Results: $PASS passed, $FAIL failed."
[ "$FAIL" -gt 0 ] && exit 1 || exit 0
