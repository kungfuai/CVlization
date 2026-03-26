#!/usr/bin/env bash
# Smoke test: render embedded sample and verify a PNG is produced
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-cvlization/lilypond:latest}"

OUTDIR="$(mktemp -d)"
trap 'rm -rf "$OUTDIR"' EXIT

docker run --rm \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${OUTDIR},dst=/mnt/cvl/out" \
  --env "PYTHONPATH=/cvlization_repo" \
  --env "PYTHONUNBUFFERED=1" \
  --env "CVL_OUTPUTS=/mnt/cvl/out" \
  "$IMG" python3 predict.py

# Check output
PNG_COUNT=$(find "$OUTDIR" -name "*.png" | wc -l)
if [ "$PNG_COUNT" -gt 0 ]; then
  echo "PASS: $PNG_COUNT PNG file(s) produced."
else
  echo "FAIL: no PNG output found in $OUTDIR"
  exit 1
fi
