#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== RAE Smoke Test ==="
echo "Generating 4 images with 25 steps..."

# Run generation with smoke test parameters
"${SCRIPT_DIR}/generate.sh" \
    --num-samples 4 \
    --num-steps 25 \
    --class-ids 207,388,971,985 \
    "$@"

echo ""
echo "=== Smoke test complete ==="
echo "Check outputs/ for generated images"
