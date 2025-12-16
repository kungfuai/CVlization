#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="cvlization/miles_qwen3_grpo:latest"

echo "Running Miles GRPO smoke test..."

# Check if image exists
if ! docker image inspect "$IMG" &>/dev/null; then
    echo "Image not found. Building first..."
    "${SCRIPT_DIR}/build.sh"
fi

# Run with minimal config for quick test
echo "Testing dry-run mode (no GPU required)..."
docker run --rm \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
    "$IMG" python train.py --dry-run

echo ""
echo "Smoke test passed!"
echo ""
echo "To run actual training (requires GPU):"
echo "  ./train.sh"
