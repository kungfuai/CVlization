#!/usr/bin/env bash
set -euo pipefail

# Smoke test for Qwen3-VL variants (default 2B)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VARIANT="${1:-2b}"

echo "Running Qwen3-VL smoke test (variant: $VARIANT)..."

QWEN3_VL_VARIANT="$VARIANT" \
bash "$SCRIPT_DIR/predict.sh" \
  --image test_images/sample.jpg \
  --task caption \
  --output "outputs/test_caption_${VARIANT}.txt"

echo "Smoke test completed successfully!"
