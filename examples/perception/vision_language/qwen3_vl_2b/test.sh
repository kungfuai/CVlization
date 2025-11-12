#!/usr/bin/env bash
set -euo pipefail

# Smoke test for Qwen3-VL-8B
echo "Running Qwen3-VL-8B smoke test..."

# Test caption task
bash "$(dirname "${BASH_SOURCE[0]}")/predict.sh" \
  --image test_images/sample.jpg \
  --task caption \
  --output outputs/test_caption.txt

echo "Smoke test completed successfully!"
