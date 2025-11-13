#!/usr/bin/env bash
set -euo pipefail

# Smoke test for Florence-2 (default base variant)
echo "Running Florence-2 smoke test (base variant)..."

# Test caption task
bash "$(dirname "${BASH_SOURCE[0]}")/predict.sh" \
  --variant base \
  --image test_images/sample.jpg \
  --task caption \
  --output outputs/test_caption.txt

echo "Smoke test completed successfully!"
