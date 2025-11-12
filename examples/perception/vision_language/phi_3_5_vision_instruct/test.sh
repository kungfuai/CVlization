#!/usr/bin/env bash
set -euo pipefail

# Smoke test for Phi-3.5-vision-instruct
echo "Running Phi-3.5-vision-instruct smoke test..."

# Test caption task
bash "$(dirname "${BASH_SOURCE[0]}")/predict.sh" \
  --image test_images/sample.jpg \
  --task caption \
  --output outputs/test_caption.txt

echo "Smoke test completed successfully!"
