#!/usr/bin/env bash
set -euo pipefail

# Smoke test for MiniCPM-V-2.6
echo "Running MiniCPM-V-2.6 smoke test..."

# Test caption task
bash "$(dirname "${BASH_SOURCE[0]}")/predict.sh" \
  --image test_images/sample.jpg \
  --task caption \
  --output outputs/test_caption.txt

echo "Smoke test completed successfully!"
