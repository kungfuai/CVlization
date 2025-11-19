#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running LightOnOCR smoke test with local vLLM inference"

bash "$SCRIPT_DIR/predict.sh" \
  --image "$SCRIPT_DIR/../test_images/sample.jpg" \
  --output outputs/test_result.txt \
  "$@"
