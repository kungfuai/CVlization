#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running SMT smoke test (inference on auto-downloaded sample)..."
bash "${SCRIPT_DIR}/predict.sh" \
  --output test_result.txt

echo ""
echo "Smoke test completed successfully!"
echo "Output saved to: outputs/test_result.txt"
