#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running CHURRO-3B smoke test..."
bash "${SCRIPT_DIR}/predict.sh" \
  --image /cvlization_repo/examples/perception/doc_ai/leaderboard/test_data/sample.jpg \
  --max-new-tokens 500 \
  --strip-xml \
  --output test_result.txt

echo "Smoke test completed successfully!"
echo "Output saved to: outputs/test_result.txt"
