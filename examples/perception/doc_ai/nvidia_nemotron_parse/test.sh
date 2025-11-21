#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running NVIDIA Nemotron Parse smoke test..."
bash "${SCRIPT_DIR}/predict.sh" \
  --image /cvlization_repo/examples/perception/doc_ai/leaderboard/test_data/sample.jpg \
  --format md \
  --output outputs/test_result.md

echo "Smoke test completed successfully!"
echo "Output saved to: outputs/test_result.md"
