#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run smoke test with a simple math problem
echo "Running Nomos-1 smoke test..."
"$SCRIPT_DIR/predict.sh" \
  --prompt "What is 2 + 2? Answer briefly." \
  --max_new_tokens 100 \
  --greedy

echo ""
echo "Smoke test completed. Check outputs/result.txt for response."
