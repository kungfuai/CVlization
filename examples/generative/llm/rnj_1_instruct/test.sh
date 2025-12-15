#!/usr/bin/env bash
set -euo pipefail

# Smoke test for RNJ-1-Instruct
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running RNJ-1-Instruct smoke test..."
echo ""

# Run a simple code generation test with limited tokens
"$SCRIPT_DIR/predict.sh" \
  --prompt "Write a Python function to check if a number is prime. Keep it simple." \
  --max-tokens 256 \
  --temperature 0

echo ""
echo "Smoke test complete!"
