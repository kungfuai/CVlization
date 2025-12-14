#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running dLLM Qwen BD3LM smoke test..."
echo ""

# Run with reduced steps for faster testing
"$SCRIPT_DIR/predict.sh" \
  --prompt "What is 2 + 2? Answer briefly." \
  --steps 64 \
  --max-tokens 64

echo ""
echo "Smoke test complete!"
