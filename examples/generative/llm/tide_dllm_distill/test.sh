#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Smoke test: self-distillation (10 steps) ==="
"$SCRIPT_DIR/train.sh" \
  --steps 10 \
  --batch-size 2 \
  --max-length 128 \
  --max-samples 100 \
  --log-interval 5 \
  --save-interval 10

echo ""
echo "=== Smoke test: predict (64 tokens, 64 steps) ==="
"$SCRIPT_DIR/predict.sh" \
  --prompt "What is 2 + 2? Answer briefly." \
  --steps 64 \
  --max-tokens 64

echo ""
echo "Smoke test complete!"
