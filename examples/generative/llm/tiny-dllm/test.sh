#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Smoke test: train (100 iters) ==="
"$SCRIPT_DIR/train.sh" --iters 100 --eval-interval 50

echo "=== Smoke test: predict (100 tokens) ==="
"$SCRIPT_DIR/predict.sh" --max-tokens 100
