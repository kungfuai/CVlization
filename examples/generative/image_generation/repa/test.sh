#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== REPA Smoke Test ==="
echo "Running short training on CIFAR-10 (500 steps)..."

# Run short training
"${SCRIPT_DIR}/train.sh" \
    --dataset cifar10 \
    --model SiT-B/2 \
    --max-train-steps 500 \
    --checkpointing-steps 500 \
    --batch-size 16 \
    --log-every 50 \
    --exp-name smoke-test \
    "$@"

echo ""
echo "=== Smoke test complete ==="
echo "Check outputs/smoke-test/ for results"
