#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-cvl-nedreamer-dmc}"
TAG="${TAG:-latest}"

echo "=== Smoke test: NE-Dreamer DMC ==="

# Run a very short training (5k steps, 2 envs) to verify everything works
STEPS=5000 ENV_NUM=2 EVAL_EPISODES=1 \
  bash "${SCRIPT_DIR}/train.sh" \
    trainer.eval_every=100000 \
    model.compile=False

echo "=== Smoke test passed ==="
