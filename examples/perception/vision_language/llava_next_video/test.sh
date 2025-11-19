#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running LLaVA-NeXT-Video smoke test (default sample URL)"

bash "$SCRIPT_DIR/predict.sh" \
  --output outputs/test_result.txt \
  "$@"
