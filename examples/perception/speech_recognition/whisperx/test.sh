#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

./predict.sh \
  --audio sample \
  --model tiny.en \
  --language en \
  --device cpu \
  --compute-type int8 \
  --batch-size 1 \
  --format json \
  --output whisperx_smoke.json
