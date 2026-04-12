#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

./predict.sh \
  --audio sample \
  --model tiny.en \
  --device cpu \
  --compute-type int8 \
  --format json \
  --output faster_whisper_smoke.json
