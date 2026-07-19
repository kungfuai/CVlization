#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

USE_GPU=1 ./predict.sh \
  --audio sample \
  --max-new-tokens 2048 \
  --output moss_smoke.json
