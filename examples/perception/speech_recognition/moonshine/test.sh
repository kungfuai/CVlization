#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# CPU smoke: tiny on the default CVL sample, JSON out.
./predict.sh \
  --audio sample \
  --model moonshine/tiny \
  --format json \
  --output moonshine_smoke.json
