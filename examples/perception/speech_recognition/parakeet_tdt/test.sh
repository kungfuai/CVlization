#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Smoke: default 1.1b on the bundled CVL sample, output JSON.
./predict.sh \
  --audio sample \
  --model nvidia/parakeet-tdt-1.1b \
  --device auto \
  --format json \
  --output parakeet_tdt_smoke.json
