#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Smoke: Moshiko on the bundled CVL sample (one user turn -> one Moshi turn).
./predict.sh --audio sample --output moshi_smoke.wav
