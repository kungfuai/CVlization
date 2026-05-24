#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Smoke: bundled CVL sample as both user query and voice-clone reference
# (self-clone). bf16 by default; set CHROMA_QUANT=4bit to test the
# consumer-card path.
./predict.sh --audio sample --output chroma_smoke.wav
