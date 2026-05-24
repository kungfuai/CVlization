#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Smoke: AWQ-4bit default on the bundled CVL dialogue prompt.
# Set QWEN3_OMNI_QUANT=bf16 to test the full model (requires ~80 GB VRAM).
./predict.sh --audio sample --output qwen3_omni_smoke.wav
