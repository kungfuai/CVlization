#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CVL_GPUS="${CVL_GPUS:-0}" "$SCRIPT_DIR/train.sh" --config config_smoke.yaml
