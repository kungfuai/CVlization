#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/predict.sh" \
  --prompt "Give a short caption of the main content." \
  --max-tokens 48 \
  "$@"
