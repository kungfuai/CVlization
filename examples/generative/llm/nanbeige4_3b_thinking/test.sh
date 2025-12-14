#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/predict.sh" \
  --prompt "Which number is larger, 9.11 or 9.8? Please explain briefly." \
  --max-new-tokens 512 \
  --temperature 0.0 \
  --output-file smoke_test.txt \
  "$@"
