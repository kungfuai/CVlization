#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/predict.sh" \
  --prompt "Explain why the sky appears blue in one sentence." \
  --max_new_tokens 16 \
  --top_k 3 \
  --export json
