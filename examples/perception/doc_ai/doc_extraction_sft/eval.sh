#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-doc_extraction_sft}"

if [ "${CVL_NO_DOCKER:-0}" = "1" ]; then
    cd "$SCRIPT_DIR"
    python3 eval_checkpoint.py "$@"
else
    "$SCRIPT_DIR/run_container.sh" python3 eval_checkpoint.py "$@"
fi
