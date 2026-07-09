#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${CVL_NO_DOCKER:-0}" = "1" ]; then
    cd "$SCRIPT_DIR"
    python3 eval_checkpoint.py --predict-script predict_eval_unsloth.py "$@"
else
    "$SCRIPT_DIR/run_unsloth_container.sh" \
        python3 eval_checkpoint.py --predict-script predict_eval_unsloth.py "$@"
fi
