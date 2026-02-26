#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/verify_hf_hub.py" --strict
python "${SCRIPT_DIR}/download_assets.py" "$@"
python "${SCRIPT_DIR}/inspect_assets.py" "$@"
