#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== V3 Video Super-Resolution Smoke Test ==="
echo "This will download the checkpoint (~200MB) on first run."
echo ""

bash "${SCRIPT_DIR}/predict.sh" \
  --space-scale 2 \
  --time-scale 2

echo ""
echo "=== Smoke test passed ==="
