#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-kosmos2-grounded-ocr}"

docker build -t "$IMG" "$SCRIPT_DIR"
