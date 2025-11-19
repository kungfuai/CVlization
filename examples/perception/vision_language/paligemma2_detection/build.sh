#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-paligemma2-detection}"

docker build -t "$IMG" "$SCRIPT_DIR"
