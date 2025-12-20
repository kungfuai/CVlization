#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-gemma_scope_2_270m_it}"

docker build -t "$IMG" "$SCRIPT_DIR"
