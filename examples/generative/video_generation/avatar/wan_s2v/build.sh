#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker build -t "${CVL_IMAGE:-wan_s2v}" "$SCRIPT_DIR"
