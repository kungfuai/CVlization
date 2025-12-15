#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-dolphin-v2}"
# CVL CLI may look for underscore variant
ALT_IMG="${IMG//-/_}"

docker build -t "$IMG" "$SCRIPT_DIR"
if [ "$ALT_IMG" != "$IMG" ]; then
  docker tag "$IMG" "$ALT_IMG"
fi
