#!/bin/bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-cvlization/rnj-1-instruct:latest}"

# Build from the script's directory
docker build --pull -t "$IMG" "$SCRIPT_DIR"
