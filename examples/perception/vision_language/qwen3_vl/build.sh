#!/usr/bin/env bash
set -euo pipefail

# Always run from this folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build the Docker image (single image supports 2B/4B/8B variants)
docker build -t qwen3-vl "$SCRIPT_DIR"
