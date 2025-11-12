#!/usr/bin/env bash
set -euo pipefail

# Always run from this folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build the Docker image
docker build -t minicpm-v-2-6 "$SCRIPT_DIR"
