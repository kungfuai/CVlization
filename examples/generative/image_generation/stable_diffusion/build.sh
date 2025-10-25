#!/usr/bin/env bash
set -euo pipefail

# Always run from this folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Image name
IMG="${CVL_IMAGE:-stable-diffusion}"

# Build the Docker image
docker build -t "$IMG" .
