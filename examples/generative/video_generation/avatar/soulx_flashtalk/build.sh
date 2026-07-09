#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building Docker image: soulx_flashtalk"
docker build -t soulx_flashtalk .
echo "Build complete!"
