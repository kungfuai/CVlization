#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building Gemma-3 Vision Docker image..."
docker build -t cvlization/gemma3-vision:latest -f Dockerfile .
echo "Build complete!"
