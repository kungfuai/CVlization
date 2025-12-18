#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building Docker image: cvlization/live-avatar:latest"
docker build -t cvlization/live-avatar:latest .
echo "Build complete!"
