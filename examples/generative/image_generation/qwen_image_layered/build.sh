#!/bin/bash
set -e

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get image name from directory
IMAGE_NAME="$(basename "$SCRIPT_DIR")"

echo "Building Docker image: $IMAGE_NAME"
cd "$SCRIPT_DIR"
# BuildKit hangs at "exporting to image" on this setup; use the legacy builder.
DOCKER_BUILDKIT=0 docker build -t "$IMAGE_NAME" .
echo "Build complete: $IMAGE_NAME"
