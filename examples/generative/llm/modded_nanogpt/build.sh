#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build the Docker image for modded-nanogpt
echo "Building modded-nanogpt Docker image..."
docker build -t modded-nanogpt "$SCRIPT_DIR"

if [ $? -eq 0 ]; then
    echo "Successfully built modded-nanogpt image"
else
    echo "Failed to build Docker image"
    exit 1
fi