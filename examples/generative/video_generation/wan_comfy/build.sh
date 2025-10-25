#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build from the script's directory, works from anywhere

docker build -t wan_comfy "$SCRIPT_DIR"

# Download required models (idempotent - skips if already exists)
echo ""
echo "Downloading required models..."
bash "$SCRIPT_DIR/download_models.sh"
