#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build from the script's directory, works from anywhere
docker build -t soccer_game_visual_tracking "$SCRIPT_DIR"

# Download required data (idempotent - skips if already exists)
echo ""
echo "Downloading required data and models..."
bash "$SCRIPT_DIR/download_data.sh"
