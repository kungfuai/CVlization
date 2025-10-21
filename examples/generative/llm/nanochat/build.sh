#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if nanochat source exists
NANOCHAT_SRC="$HOME/zz/nanochat"
if [ ! -d "$NANOCHAT_SRC" ]; then
    echo "Error: nanochat source not found at $NANOCHAT_SRC"
    echo "Please clone it first: git clone https://github.com/karpathy/nanochat $HOME/zz/nanochat"
    exit 1
fi

# Copy nanochat source to build context
echo "Copying nanochat source..."
rm -rf "$SCRIPT_DIR/nanochat"
cp -r "$NANOCHAT_SRC" "$SCRIPT_DIR/"

# Build Docker image
echo "Building nanochat Docker image..."
docker build -t nanochat "$SCRIPT_DIR"

# Cleanup
echo "Cleaning up..."
rm -rf "$SCRIPT_DIR/nanochat"

echo "Build complete! Docker image 'nanochat' is ready."
echo ""
echo "Next steps:"
echo "  bash train_single.sh  # Train on single GPU"
echo "  bash speedrun.sh      # Full speedrun (requires 8xH100)"
echo "  bash shell.sh         # Interactive shell"
