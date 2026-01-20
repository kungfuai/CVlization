#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if vendored nanochat source exists
NANOCHAT_SRC="$SCRIPT_DIR/nanochat"
if [ ! -d "$NANOCHAT_SRC" ]; then
    echo "Error: vendored nanochat source not found at $NANOCHAT_SRC"
    echo "Please add it to examples/generative/llm/nanochat/nanochat"
    exit 1
fi

# Build Docker image
echo "Building nanochat Docker image..."
docker build -t nanochat "$SCRIPT_DIR"

echo "Build complete! Docker image 'nanochat' is ready."
echo ""
echo "Next steps:"
echo "  bash train_single.sh  # Train on single GPU"
echo "  bash speedrun.sh      # Full speedrun (requires 8xH100)"
echo "  bash shell.sh         # Interactive shell"
echo ""
echo "If the build stalls on export, try:"
echo "  DOCKER_BUILDKIT=0 bash build.sh"
