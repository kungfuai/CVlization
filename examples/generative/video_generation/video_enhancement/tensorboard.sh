#!/bin/bash
# Run TensorBoard to visualize training progress
#
# Access from laptop via SSH tunnel:
#   ssh -L 6006:localhost:6006 user@remote-machine
# Then open: http://localhost:6006

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="cvl-video-enhancement"

# Check if image exists
if ! docker image inspect "${IMAGE_NAME}:latest" > /dev/null 2>&1; then
    echo "Docker image not found. Run ./build.sh first."
    exit 1
fi

# Find most recent log directory
LOG_DIR="${SCRIPT_DIR}/logs"
if [ ! -d "${LOG_DIR}" ]; then
    echo "No logs directory found. Run training first."
    exit 1
fi

echo "Starting TensorBoard..."
echo "Access at: http://localhost:6006"
echo "For remote access, use SSH tunnel: ssh -L 6006:localhost:6006 user@this-machine"
echo ""

docker run --rm \
    -p 6006:6006 \
    -v "${SCRIPT_DIR}:/workspace" \
    -w /workspace \
    "${IMAGE_NAME}:latest" \
    tensorboard --logdir=./logs --host=0.0.0.0 --port=6006
