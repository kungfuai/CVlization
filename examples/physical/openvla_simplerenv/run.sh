#!/usr/bin/env bash
# Run the OpenVLA SimplerEnv demo server
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${CVL_IMAGE:-cvlization/openvla-simplerenv:latest}"
PORT="${PORT:-8000}"

echo "Starting OpenVLA SimplerEnv demo..."
echo "Image: ${IMAGE_NAME}"
echo "Port: ${PORT}"
echo ""

# Run with GPU support, mounting the example directory
docker run --rm -it \
    --gpus all \
    -p "${PORT}:8000" \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e DISPLAY="" \
    -e MUJOCO_GL=egl \
    -e ACCEPT_EULA=Y \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
    "${IMAGE_NAME}" \
    python server.py

echo ""
echo "Server stopped."
