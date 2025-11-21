#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="$(basename "$SCRIPT_DIR")"
CACHE_DIR="${HOME}/.cache/cvlization"
mkdir -p "$CACHE_DIR"

# Default values
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
COMPILE="${COMPILE:-false}"

echo "Krea Realtime WebSocket Server"
echo "==============================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "Torch Compile: $COMPILE"
echo ""
echo "Starting server..."
echo "Web UI will be available at: http://localhost:$PORT/"
echo "Press Ctrl+C to stop"
echo ""

# Run server with port exposed
docker run --rm --gpus=all \
    -p "$PORT:$PORT" \
    -v "$CACHE_DIR:/root/.cache" \
    -v "$SCRIPT_DIR:/workspace" \
    --workdir /workspace \
    "$IMAGE_NAME" \
    python3 serve.py \
        --host "$HOST" \
        --port "$PORT" \
        $([ "$COMPILE" = "true" ] && echo "--compile")
