#!/bin/bash
# Launch interactive shell in nanochat container

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRIPT_DIR/data"
mkdir -p "$SCRIPT_DIR/outputs"

docker run --rm --gpus all \
    --workdir /workspace \
    -v "$SCRIPT_DIR:/workspace" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    nanochat \
    bash -c "cd /workspace/nanochat && exec bash"
