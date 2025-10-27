#!/bin/bash
# Launch interactive shell in nanochat container

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRIPT_DIR/data"
mkdir -p "$SCRIPT_DIR/outputs"

docker run --rm --gpus all \
    -v "$SCRIPT_DIR/data:/workspace/data" \
    -v "$SCRIPT_DIR/outputs:/workspace/outputs" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    nanochat \
    bash -c "cd nanochat && exec bash"
