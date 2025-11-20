#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_GEN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check if tokenizers exist, download if missing
if [ ! -f "$SCRIPT_DIR/comfy/text_encoders/llama_tokenizer/tokenizer.json" ]; then
    echo "Tokenizers not found. Downloading from HuggingFace..."
    if [ -f "$VIDEO_GEN_DIR/download_tokenizers.sh" ]; then
        cd "$VIDEO_GEN_DIR" && bash download_tokenizers.sh
        cd "$SCRIPT_DIR"
    else
        echo "ERROR: download_tokenizers.sh not found!"
        echo "Please run: bash examples/generative/video_generation/download_tokenizers.sh"
        exit 1
    fi
fi

# Build from the script's directory, works from anywhere
docker build -t vace_comfy "$SCRIPT_DIR"
