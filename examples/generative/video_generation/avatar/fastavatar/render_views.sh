#!/bin/bash

# Novel View Rendering Script for FastAvatar
# ==========================================
# Renders novel views from generated Gaussian splats

set -e

# Default values
PLY_PATH="results/lenna/splats.ply"
OUTPUT_DIR="results/lenna_novel_views"
NUM_VIEWS=8
ELEVATION=0.0
WIDTH=512
HEIGHT=512

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ply_path)
            PLY_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_views)
            NUM_VIEWS="$2"
            shift 2
            ;;
        --elevation)
            ELEVATION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "==================================="
echo "FastAvatar Novel View Rendering"
echo "==================================="
echo "PLY file: $PLY_PATH"
echo "Output: $OUTPUT_DIR"
echo "Views: $NUM_VIEWS"
echo "Elevation: $ELEVATION°"
echo "==================================="

# Run in Docker container
docker run --rm \
    --gpus all \
    --mount "type=bind,src=$(pwd),dst=/workspace" \
    --mount "type=bind,src=${HOME}/.cache,dst=/root/.cache" \
    -w /workspace \
    fastavatar:latest \
    python scripts/render_novel_views.py \
        --ply_path "$PLY_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --num_views "$NUM_VIEWS" \
        --elevation "$ELEVATION" \
        --width "$WIDTH" \
        --height "$HEIGHT"

echo ""
echo "Novel views rendered successfully!"
echo "Output directory: $OUTPUT_DIR"
