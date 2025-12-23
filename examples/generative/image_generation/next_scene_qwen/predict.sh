#!/bin/bash
set -e

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
# Find repo root for cvlization package
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"

# Get image name from directory
IMAGE_NAME="$(basename "$SCRIPT_DIR")"

# Define cache directory
CACHE_DIR="${HOME}/.cache/cvlization"
mkdir -p "$CACHE_DIR"

# Default prompt
PROMPT="${1:-The camera moves slightly forward as sunlight breaks through clouds}"

echo "Generating next scene..."
echo "Prompt: Next Scene: $PROMPT"

docker run --rm --gpus=all \
    -v "$CACHE_DIR:/root/.cache" \
    -v "$SCRIPT_DIR:/workspace" \
    --workdir /workspace \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    -v "$WORK_DIR:/mnt/cvl/workspace" \
    -e CVL_INPUTS="${CVL_INPUTS:-/mnt/cvl/workspace}" \
    -e CVL_OUTPUTS="${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
    "$IMAGE_NAME" \
    python3 generate_scene.py \
        --prompt "$PROMPT" \
        "${@:2}"

echo "Scene generation complete!"
echo "Output saved to: outputs/scene.png"
