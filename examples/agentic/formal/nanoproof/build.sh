#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="${IMAGE_NAME:-nanoproof}"

echo "Building ${IMAGE_NAME}..."
docker build -t "${IMAGE_NAME}" -f Dockerfile .
echo "Done. Run predict via: docker run --rm -it --gpus all ${IMAGE_NAME} python predict.py --help"
