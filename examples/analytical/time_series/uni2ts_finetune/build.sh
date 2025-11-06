#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_NAME="${CVL_IMAGE:-analytical_uni2ts_finetune}"
ALT_TAG="uni2ts-finetune"

docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"
docker tag "$IMAGE_NAME" "$ALT_TAG"

echo "Built image $IMAGE_NAME and tagged as $ALT_TAG"
