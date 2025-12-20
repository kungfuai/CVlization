#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-nanoproof}"

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "Image ${IMAGE_NAME} not found. Build first with bash build.sh" >&2
  exit 1
fi

docker run --rm --gpus all \
  -e HF_TOKEN \
  -e NANOPROOF_TOKENIZER_REPO_ID \
  -e NANOPROOF_TOKENIZER_LOCAL \
  -e NANOPROOF_CHECKPOINT_REPO_ID \
  -e NANOPROOF_CHECKPOINT_LOCAL \
  -e NANOPROOF_MODEL_TAG \
  "${IMAGE_NAME}" \
  python3.12 predict.py "$@"
