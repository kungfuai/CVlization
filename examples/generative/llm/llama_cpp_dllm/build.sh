#!/usr/bin/env bash
# This example reuses the cvl-llama-cpp image (which already ships
# llama-diffusion-cli alongside llama-server). Build the sibling example
# if the image isn't present, otherwise nothing to do.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${LLAMA_CPP_IMAGE:-cvl-llama-cpp}"

if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Reusing existing ${IMAGE} image."
else
  SIBLING="$(cd "${SCRIPT_DIR}/../llama_cpp" && pwd)"
  echo "${IMAGE} not present; building via sibling preset (${SIBLING}/build.sh)..."
  bash "${SIBLING}/build.sh"
fi
