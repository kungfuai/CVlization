#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-cvl-stable-worldmodel-dmc}"
TAG="${TAG:-latest}"
CACHE_DIR="${HOME}/.cache/cvlization"

mkdir -p "${CACHE_DIR}/huggingface" "${CACHE_DIR}/stable-worldmodel"

if ! docker image inspect "${IMAGE_NAME}:${TAG}" >/dev/null 2>&1; then
    echo "Image ${IMAGE_NAME}:${TAG} not found. Building..."
    "${SCRIPT_DIR}/build.sh"
fi

DOCKER_ARGS=(--gpus all)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    DOCKER_ARGS=(--gpus "device=${CUDA_VISIBLE_DEVICES}")
fi

if [ $# -eq 0 ]; then
    CMD='python verify_hf_hub.py --strict && \
python download_assets.py --target-dir /cvl-cache/stable-worldmodel/assets --splits expert && \
python inspect_assets.py --asset-dir /cvl-cache/stable-worldmodel/assets --splits expert && \
python run_eval.py --asset-dir /cvl-cache/stable-worldmodel/assets --steps 200 --num-envs 1 --device cpu'
else
    CMD="$*"
fi

echo "Running in Docker with cache: ${CACHE_DIR}"

TTY_FLAGS="-it"
if [ ! -t 0 ]; then
    TTY_FLAGS="-i"
fi

docker run --rm \
    ${TTY_FLAGS} \
    "${DOCKER_ARGS[@]}" \
    --shm-size=8g \
    -v "${CACHE_DIR}:/cvl-cache" \
    -v "${SCRIPT_DIR}:/workspace" \
    -w /workspace \
    -e HF_HOME=/cvl-cache/huggingface \
    -e STABLEWM_HOME=/cvl-cache/stable-worldmodel \
    -e XDG_CACHE_HOME=/cvl-cache \
    -e MUJOCO_GL=egl \
    "${IMAGE_NAME}:${TAG}" \
    bash -lc "${CMD}"
