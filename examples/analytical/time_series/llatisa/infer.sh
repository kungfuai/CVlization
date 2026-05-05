#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

IMG="${CVL_IMAGE:-analytical_llatisa}"
CONTAINER_NAME="${CVL_CONTAINER_NAME:-}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "$HF_CACHE" "$SCRIPT_DIR/artifacts"

# Load .env if present
if [[ -f "$REPO_ROOT/.env" ]]; then
    echo "Loading defaults from $REPO_ROOT/.env" >&2
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

DOCKER_ARGS=(
    "--rm"
    "--shm-size" "16G"
    "--workdir" "/workspace"
    "--mount" "type=bind,src=${SCRIPT_DIR},dst=/workspace"
    "--mount" "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface"
    "--env" "PYTHONUNBUFFERED=1"
    "--env" "HF_HOME=/root/.cache/huggingface"
)

[[ -n "$CONTAINER_NAME" ]] && DOCKER_ARGS+=("--name" "$CONTAINER_NAME")
[[ "${CVL_ENABLE_GPU:-1}" == "1" ]] && DOCKER_ARGS+=("--gpus" "all")
[[ -n "${HF_TOKEN:-}" ]] && DOCKER_ARGS+=("--env" "HF_TOKEN=${HF_TOKEN}")

DOCKER_ARGS+=("$IMG" "python" "infer.py")

# Forward all script arguments to the Python script
for arg in "$@"; do DOCKER_ARGS+=("$arg"); done

docker run "${DOCKER_ARGS[@]}"
