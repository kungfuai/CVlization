#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

IMG="${CVL_IMAGE:-analytical_uni2ts_finetune}"
CONTAINER_NAME="${CVL_CONTAINER_NAME:-}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
CVL_CACHE="${HOME}/.cache/cvlization"

mkdir -p "$HF_CACHE" "$CVL_CACHE"

if [[ -f "$REPO_ROOT/.env" ]]; then
  echo "ℹ️  Loading defaults from $REPO_ROOT/.env (existing env takes precedence)" >&2
  while IFS= read -r export_cmd; do
    eval "$export_cmd"
  done < <(ENV_PATH="$REPO_ROOT/.env" python3 - <<'PY'
import os
from pathlib import Path
from shlex import quote

env_path = Path(os.environ["ENV_PATH"])

if env_path.exists():
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        # Remove surrounding quotes if present
        if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
            value = value[1:-1]
        print(f"export {key}={quote(value)}")
PY
)
fi

DOCKER_ARGS=(
  "--rm"
  "--shm-size" "16G"
  "--workdir" "/workspace"
  "--mount" "type=bind,src=${SCRIPT_DIR},dst=/workspace"
  "--mount" "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly"
  "--mount" "type=bind,src=${CVL_CACHE},dst=/root/.cache/cvlization"
  "--mount" "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface"
  "--env" "PYTHONPATH=/cvlization_repo"
  "--env" "PYTHONUNBUFFERED=1"
)

if [[ -n "$CONTAINER_NAME" ]]; then
  DOCKER_ARGS+=("--name" "$CONTAINER_NAME")
fi

if [[ "${CVL_ENABLE_GPU:-1}" == "1" ]]; then
  DOCKER_ARGS+=("--gpus" "all")
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  DOCKER_ARGS+=("--env" "HF_TOKEN=${HF_TOKEN}")
else
  echo "ℹ️  HF_TOKEN not provided. Public checkpoints only." >&2
fi

DOCKER_ARGS+=("$IMG" "python" "train.py")

if [[ "$#" -gt 0 ]]; then
  for arg in "$@"; do
    DOCKER_ARGS+=("$arg")
  done
fi

docker run "${DOCKER_ARGS[@]}"
