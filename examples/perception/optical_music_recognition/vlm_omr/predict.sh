#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-cvlization/vlm-omr:latest}"

mkdir -p "${HOME}/.cache/huggingface"

# Forward whichever API keys are set in the host environment
KEY_ARGS=()
[ -n "${GEMINI_API_KEY:-}" ]    && KEY_ARGS+=(--env "GEMINI_API_KEY=$GEMINI_API_KEY")
[ -n "${OPENAI_API_KEY:-}" ]    && KEY_ARGS+=(--env "OPENAI_API_KEY=$OPENAI_API_KEY")
[ -n "${ANTHROPIC_API_KEY:-}" ] && KEY_ARGS+=(--env "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY")

docker run --rm \
  ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
  --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
  --env "PYTHONPATH=/cvlization_repo" \
  --env "PYTHONUNBUFFERED=1" \
  --env "CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace}" \
  --env "CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
  "${KEY_ARGS[@]+"${KEY_ARGS[@]}"}" \
  "$IMG" python3 predict.py "$@"
