#!/usr/bin/env bash
# Shared docker-run helper for the MemSlides example.
# Usage: _run.sh <python_script> [args...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

IMAGE_NAME="${CVL_IMAGE:-memslides}"
ENTRY="$1"; shift

# Persist all learned memory (user profiles, sessions, tool recipes) on the host
# cache so a second run reuses it instead of re-learning.
HOST_CACHE="${HOME}/.cache/cvlization"
mkdir -p "${HOST_CACHE}"

docker run --rm \
  ${CVL_CONTAINER_NAME:+--name "${CVL_CONTAINER_NAME}"} \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
  --mount "type=bind,src=${HOST_CACHE},dst=/root/.cache/cvlization" \
  --env "PYTHONPATH=/cvlization_repo:/workspace" \
  ${MEMSLIDES_PROVIDER:+--env MEMSLIDES_PROVIDER=${MEMSLIDES_PROVIDER}} \
  ${MEMSLIDES_MODEL:+--env MEMSLIDES_MODEL=${MEMSLIDES_MODEL}} \
  ${OPENAI_API_KEY:+--env OPENAI_API_KEY=${OPENAI_API_KEY}} \
  ${GROQ_API_KEY:+--env GROQ_API_KEY=${GROQ_API_KEY}} \
  ${OLLAMA_BASE_URL:+--env OLLAMA_BASE_URL=${OLLAMA_BASE_URL}} \
  "${IMAGE_NAME}" \
  python "${ENTRY}" "$@"
