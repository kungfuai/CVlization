#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${CVL_WORK_DIR:-${WORK_DIR:-$(pwd)}}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

IMG="${CVL_IMAGE:-langgraph_helpdesk}"

docker run --rm ${CVL_RUN_GPU:+--gpus="${CVL_RUN_GPU}"} \
	--shm-size 16G \
	${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
	--workdir /workspace \
	--mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
	--mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
	--mount "type=bind,src=${HOME}/.cache/cvlization,dst=/root/.cache/cvlization" \
	--env "PYTHONPATH=/cvlization_repo" \
	--env "PYTHONUNBUFFERED=1" \
  --mount "type=bind,src=${WORK_DIR},dst=/mnt/cvl/workspace" \
  --env "CVL_INPUTS=${CVL_INPUTS:-/mnt/cvl/workspace}" \
  --env "CVL_OUTPUTS=${CVL_OUTPUTS:-/mnt/cvl/workspace}" \
	${OLLAMA_HOST:+--env OLLAMA_HOST=${OLLAMA_HOST}} \
	${OPENAI_API_KEY:+--env OPENAI_API_KEY=${OPENAI_API_KEY}} \
	${OPENAI_BASE_URL:+--env OPENAI_BASE_URL=${OPENAI_BASE_URL}} \
	${GROQ_API_KEY:+--env GROQ_API_KEY=${GROQ_API_KEY}} \
	"$IMG" \
	python predict.py "$@"
