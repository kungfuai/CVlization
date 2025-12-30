#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

IMG="${CVL_IMAGE:-smolagents_data_analyst}"

docker run --rm ${CVL_RUN_GPU:+--gpus="${CVL_RUN_GPU}"} \
	--shm-size 2G \
	${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
	--workdir /workspace \
	--mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
	--mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
	--mount "type=bind,src=${HOME}/.cache/cvlization,dst=/root/.cache/cvlization" \
	--env "PYTHONPATH=/cvlization_repo" \
	--env "PYTHONUNBUFFERED=1" \
	${ANALYST_LLM_PROVIDER:+--env ANALYST_LLM_PROVIDER=${ANALYST_LLM_PROVIDER}} \
	${ANALYST_LLM_MODEL:+--env ANALYST_LLM_MODEL=${ANALYST_LLM_MODEL}} \
	${ANALYST_LLM_TEMPERATURE:+--env ANALYST_LLM_TEMPERATURE=${ANALYST_LLM_TEMPERATURE}} \
	${OPENAI_API_KEY:+--env OPENAI_API_KEY=${OPENAI_API_KEY}} \
	${GROQ_API_KEY:+--env GROQ_API_KEY=${GROQ_API_KEY}} \
	${OLLAMA_BASE_URL:+--env OLLAMA_BASE_URL=${OLLAMA_BASE_URL}} \
	"$IMG" \
	python evaluate.py "$@"
