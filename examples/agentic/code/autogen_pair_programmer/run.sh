#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

IMG="${CVL_IMAGE:-autogen_pair_programmer}"

docker run --rm ${CVL_RUN_GPU:+--gpus="${CVL_RUN_GPU}" } \
	${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME" } \
	--workdir /workspace \
	--mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
	--mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
	--mount "type=bind,src=${HOME}/.cache/cvlization,dst=/root/.cache/cvlization" \
	--env "PYTHONPATH=/cvlization_repo:/workspace" \
	--env "PYTHONUNBUFFERED=1" \
	${PAIR_LLM_PROVIDER:+--env PAIR_LLM_PROVIDER=${PAIR_LLM_PROVIDER}} \
	${PAIR_LLM_MODEL:+--env PAIR_LLM_MODEL=${PAIR_LLM_MODEL}} \
	${PAIR_LLM_TEMPERATURE:+--env PAIR_LLM_TEMPERATURE=${PAIR_LLM_TEMPERATURE}} \
	${OPENAI_API_KEY:+--env OPENAI_API_KEY=${OPENAI_API_KEY}} \
	${AZURE_OPENAI_API_KEY:+--env AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}} \
	${AZURE_OPENAI_ENDPOINT:+--env AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}} \
	${GROQ_API_KEY:+--env GROQ_API_KEY=${GROQ_API_KEY}} \
	${OLLAMA_BASE_URL:+--env OLLAMA_BASE_URL=${OLLAMA_BASE_URL}} \
	${LITELLM_API_KEY:+--env LITELLM_API_KEY=${LITELLM_API_KEY}} \
	"$IMG" \
	python pair_programmer.py "$@"
