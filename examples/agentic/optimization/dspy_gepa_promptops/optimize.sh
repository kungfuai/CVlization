#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

IMG="${CVL_IMAGE:-dspy_gepa_promptops}"

docker run --rm ${CVL_RUN_GPU:+--gpus="${CVL_RUN_GPU}" } \
	${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME" } \
	--workdir /workspace \
	--mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
	--mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
	--mount "type=bind,src=${HOME}/.cache/cvlization,dst=/root/.cache/cvlization" \
	--env "PYTHONPATH=/cvlization_repo:/workspace" \
	${DSPY_LLM_PROVIDER:+--env DSPY_LLM_PROVIDER=${DSPY_LLM_PROVIDER}} \
	${DSPY_LLM_MODEL:+--env DSPY_LLM_MODEL=${DSPY_LLM_MODEL}} \
	${DSPY_LLM_TEMPERATURE:+--env DSPY_LLM_TEMPERATURE=${DSPY_LLM_TEMPERATURE}} \
	${OPENAI_API_KEY:+--env OPENAI_API_KEY=${OPENAI_API_KEY}} \
	${GROQ_API_KEY:+--env GROQ_API_KEY=${GROQ_API_KEY}} \
	${OLLAMA_BASE_URL:+--env OLLAMA_BASE_URL=${OLLAMA_BASE_URL}} \
	"$IMG" \
	python optimize.py "$@"
