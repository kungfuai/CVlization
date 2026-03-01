#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults: Anthropic backend, haiku model, 100K-line context
docker run --rm \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
  -e OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
  -e LLM_BACKEND="${LLM_BACKEND:-anthropic}" \
  -e MODEL="${MODEL:-claude-haiku-4-5-20251001}" \
  -e CONTEXT_LINES="${CONTEXT_LINES:-100000}" \
  -e MAX_ITERATIONS="${MAX_ITERATIONS:-10}" \
  rlm-needle python predict.py
