#!/usr/bin/env bash
set -euo pipefail
# Run one SearchOS open-domain research session in Docker.
#
#   ./run.sh                                    # offline demo (needs an LLM key only)
#   ./run.sh -- "your open-domain question"     # offline demo, custom query
#   ./run.sh -- --mode web "your question"      # live Serper/Tavily web search
#
# LLM key: pass whichever your --provider needs (default provider is `openai`):
#   OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY / ...
# Web mode also needs SERPER_API_KEY or TAVILY_API_KEY.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-searchos}"

# Host artifacts dir (bind-mounted to /work/outputs in the container).
# Defaults to <caller cwd>/searchos_outputs so `cvl run` (which sets
# CVL_WORK_DIR to the user's cwd) persists artifacts on the host.
WORK_DIR="${CVL_WORK_DIR:-$(pwd)}"
OUTPUTS_DIR="${SEARCHOS_OUTPUTS:-${WORK_DIR}/searchos_outputs}"
mkdir -p "${OUTPUTS_DIR}"

DOCKER_ARGS=(run --rm)
if [ -t 0 ] && [ -t 1 ]; then
  DOCKER_ARGS+=(-it)
fi

# Forward every recognized LLM / search key that is set in the host env.
ENV_ARGS=()
for var in \
  OPENAI_API_KEY ANTHROPIC_API_KEY DEEPSEEK_API_KEY ZHIPU_API_KEY ZAI_API_KEY \
  KIMI_API_KEY MOONSHOT_API_KEY MINIMAX_API_KEY DASHSCOPE_API_KEY ARK_API_KEY \
  OPENROUTER_API_KEY GEMINI_API_KEY XAI_API_KEY SILICONFLOW_API_KEY \
  SERPER_API_KEY TAVILY_API_KEY JINA_API_KEY \
  SF_PROVIDER SF_MODEL SF_FAST_MODEL SF_SEARCH_PROVIDER; do
  if [ -n "${!var:-}" ]; then
    ENV_ARGS+=(--env "${var}=${!var}")
  fi
done

docker "${DOCKER_ARGS[@]}" \
  ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
  "${ENV_ARGS[@]}" \
  --mount "type=bind,src=${OUTPUTS_DIR},dst=/work/outputs" \
  "${IMG}" --output-dir /work/outputs "$@"

echo "artifacts at: ${OUTPUTS_DIR}/" >&2
