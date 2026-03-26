#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${DOCUMENT_PATH:-}" ]]; then
  echo "Usage: DOCUMENT_PATH=/path/to/file.txt QUERY='your question' bash predict.sh"
  exit 1
fi

if [[ -z "${QUERY:-}" ]]; then
  echo "Usage: DOCUMENT_PATH=/path/to/file.txt QUERY='your question' bash predict.sh"
  exit 1
fi

# Resolve document to absolute path (portable: Linux and macOS)
if command -v realpath &>/dev/null; then
  DOC_ABS="$(realpath "$DOCUMENT_PATH")"
else
  DOC_ABS="$(cd "$(dirname "$DOCUMENT_PATH")" && pwd)/$(basename "$DOCUMENT_PATH")"
fi
DOC_DIR="$(dirname "$DOC_ABS")"
DOC_NAME="$(basename "$DOC_ABS")"

# The example directory is mounted as /workspace so Claude Code finds
# .claude/skills/ and .claude/agents/ automatically.
docker run --rm \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${DOC_DIR},dst=/docs,readonly" \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
  -e MODEL="${MODEL:-claude-sonnet-4-6}" \
  "${IMG:-rlm-claude-code}" \
  claude --print --dangerously-skip-permissions \
    "/rlm context=/docs/${DOC_NAME} query=${QUERY}"
