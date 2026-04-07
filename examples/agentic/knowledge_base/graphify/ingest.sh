#!/usr/bin/env bash
# Build a knowledge graph from a corpus directory using a coding agent (headless).
# Usage: ./ingest.sh [corpus_dir]
#   corpus_dir defaults to ./corpus
# Env vars:
#   AGENT              claude (default) | codex
#   ANTHROPIC_API_KEY  fallback creds when AGENT=claude and ~/.claude/.credentials.json absent
#   OPENAI_API_KEY     fallback creds when AGENT=codex and ~/.codex/auth.json absent
#   CVL_NO_DOCKER=1    run without Docker (agent CLI must be on PATH)
#   CVL_IMAGE          override Docker image name
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-cvl-knowledge-base-graphify}"
AGENT="${AGENT:-claude}"
CORPUS="${1:-./corpus}"

# Resolve corpus path to absolute (needed for Docker bind mount)
if [[ "${CORPUS}" != /* ]]; then
    CORPUS="${SCRIPT_DIR}/${CORPUS#./}"
fi

mkdir -p "${HOME}/.cache/cvlization"

if [[ "${CVL_NO_DOCKER:-}" == "1" ]]; then
    if [[ "${AGENT}" == "codex" ]]; then
        if ! command -v codex &>/dev/null; then
            echo "Error: codex not found. Install with: npm install -g @openai/codex" >&2
            exit 1
        fi
    else
        if ! command -v claude &>/dev/null; then
            echo "Error: claude not found. Install with: npm install -g @anthropic-ai/claude-code" >&2
            exit 1
        fi
        if ! python3 -c "import graphify" 2>/dev/null; then
            echo "Error: graphify not found. Install with: pip install graphifyy" >&2
            exit 1
        fi
    fi
    cd "${SCRIPT_DIR}"
    if [[ "${AGENT}" == "codex" ]]; then
        codex --approval-mode full-auto "/graphify ${CORPUS}"
    else
        claude -p "/graphify ${CORPUS}" --permission-mode bypassPermissions
    fi
else
    # Mount only the credentials file — avoid contamination from host CLAUDE.md,
    # settings.json (hooks), skills, and memory.
    CRED_MOUNTS=()
    CRED_ENVS=()

    if [[ "${AGENT}" == "codex" ]]; then
        CODEX_AUTH="${HOME}/.codex/auth.json"
        if [[ -f "${CODEX_AUTH}" ]]; then
            echo "Mounting ~/.codex/auth.json for Codex credentials"
            CRED_MOUNTS+=(
                "--mount" "type=bind,src=${CODEX_AUTH},dst=/root/.codex/auth.json,readonly"
            )
        elif [[ -n "${OPENAI_API_KEY:-}" ]]; then
            CRED_ENVS+=("--env" "OPENAI_API_KEY=${OPENAI_API_KEY}")
        else
            echo "Warning: no Codex credentials found. Set OPENAI_API_KEY or run 'codex login' first." >&2
        fi
    else
        CLAUDE_CREDS="${HOME}/.claude/.credentials.json"
        if [[ -f "${CLAUDE_CREDS}" ]]; then
            echo "Mounting ~/.claude/.credentials.json for Claude Code credentials"
            CRED_MOUNTS+=(
                "--mount" "type=bind,src=${CLAUDE_CREDS},dst=/root/.claude/.credentials.json,readonly"
            )
        elif [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
            CRED_ENVS+=("--env" "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}")
        else
            echo "Warning: no Claude credentials found. Run 'claude login' or set ANTHROPIC_API_KEY." >&2
        fi
    fi

    docker run --rm \
        ${CVL_RUN_GPU:+--gpus="${CVL_RUN_GPU}"} \
        ${CVL_CONTAINER_NAME:+--name "${CVL_CONTAINER_NAME}"} \
        --workdir /workspace \
        --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
        --mount "type=bind,src=${CORPUS},dst=/corpus,readonly" \
        --mount "type=bind,src=${HOME}/.cache/cvlization,dst=/root/.cache/cvlization" \
        "${CRED_MOUNTS[@]+"${CRED_MOUNTS[@]}"}" \
        "${CRED_ENVS[@]+"${CRED_ENVS[@]}"}" \
        --env AGENT="${AGENT}" \
        "${IMG}" \
        bash -c "
            if [[ \"\${AGENT}\" == 'codex' ]]; then
                codex --approval-mode full-auto '/graphify /corpus'
            else
                claude -p '/graphify /corpus' --permission-mode bypassPermissions
            fi
        "
fi
