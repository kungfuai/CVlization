#!/usr/bin/env bash
# Build a knowledge graph from a corpus directory using a coding agent (headless).
# Usage: ./ingest.sh [corpus_dir]
#   corpus_dir defaults to ./corpus
# Env vars:
#   AGENT              claude (default) | codex
#   ANTHROPIC_API_KEY  alternative auth when AGENT=claude (overrides ~/.claude credentials)
#   OPENAI_API_KEY     alternative auth when AGENT=codex (overrides ~/.codex credentials)
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
        command -v codex &>/dev/null || { echo "Error: codex not found. Install: npm install -g @openai/codex" >&2; exit 1; }
    else
        command -v claude &>/dev/null || { echo "Error: claude not found. Install: npm install -g @anthropic-ai/claude-code" >&2; exit 1; }
        python3 -c "import graphify" 2>/dev/null || { echo "Error: graphify not found. Install: pip install graphifyy" >&2; exit 1; }
    fi
    cd "${SCRIPT_DIR}"
    if [[ "${AGENT}" == "codex" ]]; then
        codex --approval-mode full-auto "/graphify ${CORPUS}"
    else
        claude -p "/graphify ${CORPUS}" --permission-mode bypassPermissions
    fi
    exit 0
fi

# Docker mode: run as the current host user to avoid root restrictions and
# ensure output files are owned by the correct user.
CRED_MOUNTS=()
CRED_ENVS=()

if [[ "${AGENT}" == "codex" ]]; then
    CODEX_AUTH="${HOME}/.codex/auth.json"
    if [[ -n "${OPENAI_API_KEY:-}" ]]; then
        CRED_ENVS+=("--env" "OPENAI_API_KEY=${OPENAI_API_KEY}")
    elif [[ -f "${CODEX_AUTH}" ]]; then
        echo "Mounting ~/.codex/auth.json for Codex credentials"
        CRED_MOUNTS+=("--mount" "type=bind,src=${CODEX_AUTH},dst=/tmp/cvl-home/.codex/auth.json")
    else
        echo "Warning: no Codex credentials found. Set OPENAI_API_KEY or run 'codex login' first." >&2
    fi
else
    CLAUDE_CREDS="${HOME}/.claude/.credentials.json"
    if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
        CRED_ENVS+=("--env" "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}")
    elif [[ -f "${CLAUDE_CREDS}" ]]; then
        echo "Mounting ~/.claude/.credentials.json for Claude Code credentials"
        # Writable so claude can refresh OAuth tokens
        CRED_MOUNTS+=("--mount" "type=bind,src=${CLAUDE_CREDS},dst=/tmp/cvl-home/.claude/.credentials.json")
    else
        echo "Warning: no Claude credentials found. Run 'claude login' or set ANTHROPIC_API_KEY." >&2
    fi
fi

docker run --rm \
    --user "$(id -u):$(id -g)" \
    --env HOME=/tmp/cvl-home \
    ${CVL_RUN_GPU:+--gpus="${CVL_RUN_GPU}"} \
    ${CVL_CONTAINER_NAME:+--name "${CVL_CONTAINER_NAME}"} \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
    --mount "type=bind,src=${CORPUS},dst=/corpus,readonly" \
    --mount "type=bind,src=${HOME}/.cache/cvlization,dst=/tmp/cvl-home/.cache/cvlization" \
    "${CRED_MOUNTS[@]+"${CRED_MOUNTS[@]}"}" \
    "${CRED_ENVS[@]+"${CRED_ENVS[@]}"}" \
    --env AGENT="${AGENT}" \
    "${IMG}" \
    bash -c "
        # Copy pre-installed graphify skill into the runtime home (no-clobber so
        # the bind-mounted credentials file is not overwritten)
        cp -rn /opt/graphify-skill/. /tmp/cvl-home/ 2>/dev/null || true

        # Fail fast: verify the agent can authenticate before running the full pipeline
        echo '[graphify] Checking agent authentication...'
        if [[ \"\${AGENT}\" == 'codex' ]]; then
            codex --approval-mode full-auto 'Reply with only: ok' || { echo 'Error: Codex agent auth failed.' >&2; exit 1; }
            codex --approval-mode full-auto '/graphify /corpus'
        else
            claude -p 'Reply with only: ok' --permission-mode bypassPermissions || { echo 'Error: Claude agent auth failed.' >&2; exit 1; }
            claude -p '/graphify /corpus' --permission-mode bypassPermissions
        fi
    "
