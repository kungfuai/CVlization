#!/usr/bin/env bash
set -euo pipefail

FRESH_SERVER=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fresh-server)
            FRESH_SERVER=1
            shift
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [--fresh-server] <adapter_name> [run_checkbox_qa.py args...]" >&2
    echo "Example: $0 qwen3_vl_2b_multipage --subset data/subset_dev.jsonl" >&2
    exit 1
fi

ADAPTER="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

EXAMPLE_DIR=""
IMAGE_NAME=""
API_VAR=""
PORT=""
HEALTH_URL=""
CONTAINER_NAME_DEFAULT=""
if [[ "$ADAPTER" == qwen3* ]]; then
        EXAMPLE_DIR="$REPO_ROOT/examples/perception/vision_language/qwen3_vl"
        IMAGE_NAME="qwen3-vl"
        API_VAR="QWEN3_VL_API_BASE"
        HOST_PORT="${QWEN3_VL_SERVE_PORT:-8100}"
        HEALTH_URL="http://localhost:${HOST_PORT}/health"
        CONTAINER_NAME_DEFAULT="checkbox_qwen3_vllm"
        EXTRA_DOCKER_ENV=(
            "-e" "QWEN3_VL_VARIANT=${QWEN3_VL_VARIANT:-2b}"
            "-e" "QWEN3_VL_MAX_MODEL_LEN=${QWEN3_VL_MAX_MODEL_LEN:-65536}"
            "-e" "HOST=0.0.0.0"
            "-e" "PORT=8000"
        )
elif [[ "$ADAPTER" == phi_* ]]; then
        EXAMPLE_DIR="$REPO_ROOT/examples/perception/vision_language/phi_4_multimodal_instruct"
        IMAGE_NAME="phi-4-multimodal-instruct"
        API_VAR="PHI4_API_BASE"
        HOST_PORT="${PHI4_SERVE_PORT:-8200}"
        HEALTH_URL="http://localhost:${HOST_PORT}/health"
        CONTAINER_NAME_DEFAULT="checkbox_phi4_vllm"
        EXTRA_DOCKER_ENV=(
            "-e" "PHI4_MAX_MODEL_LEN=${PHI4_MAX_MODEL_LEN:-8192}"
            "-e" "HOST=0.0.0.0"
            "-e" "PORT=8000"
        )
else
    echo "Cannot infer model group from adapter '$ADAPTER'. Expected names starting with 'qwen3' or 'phi_'." >&2
    exit 1
fi

CONTAINER_NAME="${VLLM_CONTAINER_NAME:-$CONTAINER_NAME_DEFAULT}"
STARTED_CONTAINER=0

RUNNING_ID="$(docker ps --filter "name=^${CONTAINER_NAME}$" --format '{{.ID}}' || true)"
EXISTING_ID="$(docker ps -a --filter "name=^${CONTAINER_NAME}$" --format '{{.ID}}' || true)"
if [[ $FRESH_SERVER -eq 1 && -n "$EXISTING_ID" ]]; then
    echo "Removing existing container $CONTAINER_NAME for fresh start..."
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    RUNNING_ID=""
    EXISTING_ID=""
fi

if [[ -n "$RUNNING_ID" ]]; then
    CONTAINER_ID="$RUNNING_ID"
    echo "Reusing running vLLM container $CONTAINER_NAME ($CONTAINER_ID)"
else
    if [[ -n "$EXISTING_ID" ]]; then
        echo "Cleaning up stopped container $CONTAINER_NAME before start..."
        docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
    echo "Starting vLLM server (image: $IMAGE_NAME, container: $CONTAINER_NAME) ..."
    echo "Command: docker run -d --gpus all --name $CONTAINER_NAME -p ${HOST_PORT}:8000 -v $EXAMPLE_DIR:/workspace -v $HOME/.cache/huggingface:/root/.cache/huggingface -w /workspace ${EXTRA_DOCKER_ENV[*]} $IMAGE_NAME bash serve.sh"

    CONTAINER_ID=$(docker run -d --gpus all \
        --name "$CONTAINER_NAME" \
        -p "${HOST_PORT}:8000" \
        -v "$EXAMPLE_DIR:/workspace" \
        -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
        -w /workspace \
        "${EXTRA_DOCKER_ENV[@]}" \
        "$IMAGE_NAME" bash serve.sh)

    echo "Container started with ID: $CONTAINER_ID"
    sleep 2
    echo "Initial logs:"
    docker logs "$CONTAINER_NAME" 2>&1 | head -20

    STARTED_CONTAINER=1
fi

cleanup() {
    if [[ $STARTED_CONTAINER -eq 1 ]]; then
        echo "Stopping vLLM container $CONTAINER_ID"
        docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
        docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

echo "Waiting for server to become healthy at $HEALTH_URL..."
echo "Container ID: $CONTAINER_ID"
echo "Container name: $CONTAINER_NAME"

for i in {1..150}; do
    # Check if container is still running (more reliable check)
    if ! docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        echo "vLLM container no longer exists. Showing logs..." >&2
        docker logs "$CONTAINER_NAME" 2>&1 | tail -200 >&2 || true
        exit 1
    fi

    CONTAINER_STATUS=$(docker inspect --format='{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "unknown")
    if [ "$CONTAINER_STATUS" != "running" ]; then
        echo "vLLM container exited with status: $CONTAINER_STATUS" >&2
        echo "--- Full container logs ---" >&2
        docker logs "$CONTAINER_NAME" 2>&1 | tail -200 >&2 || true
        exit 1
    fi

    if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
        echo "✓ Server is healthy after ${i} checks"
        break
    fi

    # Show progress every 10 iterations (20 seconds)
    if [ $((i % 10)) -eq 0 ]; then
        echo "Still waiting... (${i}/150, container status: $CONTAINER_STATUS)"
        docker logs "$CONTAINER_NAME" 2>&1 | tail -3
    fi

    sleep 2
done

if ! curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
    echo "vLLM server did not become ready in time." >&2
    echo "--- vLLM container logs ($CONTAINER_ID) ---" >&2
    docker logs --tail 200 "$CONTAINER_ID" >&2 || true
    exit 1
fi

API_BASE="http://localhost:${HOST_PORT}/v1"
export "$API_VAR=$API_BASE"
echo "Using $API_VAR=$API_BASE"

python3 "$SCRIPT_DIR/run_checkbox_qa.py" "$ADAPTER" "$@"
