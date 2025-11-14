#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <adapter_name> [run_checkbox_qa.py args...]" >&2
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
if [[ "$ADAPTER" == qwen3* ]]; then
        EXAMPLE_DIR="$REPO_ROOT/examples/perception/vision_language/qwen3_vl"
        IMAGE_NAME="qwen3-vl"
        API_VAR="QWEN3_VL_API_BASE"
        HOST_PORT="${QWEN3_VL_SERVE_PORT:-8100}"
        HEALTH_URL="http://localhost:${HOST_PORT}/health"
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
        EXTRA_DOCKER_ENV=(
            "-e" "PHI4_MAX_MODEL_LEN=${PHI4_MAX_MODEL_LEN:-65536}"
            "-e" "HOST=0.0.0.0"
            "-e" "PORT=8000"
        )
else
    echo "Cannot infer model group from adapter '$ADAPTER'. Expected names starting with 'qwen3' or 'phi_'." >&2
    exit 1
fi

echo "Starting vLLM server (image: $IMAGE_NAME) ..."
CONTAINER_ID=$(docker run --rm -d --gpus all \
    -p "${HOST_PORT}:8000" \
    -v "$EXAMPLE_DIR:/workspace" \
    -w /workspace \
    "${EXTRA_DOCKER_ENV[@]}" \
    "$IMAGE_NAME" bash serve.sh)

cleanup() {
    echo "Stopping vLLM container $CONTAINER_ID"
    docker stop "$CONTAINER_ID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Waiting for server to become healthy..."
for _ in {1..60}; do
    if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
        break
    fi
    sleep 2
done

if ! curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
    echo "vLLM server did not become ready in time." >&2
    exit 1
fi

API_BASE="http://localhost:${HOST_PORT}/v1"
export "$API_VAR=$API_BASE"
echo "Using $API_VAR=$API_BASE"

python3 "$SCRIPT_DIR/run_checkbox_qa.py" "$ADAPTER" "$@"
