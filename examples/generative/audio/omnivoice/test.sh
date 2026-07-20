#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMG="${CVL_IMAGE:-omnivoice}"

echo "=== OmniVoice Smoke Test ==="
echo ""

mkdir -p "${HOME}/.cache/huggingface"

run_docker() {
    docker run --rm --gpus=all \
      --workdir /workspace \
      --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
      --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
      --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
      --env "PYTHONPATH=/cvlization_repo" \
      --env "PYTHONUNBUFFERED=1" \
      --env "HF_HOME=/root/.cache/huggingface" \
      "$IMG" python predict.py "$@"
}

check_output() {
    local FILE="$1"
    local LABEL="$2"
    if [ -f "$FILE" ]; then
        SIZE=$(stat -f%z "$FILE" 2>/dev/null || stat -c%s "$FILE" 2>/dev/null)
        echo "  Output: $FILE ($SIZE bytes)"
        if [ "$SIZE" -gt 4000 ]; then
            echo "  $LABEL PASSED"
        else
            echo "  $LABEL FAILED: audio file too small ($SIZE bytes)"
            exit 1
        fi
    else
        echo "  $LABEL FAILED: output file not created"
        exit 1
    fi
}

# Test 1: Voice design mode
echo "--- Test 1: Voice design ---"
run_docker \
    --text "This is a test of OmniVoice voice design mode." \
    --instruct "female, British accent" \
    --output outputs/test_design.wav \
    --num-step 16

check_output "${SCRIPT_DIR}/outputs/test_design.wav" "Voice design"
echo ""

# Test 2: Voice cloning mode (uses canonical ref audio from zzsi/cvl)
echo "--- Test 2: Voice cloning ---"
run_docker \
    --text "This is a test of OmniVoice voice cloning mode." \
    --output outputs/test_clone.wav \
    --num-step 16

check_output "${SCRIPT_DIR}/outputs/test_clone.wav" "Voice cloning"
echo ""

# Clean up test outputs
rm -f "${SCRIPT_DIR}/outputs/test_design.wav" "${SCRIPT_DIR}/outputs/test_clone.wav"

echo "=== All tests PASSED ==="
