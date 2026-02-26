#!/usr/bin/env bash
# Smoke test: generate a proof for a simple induction problem.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
IMAGE="${QED_NANO_IMAGE:-qed-nano}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Image '${IMAGE}' not found. Build first with: bash build.sh" >&2
  exit 1
fi

mkdir -p "$SCRIPT_DIR/outputs"

PROBLEM="Let n be a positive integer. Prove that 1 + 2 + ... + n = n(n+1)/2."

echo "Running smoke test..."
echo "Problem: ${PROBLEM}"
echo

docker run --rm --gpus all --ipc=host \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${SCRIPT_DIR}:/workspace" \
  -v "${REPO_ROOT}:/cvlization_repo:ro" \
  -w /workspace \
  -e PYTHONPATH="/cvlization_repo" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e MODEL_ID="${MODEL_ID:-lm-provers/QED-Nano}" \
  "${IMAGE}" \
  python3 predict.py \
    --problem "${PROBLEM}" \
    --max-tokens 4096 \
    --output outputs/test_proof.txt

echo
echo "--- Output ---"
cat "$SCRIPT_DIR/outputs/test_proof.txt"
echo
echo "Smoke test passed."
