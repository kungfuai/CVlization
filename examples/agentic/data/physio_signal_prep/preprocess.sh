#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
IMAGE_NAME="${CVL_IMAGE:-physio_signal_prep}"

SPEC_PATH="specs/data_spec.sample.md"
if [[ $# -gt 0 && "$1" != --* ]]; then
  SPEC_PATH="$1"
  shift
fi

mkdir -p "${HOME}/.cache/cvlization"

if [[ "${CVL_NO_DOCKER:-}" == "1" ]]; then
  # Find python executable (prefer python3)
  PYTHON_CMD=""
  if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
  elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
  else
    echo "Error: Python not found. Please install Python 3.8+ first." >&2
    exit 1
  fi

  # Check for required dependencies
  if ! "${PYTHON_CMD}" -c "import mne" 2>/dev/null; then
    echo "" >&2
    echo "Error: Missing required dependencies for no-docker mode." >&2
    echo "" >&2
    echo "Install with:" >&2
    echo "  pip install mne==1.8.0 neurokit2==0.2.7 numpy==1.26.4 pandas==2.2.3 PyYAML==6.0.2 tqdm==4.66.5" >&2
    echo "" >&2
    echo "Or install all dependencies:" >&2
    echo "  pip install -r ${SCRIPT_DIR}/requirements.txt" >&2
    echo "" >&2
    echo "Or use Docker mode (remove --no-docker flag)." >&2
    exit 1
  fi

  export PYTHONPATH="${REPO_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"
  cd "${SCRIPT_DIR}"
  "${PYTHON_CMD}" scripts/preprocess_from_spec.py --spec "${SPEC_PATH}" "$@"
else
  docker run --rm \
    ${CVL_RUN_GPU:+--gpus="${CVL_RUN_GPU}"} \
    ${CVL_CONTAINER_NAME:+--name "${CVL_CONTAINER_NAME}"} \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    --mount "type=bind,src=${HOME}/.cache/cvlization,dst=/root/.cache/cvlization" \
    --env "PYTHONPATH=/cvlization_repo:/workspace" \
    "${IMAGE_NAME}" \
    python scripts/preprocess_from_spec.py --spec "${SPEC_PATH}" "$@"
fi
