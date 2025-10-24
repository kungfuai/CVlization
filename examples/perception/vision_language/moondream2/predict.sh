#!/usr/bin/env bash
set -euo pipefail

# Always run from this folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find repo root for cvlization package (go up 4 levels from example dir)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Inputs/Outputs: if CVL set these, great; else Python will default to ./inputs, ./outputs
IMG="${CVL_IMAGE:-moondream2}"

# In CVL docker mode, workspace is readonly; in standalone mode, it's writable for outputs
WORKSPACE_RO="${CVL_OUTPUTS:+,readonly}"

docker run --rm --gpus=all \
  ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
  --workdir /workspace \
  --mount "type=bind,src=$SCRIPT_DIR,dst=/workspace$WORKSPACE_RO" \
  --mount "type=bind,src=$REPO_ROOT,dst=/cvlization_repo,readonly" \
  --env "PYTHONPATH=/cvlization_repo" \
  --env "PYTHONUNBUFFERED=1" \
  ${CVL_INPUTS:+--mount "type=bind,src=$CVL_INPUTS,dst=/mnt/cvl/inputs,readonly"} \
  ${CVL_OUTPUTS:+--mount "type=bind,src=$CVL_OUTPUTS,dst=/mnt/cvl/outputs"} \
  ${CVL_INPUTS:+-e CVL_INPUTS=/mnt/cvl/inputs} \
  ${CVL_OUTPUTS:+-e CVL_OUTPUTS=/mnt/cvl/outputs} \
  "$IMG" bash -c "python3 predict.py $*"
