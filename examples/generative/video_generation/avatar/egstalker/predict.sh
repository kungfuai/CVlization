#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

IMG="${CVL_IMAGE:-egstalker-infer}"

AUDIO_PATH="data/joyvasa_short.wav"
REFERENCE_PATH=""
MODEL_PATH=""
OUTPUT_DIR=""
BATCH_SIZE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --audio_path)
      shift; AUDIO_PATH="$1" ;;
    --reference_path)
      shift; REFERENCE_PATH="$1" ;;
    --model_path)
      shift; MODEL_PATH="$1" ;;
    --output_dir)
      shift; OUTPUT_DIR="$1" ;;
    --batch_size)
      shift; BATCH_SIZE="$1" ;;
    *)
      break ;;
  esac
  shift || true
done

REFERENCE_PATH="${REFERENCE_PATH:-datasets/coach2_longset}"
MODEL_PATH="${MODEL_PATH:-output/coach2_longset}"
OUTPUT_DIR="${OUTPUT_DIR:-results/coach2_longset}"
BATCH_SIZE="${BATCH_SIZE:-16}"

resolve_host_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    echo "$p"
  else
    echo "$SCRIPT_DIR/$p"
  fi
}

resolve_container_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    echo "$p"
  else
    echo "/workspace/host/$p"
  fi
}

MODEL_HOST=$(resolve_host_path "$MODEL_PATH")
REF_HOST=$(resolve_host_path "$REFERENCE_PATH")
OUT_HOST=$(resolve_host_path "$OUTPUT_DIR")

if [[ ! -d "$MODEL_HOST" ]]; then
  echo "model path $MODEL_HOST not found" >&2
  exit 1
fi

if [[ ! -f "$SCRIPT_DIR/$AUDIO_PATH" && ! "$AUDIO_PATH" = /* ]]; then
  echo "audio file $SCRIPT_DIR/$AUDIO_PATH not found" >&2
  exit 1
fi

mkdir -p "$OUT_HOST"
mkdir -p "$HOME/.cache/modelscope"
mkdir -p "$HOME/.cache/huggingface"

MODEL_ABS=$(resolve_container_path "$MODEL_PATH")
REF_ABS=$(resolve_container_path "$REFERENCE_PATH")
OUT_ABS=$(resolve_container_path "$OUTPUT_DIR")

AUDIO_ABS=$(resolve_container_path "$AUDIO_PATH")

set -- --audio_path "$AUDIO_ABS" --reference_path "$REF_ABS" --model_path "$MODEL_ABS" --output_dir "$OUT_ABS" --batch_size "$BATCH_SIZE" "$@"

docker run --rm --gpus=all \
    ${CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"} \
    --workdir /workspace/host \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace/host" \
    --mount "type=bind,src=${REPO_ROOT},dst=/cvlization_repo,readonly" \
    --mount "type=bind,src=${HOME}/.cache/modelscope,dst=/root/.cache/modelscope" \
    --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" \
    --env "PYTHONPATH=/workspace/host:/cvlization_repo" \
    --env "PYTHONUNBUFFERED=1" \
    "$IMG" \
    python /workspace/host/predict.py "$@"
