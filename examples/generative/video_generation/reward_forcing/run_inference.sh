#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${CVL_IMAGE:-reward_forcing:latest}"
CKPT_DIR="${CKPT_DIR:-${HOME}/.cache/cvlization/reward-forcing/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/videos}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
CODE_DIR="${CODE_DIR:-${REWARD_FORCING_SRC:-/tmp/Reward-Forcing}}"
CONFIG_PATH="${CONFIG_PATH:-/workspace/src/configs/reward_forcing.yaml}"
DEFAULT_CONFIG_PATH="${DEFAULT_CONFIG_PATH:-/workspace/src/configs/default_config.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/Reward-Forcing-T2V-1.3B/rewardforcing.pt}"
CHECKPOINT_REPO="${CHECKPOINT_REPO:-Wan-AI/Wan2.1-T2V-1.3B}"
CHECKPOINT_REVISION="${CHECKPOINT_REVISION:-}"
RF_REPO="${RF_REPO:-JaydenLu666/Reward-Forcing-T2V-1.3B}"
DATA_PATH="${DATA_PATH:-prompts/single_prompt.txt}"
NUM_FRAMES="${NUM_FRAMES:-21}"

# Allow inline prompt via -p|--prompt or PROMPT env
PROMPT="${PROMPT:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--prompt)
      PROMPT="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

mkdir -p "${CKPT_DIR}" "${OUTPUT_DIR}" "${HF_CACHE}"

if [[ ! -d "${CODE_DIR}" ]]; then
  echo "✗ CODE_DIR not found: ${CODE_DIR}" >&2
  exit 1
fi

if [[ -n "${PROMPT}" ]]; then
  mkdir -p "${OUTPUT_DIR}"
  PROMPT_FILE="${OUTPUT_DIR}/prompt_single.txt"
  printf "%s\n" "${PROMPT}" > "${PROMPT_FILE}"
  DATA_PATH="${PROMPT_FILE}"
fi

docker run --gpus all --rm \
  --workdir /workspace \
  --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
  --mount "type=bind,src=${CODE_DIR},dst=/workspace/src" \
  --mount "type=bind,src=${CKPT_DIR},dst=/workspace/checkpoints" \
  --mount "type=bind,src=${OUTPUT_DIR},dst=/workspace/videos" \
  --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
  ${HF_TOKEN:+-e HF_TOKEN="${HF_TOKEN}"} \
  -e CHECKPOINT_REPO="${CHECKPOINT_REPO}" \
  -e CHECKPOINT_REVISION="${CHECKPOINT_REVISION}" \
  -e RF_REPO="${RF_REPO}" \
  -e CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
  -e CONFIG_PATH="${CONFIG_PATH}" \
  -e DATA_PATH="${DATA_PATH}" \
  -e NUM_FRAMES="${NUM_FRAMES}" \
  -e HF_HOME=/root/.cache/huggingface \
  -e DEFAULT_CONFIG_PATH="${DEFAULT_CONFIG_PATH}" \
  -e PYTHONPATH=/workspace/src:${PYTHONPATH:-} \
  "${IMAGE}" \
  bash -lc 'set -e;
    # Ensure Reward-Forcing checkpoint and Wan base model are present
    python - <<'"'"'PY'"'"'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

base_ckpt = os.environ.get("CHECKPOINT_PATH", "checkpoints/Reward-Forcing-T2V-1.3B/rewardforcing.pt")
base_dir = Path(base_ckpt).parent if base_ckpt.endswith(".pt") else Path(base_ckpt)
base_dir.mkdir(parents=True, exist_ok=True)

# 1) Reward-Forcing weights
rf_repo = os.environ.get("RF_REPO", "JaydenLu666/Reward-Forcing-T2V-1.3B")
if not (base_dir / "rewardforcing.pt").exists():
    snapshot_download(repo_id=rf_repo, local_dir=str(base_dir), local_dir_use_symlinks=False)
    print("Downloaded Reward-Forcing weights from", rf_repo)

# 2) Wan base model (needed by wrappers)
wan_repo = os.environ.get("CHECKPOINT_REPO", "Wan-AI/Wan2.1-T2V-1.3B")
wan_dir = base_dir / "Wan2.1-T2V-1.3B"
if not wan_dir.exists():
    snapshot_download(repo_id=wan_repo, revision=os.environ.get("CHECKPOINT_REVISION") or None,
                      local_dir=str(wan_dir), local_dir_use_symlinks=False)
    print("Downloaded Wan base model from", wan_repo, "to", wan_dir)
PY

    # Apply overrides if present
    if [ -f /workspace/overrides/wan_modules_attention.py ]; then
      cp /workspace/overrides/wan_modules_attention.py /workspace/src/wan/modules/attention.py
    fi

    python /workspace/inference.py \
      --num_output_frames "$NUM_FRAMES" \
      --config_path "$CONFIG_PATH" \
      --checkpoint_path "$CHECKPOINT_PATH" \
      --output_folder videos/demo \
      --data_path "$DATA_PATH" \
      --use_ema
  '
