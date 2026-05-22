#!/usr/bin/env bash
# Generate portraits (Flux) + speech (CosyVoice3) for every manifest item.
#
# Runs both generators on a SINGLE GPU (default: GPU 1) sequentially, to be a
# polite co-tenant on a shared host. Flux first (~16 s/item), then CosyVoice3
# (~5 s/item).
#
# Usage:
#   ./generate_assets.sh <data_dir> [gpu_id]
# where <data_dir> contains manifest.jsonl and the two batch scripts
# (flux_portraits.py, cosyvoice_tts.py).
set -euo pipefail

DATA_DIR="$(cd "$1" && pwd)"
GPU="${2:-1}"
HF_CACHE="${HOME}/.cache/huggingface"
MS_CACHE="${HOME}/.cache/modelscope"
mkdir -p "$HF_CACHE" "$MS_CACHE" "$DATA_DIR/portraits" "$DATA_DIR/audio"

echo "=== Flux portraits (GPU $GPU) ==="
docker run --rm --gpus "device=${GPU}" --shm-size 16G \
  --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
  --mount "type=bind,src=${DATA_DIR},dst=/work" --workdir /work \
  flux python flux_portraits.py manifest.jsonl portraits

echo "=== CosyVoice3 speech (GPU $GPU) ==="
docker run --rm --gpus "device=${GPU}" \
  --mount "type=bind,src=${HF_CACHE},dst=/root/.cache/huggingface" \
  --mount "type=bind,src=${MS_CACHE},dst=/root/.cache/modelscope" \
  --mount "type=bind,src=${DATA_DIR},dst=/work" --workdir /work \
  --env "PYTHONPATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS" \
  --env "HF_HOME=/root/.cache/huggingface" \
  --env "MODELSCOPE_CACHE=/root/.cache/modelscope" \
  cosyvoice3 python cosyvoice_tts.py manifest.jsonl audio

echo "=== Merge asset paths into manifest_assets.jsonl ==="
python3 - "$DATA_DIR" <<'PYEOF'
import json, os, sys
d = sys.argv[1]
out = []
for line in open(os.path.join(d, "manifest.jsonl")):
    it = json.loads(line)
    img = os.path.join("portraits", f"{it['id']}.png")
    aud = os.path.join("audio", f"{it['id']}.wav")
    it["image_path"] = img if os.path.exists(os.path.join(d, img)) else None
    it["audio_path"] = aud if os.path.exists(os.path.join(d, aud)) else None
    out.append(it)
with open(os.path.join(d, "manifest_assets.jsonl"), "w") as f:
    for it in out:
        f.write(json.dumps(it) + "\n")
ok = sum(1 for it in out if it["image_path"] and it["audio_path"])
print(f"  {ok}/{len(out)} items have both portrait + audio")
PYEOF

echo "Done. See ${DATA_DIR}/manifest_assets.jsonl"
