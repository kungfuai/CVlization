#!/usr/bin/env bash
set -euo pipefail

# Optional helper script to prefetch weights into the shared Hugging Face cache.
# The inference pipeline now downloads weights lazily, so running this script is
# no longer required—it's just a convenience for fully offline runs.

python - <<'PY'
from huggingface_hub import hf_hub_download
import os

assets = [
    ("yzd-v/DWPose", "yolox_l.onnx"),
    ("yzd-v/DWPose", "dw-ll_ucoco_384.onnx"),
    ("tencent/MimicMotion", "MimicMotion_1-1.pth"),
]

cache_dir = os.environ.get("HF_HOME")

for repo_id, filename in assets:
    path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    print(f"Cached {repo_id}/{filename} -> {path}")
PY
