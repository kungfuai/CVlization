#!/usr/bin/env bash
set -euo pipefail

echo "Downloading LiteAvatar model files..."

python - <<'PY'
import pathlib
import requests

BASE_URL = "https://modelscope.cn/api/v1/models/HumanAIGC-Engineering/LiteAvatarGallery/repo?Revision=master&FilePath="
ROOT = pathlib.Path(__file__).resolve().parent
targets = {
    "lite_avatar_weights/model_1.onnx": ROOT / "weights/model_1.onnx",
    "lite_avatar_weights/model.pb": ROOT
    / "weights/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/model.pb",
    "lite_avatar_weights/lm.pb": ROOT
    / "weights/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/lm/lm.pb",
}

for remote, dest in targets.items():
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Skipping {dest} (already exists)")
        continue
    url = f"{BASE_URL}{remote}"
    print(f"Fetching {url}")
    with requests.get(url, stream=True) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
print("All model files downloaded successfully!")
PY
