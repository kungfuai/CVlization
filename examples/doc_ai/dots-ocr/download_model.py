#!/usr/bin/env python3
"""
Download dots.ocr model from HuggingFace using snapshot_download.
This works around the issue with periods in the model name.
"""

from huggingface_hub import snapshot_download
import os

def main():
    model_id = "rednote-hilab/dots.ocr"
    cache_dir = os.path.expanduser("~/.cache/huggingface")

    print(f"Downloading model snapshot from {model_id}...")
    print("This may take a few minutes...")

    # Download entire repository snapshot
    snapshot_path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_files_only=False
    )

    print(f"\nModel downloaded successfully!")
    print(f"Snapshot path: {snapshot_path}")
    print(f"\nYou can now use this path with predict.py:")
    print(f"  python3 predict.py --model-path '{snapshot_path}' --image examples/sample.jpg")

if __name__ == "__main__":
    main()
