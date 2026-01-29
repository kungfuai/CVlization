#!/usr/bin/env python
"""
Upload model checkpoints to Hugging Face Hub.

Usage:
    # Set HF_TOKEN in .env or environment
    python upload_to_hf.py

    # Or with token directly
    HF_TOKEN=hf_xxx python upload_to_hf.py
"""
import os
import argparse
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from huggingface_hub import HfApi
import torch


def strip_checkpoint(ckpt_path: Path, model_type: str) -> dict:
    """Load checkpoint and strip optimizer state for smaller file."""
    ckpt = torch.load(ckpt_path, map_location="cpu")

    return {
        "model_state_dict": ckpt["model_state_dict"],
        "step": ckpt.get("step"),
        "config": {
            "model": model_type,
            "channels": 64,
            "mask_guidance": "modulation",
            "num_frames": 4,
            "multi_scale": [256, 320, 384],
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Upload models to Hugging Face")
    parser.add_argument("--repo", default="zzsi/cvl_models", help="HF repo id")
    parser.add_argument("--strip", action="store_true", help="Strip optimizer state (smaller files)")
    parser.add_argument("--checkpoints-dir", default="./checkpoints", help="Checkpoints directory")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found in environment. Set it in .env or export HF_TOKEN=...")

    api = HfApi(token=token)

    # Create repo if needed
    try:
        api.create_repo(args.repo, repo_type="model", exist_ok=True)
        print(f"Repo ready: {args.repo}")
    except Exception as e:
        print(f"Repo check: {e}")

    # Models to upload
    models = [
        {
            "name": "nafunet",
            "type": "temporal_nafunet",
            "checkpoint": "vimeo_exp25_nafunet_resume/best.pt",
        },
        {
            "name": "composite",
            "type": "composite",
            "checkpoint": "vimeo_exp27_composite_resume/best.pt",
        },
    ]

    for model in models:
        local_path = Path(args.checkpoints_dir) / model["checkpoint"]

        if not local_path.exists():
            print(f"Skipping {model['name']}: {local_path} not found")
            continue

        repo_path = f"video_enhancement/{model['name']}.pt"

        if args.strip:
            print(f"Stripping and uploading {model['name']}...")
            stripped = strip_checkpoint(local_path, model["type"])

            # Save to temp file
            tmp_path = Path(f"/tmp/{model['name']}_stripped.pt")
            torch.save(stripped, tmp_path)
            upload_path = tmp_path
        else:
            print(f"Uploading {model['name']} (full checkpoint)...")
            upload_path = local_path

        api.upload_file(
            path_or_fileobj=str(upload_path),
            path_in_repo=repo_path,
            repo_id=args.repo,
        )
        print(f"  Uploaded: {repo_path}")

    print(f"\nDone! View at: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
