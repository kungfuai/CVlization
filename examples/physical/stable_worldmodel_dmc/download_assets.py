#!/usr/bin/env python3
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


WORLD_MODEL_REPO = "zzsi/swm-dmc-cheetah"
EXPERT_POLICIES_REPO = "zzsi/swm-dmc-expert-policies"
DATASET_REPOS = {
    "expert": ("zzsi/swm-dmc-expert", "dmc_expert.tar.zst"),
    "mixed-small": ("zzsi/swm-dmc-mixed-small", "dmc_mixed-small.tar.zst"),
    "mixed-large": ("zzsi/swm-dmc-mixed-large", "dmc_mixed-large.tar.zst"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download stable-worldmodel DMControl assets from Hugging Face")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
        help="Local directory to store downloaded assets",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["expert", "mixed-small", "mixed-large"],
        default=["expert"],
        help="Dataset splits to download",
    )
    args = parser.parse_args()

    target = args.target_dir
    target.mkdir(parents=True, exist_ok=True)

    model_dir = target / "models" / "swm-dmc-cheetah"
    policy_dir = target / "models" / "swm-dmc-expert-policies"
    data_dir = target / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading world-model repo: {WORLD_MODEL_REPO}")
    snapshot_download(
        repo_id=WORLD_MODEL_REPO,
        repo_type="model",
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )

    print(f"Downloading expert-policy repo: {EXPERT_POLICIES_REPO}")
    snapshot_download(
        repo_id=EXPERT_POLICIES_REPO,
        repo_type="model",
        local_dir=str(policy_dir),
        local_dir_use_symlinks=False,
    )

    for split in args.splits:
        repo_id, filename = DATASET_REPOS[split]
        split_dir = data_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset split '{split}' from {repo_id}/{filename}")
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(split_dir),
            local_dir_use_symlinks=False,
        )
        print(f"Saved: {path}")

    print("Download complete.")


if __name__ == "__main__":
    main()
