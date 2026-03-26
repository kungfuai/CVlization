#!/usr/bin/env python3
import argparse
from dataclasses import dataclass

from huggingface_hub import HfApi


@dataclass
class RepoCheck:
    repo_id: str
    repo_type: str
    required_files: list[str]


CHECKS = [
    RepoCheck(
        repo_id="zzsi/swm-dmc-cheetah",
        repo_type="model",
        required_files=[
            "README.md",
            "config.yaml",
            "lejepa_epoch_50_object.ckpt",
            "lejepa_weights.ckpt",
        ],
    ),
    RepoCheck(
        repo_id="zzsi/swm-dmc-expert-policies",
        repo_type="model",
        required_files=[
            "README.md",
            "cheetah/expert_policy/expert_policy.zip",
            "cheetah/expert_policy/vec_normalize.pkl",
        ],
    ),
    RepoCheck(
        repo_id="zzsi/swm-dmc-expert",
        repo_type="dataset",
        required_files=["README.md", "dmc_expert.tar.zst"],
    ),
    RepoCheck(
        repo_id="zzsi/swm-dmc-mixed-small",
        repo_type="dataset",
        required_files=["README.md", "dmc_mixed-small.tar.zst"],
    ),
    RepoCheck(
        repo_id="zzsi/swm-dmc-mixed-large",
        repo_type="dataset",
        required_files=["README.md", "dmc_mixed-large.tar.zst"],
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify remote HF repos contain expected files")
    parser.add_argument("--strict", action="store_true", help="Return non-zero on first mismatch")
    args = parser.parse_args()

    api = HfApi()
    failures = []
    for check in CHECKS:
        files = set(api.list_repo_files(check.repo_id, repo_type=check.repo_type))
        missing = [f for f in check.required_files if f not in files]
        if missing:
            failures.append((check.repo_id, missing))
            print(f"[warn] {check.repo_id} missing: {missing}")
        else:
            print(f"[ok] {check.repo_id}")

    if failures and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
