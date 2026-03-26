#!/usr/bin/env python3
import argparse
import subprocess
import zipfile
from pathlib import Path


EXPECTED_MODEL_FILES = [
    "config.yaml",
    "lejepa_epoch_50_object.ckpt",
    "lejepa_weights.ckpt",
]

EXPECTED_ENVS = [
    "ball_in_cup",
    "cartpole",
    "cheetah",
    "finger",
    "hopper",
    "pendulum",
    "quadruped",
    "reacher",
    "walker",
]

SPLIT_ARCHIVES = {
    "expert": "dmc_expert.tar.zst",
    "mixed-small": "dmc_mixed-small.tar.zst",
    "mixed-large": "dmc_mixed-large.tar.zst",
}


def assert_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")


def inspect_tar_prefix(path: Path, expected_prefix: str) -> None:
    proc = subprocess.run(
        ["tar", "--zstd", "-tf", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"Archive is empty: {path}")
    bad = [ln for ln in lines if not ln.startswith(expected_prefix)]
    if bad:
        raise RuntimeError(
            f"Archive {path} contains unexpected path(s), first: {bad[0]}"
        )
    print(f"[ok] {path.name}: {len(lines)} entries under prefix '{expected_prefix}'")


def inspect_policy_zip(path: Path) -> None:
    required = {
        "data",
        "policy.pth",
        "actor.optimizer.pth",
        "critic.optimizer.pth",
        "pytorch_variables.pth",
        "_stable_baselines3_version",
        "system_info.txt",
    }
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"{path} missing zip members: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect downloaded stable-worldmodel assets")
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
        help="Asset directory created by download_assets.py",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["expert", "mixed-small", "mixed-large"],
        default=["expert"],
        help="Dataset splits to inspect",
    )
    args = parser.parse_args()

    asset_dir = args.asset_dir
    model_dir = asset_dir / "models" / "swm-dmc-cheetah"
    policy_dir = asset_dir / "models" / "swm-dmc-expert-policies"
    datasets_dir = asset_dir / "datasets"

    print("Checking world-model files...")
    for name in EXPECTED_MODEL_FILES:
        assert_exists(model_dir / name)
    print("[ok] world-model artifacts present")

    print("Checking expert-policy files...")
    for env in EXPECTED_ENVS:
        base = policy_dir / env / "expert_policy"
        zip_path = base / "expert_policy.zip"
        norm_path = base / "vec_normalize.pkl"
        assert_exists(zip_path)
        assert_exists(norm_path)
        inspect_policy_zip(zip_path)
    print(f"[ok] expert policies verified for {len(EXPECTED_ENVS)} envs")

    print("Checking dataset split archives...")
    for split in args.splits:
        archive = datasets_dir / split / SPLIT_ARCHIVES[split]
        assert_exists(archive)
        inspect_tar_prefix(archive, f"dmc/{split}/")

    print("All requested checks passed.")


if __name__ == "__main__":
    main()
