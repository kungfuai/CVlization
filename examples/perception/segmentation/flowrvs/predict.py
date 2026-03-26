#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, snapshot_download

try:
    from cvlization.paths import get_input_dir, get_output_dir, resolve_input_path, resolve_output_path
except ImportError:
    def get_input_dir() -> str:
        return os.getcwd()

    def get_output_dir() -> str:
        out = os.path.join(os.getcwd(), "outputs")
        os.makedirs(out, exist_ok=True)
        return out

    def resolve_input_path(path: str, base_dir: str) -> str:
        return path if os.path.isabs(path) else os.path.join(base_dir, path)

    def resolve_output_path(path: str, base_dir: str) -> str:
        return path if os.path.isabs(path) else os.path.join(base_dir, path)


def ensure_weights(repo_id: str, cache_dir: str, token: Optional[str]) -> tuple[str, str]:
    common = dict(repo_id=repo_id, repo_type="model", token=token, local_dir=cache_dir)
    dit_ckpt = hf_hub_download(filename="FlowRVS_dit_mevis.pth", **common)
    vae_ckpt = hf_hub_download(filename="tuned_vae.pth", **common)
    return dit_ckpt, vae_ckpt


SAMPLES = {
    "sample": ("flowrvs/sample_ultraman.mp4", ["the Ultraman", "the devil cat"]),
    "basketball": ("flowrvs/sample_basketball.mp4", ["the man wearing colorful shoes shoots the ball", "the man who is defending", "basketball"]),
}


def ensure_sample_input(name: str, token: Optional[str]) -> tuple[Path, list[str]]:
    filename, default_prompts = SAMPLES[name]
    path = hf_hub_download(
        repo_id="zzsi/cvl",
        repo_type="dataset",
        filename=filename,
        token=token,
        local_dir="/root/.cache/huggingface/cvl",
    )
    return Path(path).resolve(), default_prompts


def ensure_wan_base_model(token: Optional[str]) -> str:
    return snapshot_download(
        repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        repo_type="model",
        token=token,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FlowRVS wrapper for referring video segmentation inference."
    )
    parser.add_argument("--input", default="basketball", help=f'Input video path or one of: {list(SAMPLES)}.')
    parser.add_argument("--prompts", nargs="+", default=None, help="Referring text prompts (defaults to sample-specific prompts).")
    parser.add_argument("--output", default="outputs/flowrvs_result.mp4", help="Output overlay video path.")
    parser.add_argument("--fps", type=int, default=12, help="FPS for video decoding.")
    parser.add_argument("--height", type=int, default=480, help="Inference height.")
    parser.add_argument("--width", type=int, default=832, help="Inference width.")
    parser.add_argument("--repo-dir", default="/workspace", help="Path to FlowRVS repository inside container.")
    parser.add_argument("--dit-ckpt", default="", help="Optional local DiT checkpoint path.")
    parser.add_argument("--vae-ckpt", default="", help="Optional local VAE checkpoint path.")
    parser.add_argument("--weights-repo", default="xmz111/FlowRVS", help="HF model repo for FlowRVS checkpoints.")
    parser.add_argument("--weights-cache", default="/root/.cache/huggingface/cvl/flowrvs", help="Cache dir for FlowRVS checkpoints.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve paths/checkpoints then exit without running inference.")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    repo_dir = Path(args.repo_dir).resolve()

    if not repo_dir.exists():
        raise FileNotFoundError(f"FlowRVS repo not found: {repo_dir}")

    if args.input in SAMPLES:
        input_path, default_prompts = ensure_sample_input(args.input, hf_token)
        prompts = args.prompts or default_prompts
    else:
        input_path = Path(resolve_input_path(args.input, get_input_dir())).resolve()
        prompts = args.prompts or ["the subject"]
    output_path = Path(resolve_output_path(args.output, get_output_dir())).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cache_dir = Path(args.weights_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    wan_model_path = ensure_wan_base_model(hf_token)

    dit_ckpt = args.dit_ckpt
    vae_ckpt = args.vae_ckpt
    if not dit_ckpt or not vae_ckpt:
        dit_ckpt, vae_ckpt = ensure_weights(args.weights_repo, str(cache_dir), hf_token)

    if args.dry_run:
        print(f"Input:          {input_path}")
        print(f"Output:         {output_path}")
        print(f"Repo:           {repo_dir}")
        print(f"Wan model:      {wan_model_path}")
        print(f"DiT checkpoint: {dit_ckpt}")
        print(f"VAE checkpoint: {vae_ckpt}")
        return

    tmp_output_dir = output_path.parent / "_flowrvs_tmp"
    tmp_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(repo_dir / "inference_demo.py"),
        "--input_path", str(input_path),
        "--text_prompts", *prompts,
        "--fps", str(args.fps),
        "--reso_h", str(args.height),
        "--reso_w", str(args.width),
        "--output_dir", str(tmp_output_dir),
        "--dit_ckpt", str(dit_ckpt),
        "--vae_ckpt", str(vae_ckpt),
        "--wan_model_path", wan_model_path,
    ]

    print("Running FlowRVS inference:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(repo_dir), check=True)

    produced = tmp_output_dir / f"{input_path.stem}_result.mp4"
    if not produced.exists():
        raise FileNotFoundError(f"Expected output not produced: {produced}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, output_path)
    print(f"Saved output video: {output_path}")


if __name__ == "__main__":
    main()
