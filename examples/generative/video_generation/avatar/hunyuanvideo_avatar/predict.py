#!/usr/bin/env python3
"""
HunyuanVideo-Avatar inference wrapper for CVlization.

Generates audio-driven avatar video from a reference image and audio file.
"""

import argparse
import csv
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from cvlization.paths import resolve_input_path, resolve_output_path


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)


def ensure_weights(weights_dir: Path, skip_download: bool, use_fp8: bool) -> Path:
    checkpoint_name = "mp_rank_00_model_states_fp8.pt" if use_fp8 else "mp_rank_00_model_states.pt"
    checkpoint_path = (
        weights_dir
        / "ckpts"
        / "hunyuan-video-t2v-720p"
        / "transformers"
        / checkpoint_name
    )
    if checkpoint_path.exists():
        return checkpoint_path

    if skip_download:
        raise FileNotFoundError(
            "Model weights not found and --skip-download was set. "
            f"Expected: {checkpoint_path}"
        )

    logging.info("Downloading HunyuanVideo-Avatar weights (large download)...")
    from huggingface_hub import snapshot_download

    weights_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="tencent/HunyuanVideo-Avatar",
        local_dir=str(weights_dir),
        local_dir_use_symlinks=False,
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model weights downloaded but checkpoint missing: {checkpoint_path}")
    return checkpoint_path


def get_example_dir() -> Path:
    return Path(os.environ.get("CVL_EXAMPLE_DIR", os.path.dirname(os.path.abspath(__file__))))


def download_sample_inputs_if_needed(image_path: str, audio_path: str) -> None:
    from huggingface_hub import hf_hub_download

    example_dir = get_example_dir()
    test_inputs_dir = example_dir / "test_inputs"
    default_image = test_inputs_dir / "1.png"
    default_audio = test_inputs_dir / "2.WAV"

    needs_download = False
    if image_path == str(default_image) and not default_image.exists():
        needs_download = True
    if audio_path == str(default_audio) and not default_audio.exists():
        needs_download = True

    if not needs_download:
        return

    logging.info("Downloading sample inputs from HuggingFace...")
    test_inputs_dir.mkdir(parents=True, exist_ok=True)

    if not default_image.exists():
        hf_hub_download(
            repo_id="zzsi/cvl",
            filename="hunyuan_avatar/1.png",
            repo_type="dataset",
            local_dir=str(test_inputs_dir),
        )
        src = test_inputs_dir / "hunyuan_avatar" / "1.png"
        if src.exists():
            src.rename(default_image)

    if not default_audio.exists():
        hf_hub_download(
            repo_id="zzsi/cvl",
            filename="hunyuan_avatar/2.WAV",
            repo_type="dataset",
            local_dir=str(test_inputs_dir),
        )
        src = test_inputs_dir / "hunyuan_avatar" / "2.WAV"
        if src.exists():
            src.rename(default_audio)

    subdir = test_inputs_dir / "hunyuan_avatar"
    if subdir.exists():
        shutil.rmtree(subdir, ignore_errors=True)


def write_input_csv(csv_path: Path, image_path: str, audio_path: str, prompt: str, fps: int) -> str:
    video_id = "result"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["videoid", "image", "audio", "prompt", "fps"])
        writer.writerow([video_id, image_path, audio_path, prompt, str(fps)])
    return video_id


def run_inference(args, checkpoint_path: Path, output_dir: Path) -> Path:
    csv_path = Path("/workspace/local/inputs.csv")
    output_dir.mkdir(parents=True, exist_ok=True)

    video_id = write_input_csv(csv_path, args.image, args.audio, args.prompt, args.fps)

    cmd = [
        sys.executable,
        "/workspace/HunyuanVideo-Avatar/hymm_sp/sample_gpu_poor.py",
        "--input",
        str(csv_path),
        "--ckpt",
        str(checkpoint_path),
        "--sample-n-frames",
        str(args.sample_frames),
        "--seed",
        str(args.seed),
        "--image-size",
        str(args.image_size),
        "--cfg-scale",
        str(args.cfg_scale),
        "--use-deepcache",
        str(args.use_deepcache),
        "--flow-shift-eval-video",
        str(args.flow_shift),
        "--save-path",
        str(output_dir),
    ]

    if args.infer_steps is not None:
        cmd.extend(["--infer-steps", str(args.infer_steps)])

    if args.fp8:
        cmd.append("--use-fp8")
    if args.cpu_offload:
        cmd.append("--cpu-offload")
    if args.infer_min:
        cmd.append("--infer-min")

    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace/HunyuanVideo-Avatar"

    logging.info("Running inference...")
    subprocess.run(cmd, check=True, cwd="/workspace/HunyuanVideo-Avatar", env=env)

    result_path = output_dir / f"{video_id}_audio.mp4"
    if not result_path.exists():
        raise FileNotFoundError(f"Expected output not found: {result_path}")
    return result_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate avatar video with HunyuanVideo-Avatar")
    parser.add_argument("--image", type=str, default=None, help="Path to reference image (PNG/JPG)")
    parser.add_argument("--audio", type=str, default=None, help="Path to driving audio (WAV)")
    parser.add_argument("--output", type=str, default="outputs/output.mp4", help="Output video path")
    parser.add_argument("--prompt", type=str, default="A person is speaking.", help="Text prompt")
    parser.add_argument("--fps", type=int, default=25, help="Output FPS")
    parser.add_argument("--image-size", type=int, default=704, help="Target image size (square)")
    parser.add_argument("--sample-frames", type=int, default=129, help="Number of frames to sample")
    parser.add_argument("--infer-steps", type=int, default=None, help="Override denoising steps")
    parser.add_argument("--cfg-scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--flow-shift", type=float, default=5.0, help="Flow shift for video inference")
    parser.add_argument("--seed", type=int, default=128, help="Random seed")
    parser.add_argument("--use-deepcache", type=int, default=1, help="Enable DeepCache (1/0)")
    parser.add_argument("--no-fp8", dest="fp8", action="store_false", help="Disable FP8 checkpoint")
    parser.add_argument("--cpu-offload", action="store_true", help="Enable CPU offload (lower VRAM)")
    parser.add_argument("--infer-min", action="store_true", help="Force short (~5s) inference")
    parser.add_argument("--skip-download", action="store_true", help="Skip model download check")
    return parser.parse_args()


def main():
    args = parse_args()

    example_dir = get_example_dir()
    default_image = str(example_dir / "test_inputs" / "1.png")
    default_audio = str(example_dir / "test_inputs" / "2.WAV")

    if args.image is None:
        args.image = default_image
    else:
        args.image = resolve_input_path(args.image)

    if args.audio is None:
        args.audio = default_audio
    else:
        args.audio = resolve_input_path(args.audio)

    download_sample_inputs_if_needed(args.image, args.audio)
    output_path = Path(resolve_output_path(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    weights_dir = Path(os.environ.get("MODEL_BASE", "/workspace/HunyuanVideo-Avatar/weights"))
    checkpoint_path = ensure_weights(weights_dir, args.skip_download, args.fp8)

    output_dir = Path("/workspace/local/outputs")
    result_path = run_inference(args, checkpoint_path, output_dir)

    if result_path.resolve() != output_path.resolve():
        shutil.move(str(result_path), str(output_path))
    logging.info("Output saved to %s", output_path)


if __name__ == "__main__":
    main()
