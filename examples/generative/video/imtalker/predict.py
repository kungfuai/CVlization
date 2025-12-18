#!/usr/bin/env python3
"""
IMTalker inference wrapper for CVlization.

Generates talking head videos from an image and audio file using
implicit motion transfer with flow matching.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add IMTalker and generator to path (generator has relative imports)
sys.path.insert(0, "/workspace/IMTalker")
sys.path.insert(0, "/workspace/IMTalker/generator")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)


def download_models_if_needed():
    """Download model checkpoints if not present."""
    from huggingface_hub import hf_hub_download

    checkpoints_dir = Path("/workspace/IMTalker/checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Download generator checkpoint
    generator_path = checkpoints_dir / "generator.ckpt"
    if not generator_path.exists() or generator_path.stat().st_size < 1024:
        logging.info("Downloading generator.ckpt from cbsjtu01/IMTalker...")
        hf_hub_download(
            repo_id="cbsjtu01/IMTalker",
            filename="generator.ckpt",
            local_dir=str(checkpoints_dir),
            local_dir_use_symlinks=False,
        )

    # Download renderer checkpoint
    renderer_path = checkpoints_dir / "renderer.ckpt"
    if not renderer_path.exists() or renderer_path.stat().st_size < 1024:
        logging.info("Downloading renderer.ckpt from cbsjtu01/IMTalker...")
        hf_hub_download(
            repo_id="cbsjtu01/IMTalker",
            filename="renderer.ckpt",
            local_dir=str(checkpoints_dir),
            local_dir_use_symlinks=False,
        )

    # Download config
    config_path = checkpoints_dir / "config.yaml"
    if not config_path.exists() or config_path.stat().st_size < 100:
        logging.info("Downloading config.yaml from cbsjtu01/IMTalker...")
        hf_hub_download(
            repo_id="cbsjtu01/IMTalker",
            filename="config.yaml",
            local_dir=str(checkpoints_dir),
            local_dir_use_symlinks=False,
        )

    # Download wav2vec2 model (from Facebook)
    wav2vec_dir = checkpoints_dir / "wav2vec2-base-960h"
    if not wav2vec_dir.exists() or not any(wav2vec_dir.glob("*.bin")):
        logging.info("Downloading wav2vec2-base-960h from facebook...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="facebook/wav2vec2-base-960h",
            local_dir=str(wav2vec_dir),
            local_dir_use_symlinks=False,
        )

    logging.info("All models ready.")


def run_inference(
    image_path: str,
    audio_path: str,
    output_path: str,
    crop: bool = True,
    a_cfg_scale: float = 3.0,
    nfe: int = 10,
    seed: int = 42,
):
    """Run IMTalker audio-driven inference."""
    import torch

    # Enable CUDA optimizations
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    logging.info(f"Running IMTalker inference...")
    logging.info(f"  Image: {image_path}")
    logging.info(f"  Audio: {audio_path}")
    logging.info(f"  Output: {output_path}")

    # Import IMTalker classes directly
    from generator.generate import InferenceAgent, InferenceOptions

    # Build args list for InferenceOptions parser
    args_list = [
        "--ref_path", image_path,
        "--aud_path", audio_path,
        "--res_dir", os.path.dirname(output_path) or ".",
        "--generator_path", "/workspace/IMTalker/checkpoints/generator.ckpt",
        "--renderer_path", "/workspace/IMTalker/checkpoints/renderer.ckpt",
        "--wav2vec_model_path", "/workspace/IMTalker/checkpoints/wav2vec2-base-960h",
        "--a_cfg_scale", str(a_cfg_scale),
        "--nfe", str(nfe),
        "--seed", str(seed),
    ]
    if crop:
        args_list.append("--crop")

    # Parse options using IMTalker's parser
    sys.argv = ["generate.py"] + args_list
    opt = InferenceOptions().parse()
    opt.rank, opt.ngpus = 0, 1

    # Create inference agent and run
    agent = InferenceAgent(opt)
    os.makedirs(opt.res_dir, exist_ok=True)

    # Generate output name based on input names
    image_stem = Path(image_path).stem
    audio_stem = Path(audio_path).stem
    generated_name = f"{image_stem}_{audio_stem}.mp4"
    generated_path = os.path.join(opt.res_dir, generated_name)

    logging.info(f"Processing: {image_stem}")
    agent.run_inference(
        generated_path, image_path, audio_path,
        pose_path=None, gaze_path=None,
        a_cfg_scale=a_cfg_scale, nfe=nfe, crop=crop, seed=seed
    )

    # Rename to expected output path if needed
    if os.path.exists(generated_path) and generated_path != output_path:
        os.rename(generated_path, output_path)
        logging.info(f"Renamed output to: {output_path}")
    elif not os.path.exists(output_path):
        # List what was actually generated
        output_dir = os.path.dirname(output_path) or "."
        mp4_files = list(Path(output_dir).glob("*.mp4"))
        if mp4_files:
            logging.info(f"Generated files: {[f.name for f in mp4_files]}")
            # Use the most recent one
            latest = max(mp4_files, key=lambda f: f.stat().st_mtime)
            os.rename(str(latest), output_path)
            logging.info(f"Renamed {latest.name} to: {output_path}")

    logging.info(f"Video saved to: {output_path}")


def resolve_path(path: str) -> str:
    """Resolve path - prepend /workspace/local/ for relative paths if they exist there."""
    if os.path.isabs(path):
        return path
    # Check if exists relative to /workspace/local/
    local_path = os.path.join("/workspace/local", path)
    if os.path.exists(local_path):
        return local_path
    # Check if exists as-is
    if os.path.exists(path):
        return path
    # Default to local path
    return local_path


def main():
    parser = argparse.ArgumentParser(description="Generate talking head video with IMTalker")
    parser.add_argument("--image", type=str, default="examples/images/1p-0.png", help="Path to input image (face)")
    parser.add_argument("--audio", type=str, default="examples/audios/1p-0.wav", help="Path to input audio (WAV)")
    parser.add_argument("--output", type=str, default="/workspace/local/outputs/output.mp4", help="Output video path")
    parser.add_argument("--no-crop", action="store_true", help="Disable automatic face cropping")
    parser.add_argument("--cfg-scale", type=float, default=3.0, help="Audio classifier-free guidance scale")
    parser.add_argument("--nfe", type=int, default=10, help="Number of function evaluations for ODE solver")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-download", action="store_true", help="Skip model download check")

    args = parser.parse_args()

    # Resolve paths
    image_path = resolve_path(args.image)
    audio_path = resolve_path(args.audio)
    output_path = resolve_path(args.output)

    logging.info(f"Image: {image_path}")
    logging.info(f"Audio: {audio_path}")
    logging.info(f"Output: {output_path}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Ensure models are downloaded
    if not args.skip_download:
        download_models_if_needed()

    # Run inference
    run_inference(
        image_path=image_path,
        audio_path=audio_path,
        output_path=output_path,
        crop=not args.no_crop,
        a_cfg_scale=args.cfg_scale,
        nfe=args.nfe,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
