#!/usr/bin/env python3
"""
LingBot-Video: MoE Video Foundation Model for Embodied Intelligence

Open-source Mixture-of-Experts video generation model from Robbyant. Supports
text-to-video (T2V) and text+image-to-video (TI2V) with dense 1.3B and MoE
30B-A3B variants. Trained on 70k+ hours of embodied + web video data with
physical-rationality and task-completion reward alignment.

License: Apache-2.0
Models: https://huggingface.co/robbyant/lingbot-video-dense-1.3b
        https://huggingface.co/robbyant/lingbot-video-moe-30b-a3b
Paper:  https://huggingface.co/papers/2607.07675
"""

import os
import sys
import logging
import warnings

# Suppress verbose logging by default
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.ERROR)
for logger_name in ["transformers", "diffusers", "torch", "accelerate", "safetensors"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import argparse
from pathlib import Path

import torch

# CVL dual-mode execution support
try:
    from cvlization.paths import (
        get_input_dir,
        get_output_dir,
        resolve_input_path,
        resolve_output_path,
    )
    CVL_AVAILABLE = True
except ImportError:
    CVL_AVAILABLE = False

    def get_input_dir():
        return os.getenv("CVL_INPUTS", os.getcwd())

    def get_output_dir():
        output_dir = os.getenv("CVL_OUTPUTS", "./outputs")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def resolve_input_path(path):
        if path.startswith(("http://", "https://")) or path.startswith("/"):
            return path
        base = get_input_dir()
        return os.path.join(base, path) if os.getenv("CVL_INPUTS") else path

    def resolve_output_path(path):
        if path is None:
            path = "result.txt"
        if path.startswith("/"):
            return path
        base = get_output_dir()
        return os.path.join(base, path)


# Model repos on HuggingFace
DENSE_REPO = "robbyant/lingbot-video-dense-1.3b"
MOE_REPO = "robbyant/lingbot-video-moe-30b-a3b"


def get_default_negative_prompt(mode: str = "t2v") -> str:
    """Load the default negative prompt from the lingbot_video package."""
    import json
    import importlib.resources

    try:
        if mode == "t2i":
            fname = "default_negative_prompt_image.json"
        else:
            fname = "default_negative_prompt.json"
        ref = importlib.resources.files("lingbot_video").joinpath(fname)
        data = json.loads(ref.read_text(encoding="utf-8"))
        # Extract the universal_negative text
        if isinstance(data, dict) and "universal_negative" in data:
            neg = data["universal_negative"]
            if isinstance(neg, dict):
                parts = []
                for section in neg.values():
                    if isinstance(section, list):
                        parts.extend(section)
                    elif isinstance(section, str):
                        parts.append(section)
                return ", ".join(parts)
            return str(neg)
        return json.dumps(data)
    except Exception:
        return ""


def load_pipeline(model_id: str, dtype: torch.dtype, device: str, mode: str = "t2v"):
    """Load the LingBot-Video pipeline from HuggingFace.

    Downloads the model snapshot, then loads the pipeline using the
    lingbot_video package classes directly (the model_index.json references
    lingbot_video.* modules that must be installed as a Python package).

    The VAE is cast to fp32 (upstream default) while the transformer and text
    encoder remain in bf16 for memory efficiency.
    """
    from huggingface_hub import snapshot_download
    from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel
    from lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline

    print(f"Loading model: {model_id}", flush=True)
    print(f"  dtype: {dtype}", flush=True)
    print(f"  device: {device}", flush=True)

    # Download model snapshot to HF cache (cached after first run)
    print("Downloading model snapshot (cached after first run)...", flush=True)
    model_dir = snapshot_download(repo_id=model_id)
    print(f"Model directory: {model_dir}", flush=True)

    # Load transformer with the specified dtype
    print("Loading transformer...", flush=True)
    transformer = LingBotVideoTransformer3DModel.from_pretrained(
        model_dir,
        subfolder="transformer",
        torch_dtype=dtype,
    )

    # Select pipeline class based on mode
    if mode == "ti2v":
        from lingbot_video.pipeline_lingbot_video_i2v import LingBotVideoImageToVideoPipeline
        pipeline_cls = LingBotVideoImageToVideoPipeline
    else:
        pipeline_cls = LingBotVideoPipeline

    # Load full pipeline from local snapshot
    # Use bf16 for transformer/text_encoder, but VAE will be cast to fp32 after
    print("Loading pipeline...", flush=True)
    pipe = pipeline_cls.from_pretrained(
        model_dir,
        transformer=transformer,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    pipe = pipe.to(device)

    # Cast VAE to fp32 (upstream default for quality)
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae = pipe.vae.to(dtype=torch.float32)

    print("Pipeline loaded successfully.", flush=True)
    return pipe


def generate_video(
    pipe,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    shift: float,
    seed: int,
    image_path: str = None,
):
    """Run video generation and return frames as numpy array."""
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    call_kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        shift=shift,
        generator=generator,
        output_type="np",
    )

    # Load conditioning image for TI2V mode
    if image_path:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        call_kwargs["image"] = image

    print(
        f"Generating: {width}x{height}, {num_frames} frames, "
        f"{num_inference_steps} steps, guidance={guidance_scale}, shift={shift}",
        flush=True,
    )

    with torch.inference_mode():
        result = pipe(**call_kwargs)

    return result


def save_video(frames, output_path: str, fps: float):
    """Save generated frames as MP4 video."""
    from diffusers.utils import export_to_video

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    export_to_video(frames, output_path, fps=fps)
    print(f"Video saved: {output_path}", flush=True)


def save_image(frames, output_path: str):
    """Save a single generated frame as PNG image."""
    from PIL import Image
    import numpy as np

    frame = frames[0] if hasattr(frames, '__len__') else frames
    if isinstance(frame, np.ndarray):
        if frame.max() <= 1.0:
            frame = (frame * 255).clip(0, 255).astype("uint8")
        img = Image.fromarray(frame)
    else:
        img = frame

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    img.save(output_path)
    print(f"Image saved: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="LingBot-Video: MoE Video Foundation Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["dense-1.3b", "moe-30b-a3b"],
        default="dense-1.3b",
        help="Model variant: dense-1.3b (single GPU, ~8GB VRAM) or "
             "moe-30b-a3b (~80GB VRAM at 81 frames 832x480, ~3B active params)",
    )

    # Input/output
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "A robotic arm smoothly picks up a red cube from a table and "
            "places it on a shelf, filmed from a fixed overhead camera angle"
        ),
        help="Text prompt describing the video to generate",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt for quality guidance",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional conditioning image for text+image-to-video (TI2V) mode",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Output video file path (use .png for T2I mode)",
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["t2v", "t2i"],
        default="t2v",
        help="Generation mode: t2v (text-to-video) or t2i (text-to-image)",
    )

    # Video parameters
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height (must be multiple of 16)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width (must be multiple of 16)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=81,
        help="Number of frames (must be 1 for t2i, or 4n+1 for t2v, e.g. 21, 41, 61, 81)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Output video frame rate",
    )

    # Sampling parameters
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=3.0,
        help="Flow matching timestep shift (upstream default 3.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Debug
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Enable verbose logging if requested
    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)
        for name in ["transformers", "diffusers", "torch", "accelerate"]:
            logging.getLogger(name).setLevel(logging.INFO)

    # Select model repo
    model_id = DENSE_REPO if args.model == "dense-1.3b" else MOE_REPO

    # Resolve output path
    output_path = resolve_output_path(args.output)

    # Resolve image path if provided
    image_path = None
    if args.image:
        image_path = resolve_input_path(args.image)
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}", flush=True)
            sys.exit(1)

    # Adjust num_frames for t2i mode
    num_frames = args.num_frames
    if args.mode == "t2i":
        num_frames = 1
        if not output_path.endswith(".png"):
            output_path = output_path.rsplit(".", 1)[0] + ".png"

    print(f"Model: {args.model} ({model_id})", flush=True)
    print(f"Mode: {args.mode}", flush=True)
    print(f"Output: {output_path}", flush=True)

    # Load pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # Select pipeline mode: use TI2V if image is provided
    pipeline_mode = args.mode
    if image_path:
        pipeline_mode = "ti2v"

    pipe = load_pipeline(model_id, dtype=dtype, device=device, mode=pipeline_mode)

    # Use default negative prompt from package if none specified
    negative_prompt = args.negative_prompt
    if not negative_prompt:
        negative_prompt = get_default_negative_prompt(args.mode)

    # Generate
    result = generate_video(
        pipe=pipe,
        prompt=args.prompt,
        negative_prompt=negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        shift=args.shift,
        seed=args.seed,
        image_path=image_path,
    )

    # Save output
    frames = result.frames[0] if hasattr(result, "frames") else result[0]

    if args.mode == "t2i":
        save_image(frames, output_path)
    else:
        save_video(frames, output_path, fps=args.fps)

    print(f"\nDone! Output saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
