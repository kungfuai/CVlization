#!/usr/bin/env python3
"""
LTX-2: Audio-Video Generation Model

19B parameter DiT-based foundation model for text-to-video and image-to-video generation
with synchronized audio. Supports multiple pipeline modes (distilled for speed, two_stage
for quality).

License: Apache-2.0
Model: https://huggingface.co/Lightricks/LTX-2
"""

import os
import sys
import logging
import warnings

# Suppress verbose logging by default
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.ERROR)
for logger_name in ["transformers", "torch", "accelerate", "safetensors"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download

# LTX imports
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE, DEFAULT_NEGATIVE_PROMPT
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number

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
        return os.getcwd()

    def get_output_dir():
        return os.getcwd()

    def resolve_input_path(path):
        return path

    def resolve_output_path(path):
        return path


# Model paths on HuggingFace
LTX2_REPO = "Lightricks/LTX-2"
GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"

# Model file names
CHECKPOINT_DISTILLED_FP8 = "ltx-2-19b-distilled-fp8.safetensors"
CHECKPOINT_DISTILLED = "ltx-2-19b-distilled.safetensors"
CHECKPOINT_DEV_FP8 = "ltx-2-19b-dev-fp8.safetensors"
CHECKPOINT_DEV = "ltx-2-19b-dev.safetensors"
SPATIAL_UPSAMPLER = "ltx-2-spatial-upscaler-x2-1.0.safetensors"
DISTILLED_LORA = "ltx-2-19b-distilled-lora-384.safetensors"


def get_cache_dir():
    """Get the cache directory for model weights."""
    return os.path.expanduser("~/.cache/huggingface/hub")


def download_model_file(filename: str, repo_id: str = LTX2_REPO) -> str:
    """Download a single model file from HuggingFace."""
    print(f"Downloading {filename}...")
    return hf_hub_download(repo_id=repo_id, filename=filename)


def download_gemma_encoder() -> str:
    """Download Gemma text encoder."""
    print("Downloading Gemma text encoder...")
    return snapshot_download(repo_id=GEMMA_REPO)


def parse_loras(lora_args: list) -> list:
    """Parse LoRA arguments into LoraPathStrengthAndSDOps objects."""
    if not lora_args:
        return []

    loras = []
    for lora_spec in lora_args:
        path = lora_spec[0]
        strength = float(lora_spec[1]) if len(lora_spec) > 1 else 1.0
        loras.append(LoraPathStrengthAndSDOps(path, strength, LTXV_LORA_COMFY_RENAMING_MAP))
    return loras


def get_model_paths(pipeline: str, fp8: bool) -> dict:
    """Get or download required model paths."""
    paths = {}

    # Select checkpoint based on pipeline and precision
    if pipeline == "distilled":
        checkpoint_file = CHECKPOINT_DISTILLED_FP8 if fp8 else CHECKPOINT_DISTILLED
    else:  # two_stage
        checkpoint_file = CHECKPOINT_DEV_FP8 if fp8 else CHECKPOINT_DEV

    paths["checkpoint"] = download_model_file(checkpoint_file)
    paths["spatial_upsampler"] = download_model_file(SPATIAL_UPSAMPLER)
    paths["gemma"] = download_gemma_encoder()

    # two_stage requires distilled LoRA
    if pipeline == "two_stage":
        paths["distilled_lora"] = download_model_file(DISTILLED_LORA)

    return paths


def run_distilled_pipeline(
    prompt: str,
    output_path: str,
    model_paths: dict,
    image_path: str = None,
    seed: int = 10,
    height: int = 1024,
    width: int = 1536,
    num_frames: int = 121,
    frame_rate: float = 24.0,
    fp8: bool = True,
    enhance_prompt: bool = False,
    loras: list = None,
):
    """Run the distilled pipeline (faster, 8+4 steps)."""
    if loras is None:
        loras = []

    print("Loading distilled pipeline...")
    pipeline = DistilledPipeline(
        checkpoint_path=model_paths["checkpoint"],
        gemma_root=model_paths["gemma"],
        spatial_upsampler_path=model_paths["spatial_upsampler"],
        loras=loras,
        fp8transformer=fp8,
    )

    # Prepare image conditioning if provided
    images = []
    if image_path:
        images = [(image_path, 0, 1.0)]  # Image at frame 0, strength 1.0

    print(f"Generating video: {width}x{height}, {num_frames} frames...")
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    with torch.inference_mode():
        video, audio = pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            tiling_config=tiling_config,
            enhance_prompt=enhance_prompt,
        )

        print(f"Encoding video to {output_path}...")
        encode_video(
            video=video,
            fps=frame_rate,
            audio=audio,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            output_path=output_path,
            video_chunks_number=video_chunks_number,
        )

    return output_path


def run_two_stage_pipeline(
    prompt: str,
    output_path: str,
    model_paths: dict,
    image_path: str = None,
    seed: int = 10,
    height: int = 1024,
    width: int = 1536,
    num_frames: int = 121,
    frame_rate: float = 24.0,
    num_inference_steps: int = 40,
    cfg_guidance_scale: float = 4.0,
    negative_prompt: str = "",
    fp8: bool = True,
    enhance_prompt: bool = False,
    loras: list = None,
):
    """Run the two-stage pipeline (higher quality, more steps)."""
    if loras is None:
        loras = []

    if not negative_prompt:
        negative_prompt = DEFAULT_NEGATIVE_PROMPT

    # Prepare distilled LoRA
    distilled_lora = [
        LoraPathStrengthAndSDOps(
            model_paths["distilled_lora"],
            0.6,
            LTXV_LORA_COMFY_RENAMING_MAP
        )
    ]

    print("Loading two-stage pipeline...")
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=model_paths["checkpoint"],
        distilled_lora=distilled_lora,
        spatial_upsampler_path=model_paths["spatial_upsampler"],
        gemma_root=model_paths["gemma"],
        loras=loras,
        fp8transformer=fp8,
    )

    # Prepare image conditioning if provided
    images = []
    if image_path:
        images = [(image_path, 0, 1.0)]

    print(f"Generating video: {width}x{height}, {num_frames} frames, {num_inference_steps} steps...")
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    with torch.inference_mode():
        video, audio = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            cfg_guidance_scale=cfg_guidance_scale,
            images=images,
            tiling_config=tiling_config,
            enhance_prompt=enhance_prompt,
        )

        print(f"Encoding video to {output_path}...")
        encode_video(
            video=video,
            fps=frame_rate,
            audio=audio,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            output_path=output_path,
            video_chunks_number=video_chunks_number,
        )

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="LTX-2: Audio-Video Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Pipeline selection
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["distilled", "two_stage"],
        default="distilled",
        help="Pipeline mode: distilled (fast, 8+4 steps) or two_stage (quality, 40 steps)",
    )

    # Input/output
    parser.add_argument(
        "--prompt",
        type=str,
        default="A serene mountain landscape at sunset, with golden light reflecting off a calm lake. "
                "Birds fly across the sky as gentle wind rustles through pine trees.",
        help="Text prompt describing the video to generate",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional input image for image-to-video generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Output video path",
    )

    # Video parameters
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Video height (must be divisible by 64 for two-stage)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1536,
        help="Video width (must be divisible by 64 for two-stage)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=121,
        help="Number of frames (must satisfy (F-1) %% 8 == 0)",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=24.0,
        help="Frame rate (FPS)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Random seed for reproducibility",
    )

    # Two-stage specific
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=40,
        help="Denoising steps (two_stage pipeline only)",
    )
    parser.add_argument(
        "--cfg-guidance-scale",
        type=float,
        default=4.0,
        help="CFG guidance scale (two_stage pipeline only)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt (two_stage pipeline only, uses default if empty)",
    )

    # LoRA
    parser.add_argument(
        "--lora",
        action="append",
        nargs="+",
        metavar=("PATH", "STRENGTH"),
        default=[],
        help="LoRA model: path and optional strength (default 1.0). Can be specified multiple times.",
    )

    # Optimization
    parser.add_argument(
        "--no-fp8",
        action="store_true",
        help="Disable FP8 mode (uses more VRAM but may be slightly higher quality)",
    )
    parser.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="Enhance prompt using the model's prompt enhancement feature",
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
        for logger_name in ["transformers", "torch", "accelerate"]:
            logging.getLogger(logger_name).setLevel(logging.INFO)

    # Resolve paths
    output_path = resolve_output_path(args.output)
    image_path = None
    if args.image:
        image_path = resolve_input_path(args.image)
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            sys.exit(1)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fp8 = not args.no_fp8
    loras = parse_loras(args.lora)

    print(f"Pipeline: {args.pipeline}")
    print(f"FP8 mode: {'enabled' if fp8 else 'disabled'}")
    if loras:
        print(f"LoRAs: {len(loras)}")
    print(f"Output: {output_path}")

    # Download models
    print("\nPreparing models...")
    model_paths = get_model_paths(args.pipeline, fp8)

    # Run pipeline
    if args.pipeline == "distilled":
        run_distilled_pipeline(
            prompt=args.prompt,
            output_path=output_path,
            model_paths=model_paths,
            image_path=image_path,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            fp8=fp8,
            enhance_prompt=args.enhance_prompt,
            loras=loras,
        )
    else:
        run_two_stage_pipeline(
            prompt=args.prompt,
            output_path=output_path,
            model_paths=model_paths,
            image_path=image_path,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            num_inference_steps=args.num_inference_steps,
            cfg_guidance_scale=args.cfg_guidance_scale,
            negative_prompt=args.negative_prompt,
            fp8=fp8,
            enhance_prompt=args.enhance_prompt,
            loras=loras,
        )

    print(f"\nDone! Video saved to: {output_path}")


if __name__ == "__main__":
    main()
