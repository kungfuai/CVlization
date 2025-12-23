#!/usr/bin/env python3
"""
Wan2GP Video Generation - Command Line Interface

This script provides a simple CLI for generating videos using the Wan2GP models.
Supports both text-to-video (T2V) and image-to-video (I2V) generation.
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torchvision
from PIL import Image

from cvlization.paths import resolve_input_path, resolve_output_path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import wan
from wan.configs import WAN_CONFIGS


def save_video(video_tensor, output_path, fps=8):
    """
    Save video tensor to file.

    Args:
        video_tensor: torch.Tensor of shape (C, T, H, W) in range [-1, 1]
        output_path: str, path to save video
        fps: int, frames per second
    """
    # Denormalize from [-1, 1] to [0, 1]
    video = (video_tensor + 1.0) / 2.0
    video = video.clamp(0, 1)

    # Save using torchvision
    video = (video * 255).to(torch.uint8)
    torchvision.io.write_video(output_path, video.permute(1, 2, 3, 0).cpu(), fps=fps)
    print(f"Video saved to: {output_path}")


def text_to_video(args):
    """Generate video from text prompt."""
    print(f"Initializing T2V model: {args.model_size}")

    # Select config based on model size
    if args.model_size == "1.3B":
        config_key = "t2v-1.3B"
    elif args.model_size == "14B":
        config_key = "t2v-14B"
    else:
        raise ValueError(f"Unsupported model size: {args.model_size}")

    config = WAN_CONFIGS[config_key]

    # Initialize model
    model = wan.WanT2V(
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        device_id=args.device_id,
        model_filename=args.model_path,
        text_encoder_filename=args.text_encoder_path,
    )

    print(f"Generating video from prompt: {args.prompt}")

    # Parse size
    width, height = map(int, args.size.split('x'))

    # Generate video
    video = model.generate(
        input_prompt=args.prompt,
        size=(width, height),
        frame_num=args.frames,
        shift=args.shift,
        sample_solver=args.solver,
        sampling_steps=args.steps,
        guide_scale=args.guidance_scale,
        n_prompt=args.negative_prompt,
        seed=args.seed,
        offload_model=not args.no_offload,
        enable_RIFLEx=args.enable_riflex,
        VAE_tile_size=args.vae_tile_size,
    )

    if video is not None:
        output_path = resolve_output_path(args.output)
        save_video(video, output_path, fps=args.fps)
    else:
        print("Generation was interrupted or failed")


def image_to_video(args):
    """Generate video from image and text prompt."""
    print(f"Initializing I2V model: 14B")

    config = WAN_CONFIGS["i2v-14B"]

    # Initialize model
    model = wan.WanI2V(
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        device_id=args.device_id,
        i2v720p=args.resolution == "720p",
        model_filename=args.model_path,
        text_encoder_filename=args.text_encoder_path,
    )

    # Load input image
    image_path = resolve_input_path(args.image)
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")

    print(f"Generating video from image with prompt: {args.prompt}")

    # Parse max area
    width, height = map(int, args.max_area.split('x'))
    max_area = width * height

    # Generate video
    video = model.generate(
        input_prompt=args.prompt,
        img=image,
        max_area=max_area,
        frame_num=args.frames,
        shift=args.shift,
        sample_solver=args.solver,
        sampling_steps=args.steps,
        guide_scale=args.guidance_scale,
        n_prompt=args.negative_prompt,
        seed=args.seed,
        offload_model=not args.no_offload,
        enable_RIFLEx=args.enable_riflex,
        VAE_tile_size=args.vae_tile_size,
    )

    if video is not None:
        output_path = resolve_output_path(args.output)
        save_video(video, output_path, fps=args.fps)
    else:
        print("Generation was interrupted or failed")


def main():
    parser = argparse.ArgumentParser(
        description="Wan2GP Video Generation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text-to-video (1.3B model)
  python predict.py t2v \\
    --prompt "A cat playing piano" \\
    --checkpoint-dir /workspace/models \\
    --output output.mp4

  # Text-to-video (14B model, higher quality)
  python predict.py t2v \\
    --prompt "A sunset over the ocean" \\
    --model-size 14B \\
    --checkpoint-dir /workspace/models \\
    --output sunset.mp4

  # Image-to-video
  python predict.py i2v \\
    --image input.jpg \\
    --prompt "The scene comes to life" \\
    --checkpoint-dir /workspace/models \\
    --output animated.mp4
        """
    )

    subparsers = parser.add_subparsers(dest="mode", help="Generation mode", required=True)

    # Text-to-video arguments
    t2v_parser = subparsers.add_parser("t2v", help="Text-to-video generation")
    t2v_parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    t2v_parser.add_argument("--model-size", type=str, default="1.3B", choices=["1.3B", "14B"],
                           help="Model size (default: 1.3B)")
    t2v_parser.add_argument("--size", type=str, default="1280x720",
                           help="Video resolution as WIDTHxHEIGHT (default: 1280x720)")

    # Image-to-video arguments
    i2v_parser = subparsers.add_parser("i2v", help="Image-to-video generation")
    i2v_parser.add_argument("--image", type=str, required=True, help="Input image path")
    i2v_parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    i2v_parser.add_argument("--resolution", type=str, default="720p", choices=["480p", "720p"],
                           help="Output resolution (default: 720p)")
    i2v_parser.add_argument("--max-area", type=str, default="1280x720",
                           help="Max area as WIDTHxHEIGHT (default: 1280x720 = 921600 pixels)")

    # Common arguments for both modes
    for p in [t2v_parser, i2v_parser]:
        p.add_argument("--checkpoint-dir", type=str, required=True,
                      help="Directory containing model checkpoints")
        p.add_argument("--model-path", type=str, default=None,
                      help="Path to model checkpoint (optional, overrides checkpoint-dir)")
        p.add_argument("--text-encoder-path", type=str, default=None,
                      help="Path to text encoder checkpoint (optional)")
        p.add_argument("--output", "-o", type=str, default="output.mp4",
                      help="Output video path (default: output.mp4)")
        p.add_argument("--frames", type=int, default=81,
                      help="Number of frames (must be 4n+1, default: 81)")
        p.add_argument("--fps", type=int, default=8,
                      help="Frames per second for output video (default: 8)")
        p.add_argument("--steps", type=int, default=50,
                      help="Number of sampling steps (default: 50)")
        p.add_argument("--guidance-scale", type=float, default=5.0,
                      help="Classifier-free guidance scale (default: 5.0)")
        p.add_argument("--shift", type=float, default=5.0,
                      help="Noise schedule shift parameter (default: 5.0, use 3.0 for 480p)")
        p.add_argument("--negative-prompt", type=str, default="",
                      help="Negative prompt (default: empty)")
        p.add_argument("--solver", type=str, default="unipc", choices=["unipc", "dpm++"],
                      help="Sampling solver (default: unipc)")
        p.add_argument("--seed", type=int, default=-1,
                      help="Random seed (-1 for random, default: -1)")
        p.add_argument("--device-id", type=int, default=0,
                      help="CUDA device ID (default: 0)")
        p.add_argument("--no-offload", action="store_true",
                      help="Disable model offloading to CPU (uses more VRAM)")
        p.add_argument("--enable-riflex", action="store_true",
                      help="Enable RIFLEx (experimental)")
        p.add_argument("--vae-tile-size", type=int, default=0,
                      help="VAE tile size for tiled decoding (0 to disable, default: 0)")

    args = parser.parse_args()

    # Validate checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        print(f"Error: Checkpoint directory does not exist: {args.checkpoint_dir}")
        sys.exit(1)

    # Validate frame count
    if (args.frames - 1) % 4 != 0:
        print(f"Error: Frame count must be 4n+1 (got {args.frames})")
        print("Valid values: 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81...")
        sys.exit(1)

    # Run generation
    try:
        if args.mode == "t2v":
            text_to_video(args)
        elif args.mode == "i2v":
            image_to_video(args)
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
