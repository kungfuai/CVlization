#!/usr/bin/env python3
"""
CVlization wrapper for Krea Realtime video generation using Scope framework.
Supports offline batch video generation from text prompts.
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from diffusers.utils import export_to_video
from PIL import Image

# Add Scope to path
sys.path.insert(0, "/opt/scope/src")

from scope.core.config import get_model_file_path, get_models_dir
from scope.core.pipelines.krea_realtime_video.pipeline import KreaRealtimeVideoPipeline
from scope.core.pipelines.utils import Quantization


def download_models():
    """Download required models if they don't exist."""
    import json
    from huggingface_hub import hf_hub_download, snapshot_download

    model_folder = os.environ.get("MODEL_FOLDER", "/root/.cache/huggingface/wan_models")

    # Download base model (Wan 2.1 T2V 1.3B) - needed for VAE and text encoder
    base_model_path = Path(model_folder) / "Wan2.1-T2V-1.3B"
    if not base_model_path.exists():
        print("📥 Downloading base model (Wan-AI/Wan2.1-T2V-1.3B)...")
        print("   This is a large download (~15GB) and will take several minutes...")
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-T2V-1.3B",
            local_dir=str(base_model_path),
            local_dir_use_symlinks=False,
        )
        print("✅ Base model downloaded successfully")

    # Create 14B model config if it doesn't exist or is a symlink
    model_14b_path = Path(model_folder) / "Wan2.1-T2V-14B"
    if model_14b_path.is_symlink():
        print(f"🔧 Removing incorrect symlink: {model_14b_path}")
        model_14b_path.unlink()

    if not model_14b_path.exists():
        print(f"🔧 Creating 14B model config directory: {model_14b_path}")
        model_14b_path.mkdir(parents=True, exist_ok=True)

    config_path = model_14b_path / "config.json"
    if not config_path.exists():
        print(f"🔧 Creating 14B model config.json")
        config_14b = {
            "_class_name": "WanModel",
            "_diffusers_version": "0.30.0",
            "dim": 5120,
            "eps": 1e-06,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "in_dim": 16,
            "model_type": "t2v",
            "num_heads": 40,
            "num_layers": 40,
            "out_dim": 16,
            "text_len": 512
        }
        with open(config_path, "w") as f:
            json.dump(config_14b, f, indent=2)
        print("✅ 14B model config created successfully")

    # Download Krea checkpoint to persistent cache
    checkpoint_cache_dir = Path("/root/.cache/huggingface/checkpoints")
    checkpoint_cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cache_path = checkpoint_cache_dir / "krea-realtime-video-14b.safetensors"

    if not checkpoint_cache_path.exists():
        print("📥 Downloading Krea checkpoint (krea/krea-realtime-video)...")
        print("   This is a large file (~28.6GB) and will take several minutes...")
        hf_hub_download(
            repo_id="krea/krea-realtime-video",
            filename="krea-realtime-video-14b.safetensors",
            local_dir=str(checkpoint_cache_dir),
            local_dir_use_symlinks=False,
        )
        print("✅ Krea checkpoint downloaded successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Generate videos using Krea Realtime via Scope framework"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for video generation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/video.mp4",
        help="Output video path (default: outputs/video.mp4)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width (default: 832)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height (default: 480)"
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=9,
        help="Number of blocks to generate (default: 9, ~9 seconds at 24fps)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second for output video (default: 24)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["none", "fp8"],
        help="Quantization mode (default: none, use fp8 for 32GB VRAM GPUs)"
    )

    args = parser.parse_args()

    # Set up paths
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download models if needed
    download_models()

    print("=" * 80)
    print("Krea Realtime Video Generation (via Scope)")
    print("=" * 80)
    print(f"Prompt: {args.prompt}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Blocks: {args.num_blocks}")
    print(f"Seed: {args.seed}")
    print(f"FPS: {args.fps}")
    print(f"Output: {output_path.absolute()}")
    print("=" * 80)
    print()

    # Set up configuration
    model_folder = os.environ.get("MODEL_FOLDER", "/root/.cache/huggingface/wan_models")
    checkpoint_path = Path("/root/.cache/huggingface/checkpoints/krea-realtime-video-14b.safetensors")

    config = OmegaConf.create({
        "model_dir": str(model_folder),
        "generator_path": str(checkpoint_path),
        "text_encoder_path": str(Path(model_folder) / "Wan2.1-T2V-1.3B" / "models_t5_umt5-xxl-enc-bf16.pth"),
        "tokenizer_path": str(Path(model_folder) / "Wan2.1-T2V-1.3B" / "google" / "umt5-xxl"),
        "vae_path": str(Path(model_folder) / "Wan2.1-T2V-1.3B" / "Wan2.1_VAE.pth"),
        "model_config": OmegaConf.load(Path(__file__).parent / "model_config.yaml"),
        "height": args.height,
        "width": args.width,
    })

    # Initialize pipeline
    print("🔄 Loading models (this may take a few minutes on first run)...")
    device = torch.device("cuda")
    quantization = Quantization.FP8_E4M3FN if args.quantization == "fp8" else None

    pipeline = KreaRealtimeVideoPipeline(
        config,
        quantization=quantization,
        compile=False,
        device=device,
        dtype=torch.bfloat16,
    )

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Generate video
    print(f"🎬 Generating video ({args.num_blocks} blocks)...")
    prompts = [{"text": args.prompt, "weight": 100}]

    # Generate frames by calling pipeline multiple times (realtime streaming approach)
    # Each call generates one block (~9 frames), KV caching maintains temporal consistency
    outputs = []
    for block_idx in range(args.num_blocks):
        print(f"  Generating block {block_idx + 1}/{args.num_blocks}...")
        output = pipeline(prompts=prompts, kv_cache_attention_bias=0.3)
        outputs.append(output.detach().cpu())

    # Concatenate all blocks into a single video tensor (T, C, H, W)
    output_frames = torch.concat(outputs)
    print(f"  Generated {output_frames.shape[0]} total frames")

    # Convert frames to numpy arrays if needed
    # Handle both single tensor and list of frames
    if isinstance(output_frames, torch.Tensor):
        # Single tensor output: (B, T, C, H, W) or similar
        # Convert to list of frames
        if output_frames.ndim == 5:
            # (B, T, C, H, W) -> list of (H, W, C)
            B, T, C, H, W = output_frames.shape
            output_frames = output_frames[0]  # Take first batch
        if output_frames.ndim == 4:
            # (T, C, H, W) -> list of (H, W, C)
            converted_frames = []
            for t in range(output_frames.shape[0]):
                frame = output_frames[t].permute(1, 2, 0).cpu().numpy()
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                # Ensure correct (H, W, C) shape - fix if needed
                if frame.shape[2] != 3 and frame.shape[1] == 3:
                    # Shape is (W, C, H), need to transpose to (H, W, C)
                    frame = frame.transpose(2, 0, 1)
                converted_frames.append(frame)
            output_frames = converted_frames
    elif isinstance(output_frames, (list, tuple)):
        # List of frames - check if they need conversion
        if len(output_frames) > 0 and not isinstance(output_frames[0], np.ndarray):
            converted_frames = []
            for frame in output_frames:
                if isinstance(frame, Image.Image):
                    converted_frames.append(np.array(frame))
                elif isinstance(frame, torch.Tensor):
                    # Convert tensor to numpy: (C, H, W) -> (H, W, C) and scale to [0, 255]
                    frame_np = frame.permute(1, 2, 0).cpu().numpy()
                    if frame_np.max() <= 1.0:
                        frame_np = (frame_np * 255).astype(np.uint8)
                    converted_frames.append(frame_np)
                else:
                    converted_frames.append(frame)
            output_frames = converted_frames

    # Export to video
    print("💾 Saving video...")
    export_to_video(output_frames, str(output_path), fps=args.fps)

    print()
    print("=" * 80)
    print(f"✅ Video saved to: {output_path.absolute()}")
    print(f"📊 Frames: {len(output_frames)}")
    print(f"🎬 Duration: {len(output_frames) / args.fps:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
