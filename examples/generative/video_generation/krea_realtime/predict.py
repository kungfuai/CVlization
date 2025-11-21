#!/usr/bin/env python3
"""
CVlization wrapper for Krea Realtime video generation using official SDK.
Supports offline batch video generation from text prompts.
"""
import os
import sys
import argparse
from pathlib import Path

# Ensure SDK is in path
sys.path.insert(0, "/opt/krea-sdk")

from release_server import GenerateParams, load_merge_config, load_all
from sample import sample_videos


def download_models():
    """Download required models if they don't exist."""
    from huggingface_hub import hf_hub_download, snapshot_download

    # Download base model (Wan 2.1 T2V 1.3B)
    model_folder = os.environ.get("MODEL_FOLDER", "/root/.cache/huggingface/wan_models")
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

    # Download Krea checkpoint
    checkpoint_path = Path("/opt/krea-sdk/checkpoints/krea-realtime-video-14b.safetensors")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        print("📥 Downloading Krea checkpoint (krea/krea-realtime-video)...")
        print("   This is a large file (~28.6GB) and will take several minutes...")
        hf_hub_download(
            repo_id="krea/krea-realtime-video",
            filename="krea-realtime-video-14b.safetensors",
            local_dir=str(checkpoint_path.parent),
            local_dir_use_symlinks=False,
        )
        print("✅ Krea checkpoint downloaded successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Generate videos using Krea Realtime SDK"
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
        "--kv-cache-frames",
        type=int,
        default=3,
        help="Number of frames for KV cache (default: 3)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/self_forcing_server_14b.yaml",
        help="Path to SDK config file (relative to /opt/krea-sdk)"
    )

    args = parser.parse_args()

    # Set up paths
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Change to SDK directory for config access
    sdk_dir = Path("/opt/krea-sdk")
    os.chdir(sdk_dir)

    # Download models if needed
    download_models()

    print("=" * 80)
    print("Krea Realtime Video Generation")
    print("=" * 80)
    print(f"Prompt: {args.prompt}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Blocks: {args.num_blocks}")
    print(f"Seed: {args.seed}")
    print(f"FPS: {args.fps}")
    print(f"Output: {output_path.absolute()}")
    print("=" * 80)
    print()

    # Configure generation parameters
    params = GenerateParams(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_blocks=args.num_blocks,
        seed=args.seed,
        kv_cache_num_frames=args.kv_cache_frames,
    )

    # Generate video
    print("🔄 Loading models (this may take a few minutes on first run)...")
    results = sample_videos(
        prompts_list=[args.prompt],
        config_path=args.config,
        output_dir=str(output_dir.absolute()),
        params=params,
        models=None,  # Will load from config
        save_videos=True,
        fps=args.fps,
    )

    # Move generated video to desired output path
    if results and 0 in results:
        generated_path = results[0].get("video_path")
        if generated_path and Path(generated_path).exists():
            import shutil
            shutil.move(str(generated_path), str(output_path))
            print()
            print("=" * 80)
            print(f"✅ Video saved to: {output_path.absolute()}")
            print(f"📊 Frames: {results[0]['num_frames']}")
            print(f"🎬 Duration: {results[0]['num_frames'] / args.fps:.2f}s")
            print("=" * 80)
        else:
            print("❌ Video generation failed")
            sys.exit(1)
    else:
        print("❌ No results returned from video generation")
        sys.exit(1)


if __name__ == "__main__":
    main()
