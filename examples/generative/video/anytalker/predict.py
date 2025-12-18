#!/usr/bin/env python3
"""
AnyTalker inference wrapper for CVlization.

Generates talking head videos from an image and audio file.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add vendored AnyTalker to path
sys.path.insert(0, "/workspace/local/vendor")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)


def download_models_if_needed():
    """Download model checkpoints if not present."""
    from huggingface_hub import snapshot_download

    checkpoints_dir = Path("/workspace/AnyTalker/checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Download Wan2.1-Fun-V1.1-1.3B-InP (base model)
    base_model_dir = checkpoints_dir / "Wan2.1-Fun-1.3B-Inp"
    if not base_model_dir.exists() or not any(base_model_dir.iterdir()):
        logging.info("Downloading Wan2.1-Fun-V1.1-1.3B-InP base model...")
        snapshot_download(
            repo_id="alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP",
            local_dir=str(base_model_dir),
            local_dir_use_symlinks=False,
        )

    # Download AnyTalker-1.3B (fine-tuned weights)
    anytalker_dir = checkpoints_dir / "AnyTalker"
    if not anytalker_dir.exists() or not any(anytalker_dir.iterdir()):
        logging.info("Downloading AnyTalker-1.3B fine-tuned weights...")
        snapshot_download(
            repo_id="zzz66/AnyTalker-1.3B",
            local_dir=str(anytalker_dir),
            local_dir_use_symlinks=False,
        )

    # Download wav2vec2-base-960h (audio encoder)
    wav2vec_dir = checkpoints_dir / "wav2vec2-base-960h"
    if not wav2vec_dir.exists() or not any(wav2vec_dir.iterdir()):
        logging.info("Downloading wav2vec2-base-960h audio encoder...")
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
    caption: str = "A person is talking.",
    use_half: bool = True,
    sample_steps: int = 40,
    guide_scale: float = 4.5,
):
    """Run AnyTalker inference."""
    import torch
    from PIL import Image

    import wan
    from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
    from wan.utils.utils import cache_video
    from wan.utils.infer_utils import calculate_frame_num_from_audio
    from utils.get_face_bbox import FaceInference

    # Paths
    checkpoint_dir = "/workspace/AnyTalker/checkpoints/Wan2.1-Fun-1.3B-Inp"
    post_trained_path = "/workspace/AnyTalker/checkpoints/AnyTalker/1_3B-single-v1.pth"
    dit_config_path = "/workspace/AnyTalker/checkpoints/AnyTalker/config_af2v_1_3B.json"

    # Load config
    cfg = WAN_CONFIGS["a2v-1.3B"]
    cfg.fps = 24
    size = "832*480"

    # Initialize model
    logging.info("Initializing AnyTalker model...")
    model = wan.WanAF2V(
        config=cfg,
        checkpoint_dir=checkpoint_dir,
        device_id=0,
        rank=0,
        use_half=use_half,
        t5_fsdp=False,
        dit_fsdp=False,
        t5_cpu=False,
        post_trained_checkpoint_path=post_trained_path,
        dit_config=dit_config_path,
    )

    # Initialize face processor
    logging.info("Initializing face processor...")
    face_processor = FaceInference(det_thresh=0.15, ctx_id=0)

    # Prepare inputs
    audio_paths = [audio_path]

    # Calculate frame number from audio
    frame_num = calculate_frame_num_from_audio(audio_paths, fps=24, mode="pad")
    logging.info(f"Calculated frame number: {frame_num}")

    # Load image
    img = Image.open(image_path).convert("RGB")

    # Generate video
    logging.info(f"Generating video from {image_path} + {audio_path}...")
    video = model.generate(
        input_prompt=caption,
        img=img,
        audio=audio_path,
        max_area=MAX_AREA_CONFIGS[size],
        frame_num=frame_num,
        shift=3.0,  # Recommended for 480p
        sample_solver="unipc",
        sampling_steps=sample_steps,
        guide_scale=guide_scale,
        seed=42,
        offload_model=True,
        cfg_zero=False,
        zero_init_steps=0,
        face_processor=face_processor,
        img_path=image_path,
        audio_paths=audio_paths,
        task_key="cvl_inference",
        mode="pad",
    )

    # Handle output
    if isinstance(video, dict):
        video = video["original"]

    if video is None:
        raise RuntimeError("Video generation failed")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save video without audio first
    temp_video_path = output_path.replace(".mp4", "_temp.mp4")
    logging.info(f"Saving video to {temp_video_path}...")
    cache_video(
        tensor=video[None],
        save_file=temp_video_path,
        fps=24,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )

    # Add audio using ffmpeg
    logging.info("Adding audio track...")
    ffmpeg_cmd = f'ffmpeg -y -i "{temp_video_path}" -i "{audio_path}" -vcodec libx264 -acodec aac -crf 18 -shortest "{output_path}"'
    os.system(ffmpeg_cmd)

    # Clean up temp file
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

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
    parser = argparse.ArgumentParser(description="Generate talking head video with AnyTalker")
    parser.add_argument("--image", type=str, default="examples/images/1p-0.png", help="Path to input image")
    parser.add_argument("--audio", type=str, default="examples/audios/1p-0.wav", help="Path to input audio (WAV)")
    parser.add_argument("--output", type=str, default="/workspace/local/outputs/output.mp4", help="Output video path")
    parser.add_argument("--caption", type=str, default="A person is talking.", help="Scene description")
    parser.add_argument("--steps", type=int, default=40, help="Sampling steps")
    parser.add_argument("--guidance", type=float, default=4.5, help="Guidance scale")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 instead of FP16")
    parser.add_argument("--skip-download", action="store_true", help="Skip model download check")

    args = parser.parse_args()

    # Resolve paths
    image_path = resolve_path(args.image)
    audio_path = resolve_path(args.audio)
    output_path = resolve_path(args.output)

    logging.info(f"Image: {image_path}")
    logging.info(f"Audio: {audio_path}")
    logging.info(f"Output: {output_path}")

    # Ensure models are downloaded
    if not args.skip_download:
        download_models_if_needed()

    # Run inference
    run_inference(
        image_path=image_path,
        audio_path=audio_path,
        output_path=output_path,
        caption=args.caption,
        use_half=not args.fp32,
        sample_steps=args.steps,
        guide_scale=args.guidance,
    )


if __name__ == "__main__":
    main()
