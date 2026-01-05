#!/usr/bin/env python3
"""
Wan2.2-S2V-14B inference wrapper for CVlization.

Generates audio-driven talking head videos from an image, audio, and text prompt.
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)

# HuggingFace dataset for sample inputs
HF_DATASET_REPO = "zzsi/cvl"
HF_SAMPLE_SUBDIR = "wan_s2v"

# Default paths
DEFAULT_OUTPUT = "outputs/output.mp4"
MODELS_DIR = Path("/models")
TEST_INPUTS_DIR = Path("/tmp/wan_s2v_samples")


def download_sample_inputs():
    """Download sample inputs from HuggingFace dataset if not present."""
    from huggingface_hub import hf_hub_download

    TEST_INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    default_image = TEST_INPUTS_DIR / "sample.jpg"
    default_audio = TEST_INPUTS_DIR / "sample.wav"

    try:
        if not default_image.exists():
            logging.info("Downloading sample image from HuggingFace...")
            hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=f"{HF_SAMPLE_SUBDIR}/sample.jpg",
                repo_type="dataset",
                local_dir=str(TEST_INPUTS_DIR),
            )
            src = TEST_INPUTS_DIR / HF_SAMPLE_SUBDIR / "sample.jpg"
            if src.exists():
                src.rename(default_image)

        if not default_audio.exists():
            logging.info("Downloading sample audio from HuggingFace...")
            hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=f"{HF_SAMPLE_SUBDIR}/sample.wav",
                repo_type="dataset",
                local_dir=str(TEST_INPUTS_DIR),
            )
            src = TEST_INPUTS_DIR / HF_SAMPLE_SUBDIR / "sample.wav"
            if src.exists():
                src.rename(default_audio)

        # Clean up subdirectory
        subdir = TEST_INPUTS_DIR / HF_SAMPLE_SUBDIR
        if subdir.exists():
            shutil.rmtree(subdir, ignore_errors=True)

    except Exception as e:
        logging.warning(f"Could not download samples from HuggingFace: {e}")
        # Fall back to Wan2.2 repo examples
        wan_examples = Path("/opt/Wan2.2/examples")
        if wan_examples.exists():
            if not default_image.exists() and (wan_examples / "i2v_input.JPG").exists():
                shutil.copy(wan_examples / "i2v_input.JPG", default_image)
            if not default_audio.exists() and (wan_examples / "talk.wav").exists():
                shutil.copy(wan_examples / "talk.wav", default_audio)

    return default_image, default_audio


def download_models_if_needed():
    """Download Wan2.2-S2V-14B model checkpoints if not present."""
    from huggingface_hub import snapshot_download

    model_dir = MODELS_DIR / "Wan2.2-S2V-14B"

    # Check if model is already downloaded
    if model_dir.exists() and any(model_dir.glob("*.safetensors")):
        logging.info(f"Model already exists at {model_dir}")
        return model_dir

    logging.info("Downloading Wan2.2-S2V-14B from HuggingFace (this may take a while)...")
    snapshot_download(
        repo_id="Wan-AI/Wan2.2-S2V-14B",
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )

    logging.info(f"Model downloaded to {model_dir}")
    return model_dir


def run_inference(
    image_path: str,
    audio_path: str,
    output_path: str,
    prompt: str = "A person is talking naturally with expressive gestures.",
    sample_steps: int = 40,
    guide_scale: float = 4.5,
    infer_frames: int = 80,
    seed: int = 42,
    offload_model: bool = True,
):
    """Run Wan2.2-S2V-14B inference."""
    import torch
    from PIL import Image

    import wan
    from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
    from wan.utils.utils import save_video, merge_video_audio

    # Get model directory
    model_dir = MODELS_DIR / "Wan2.2-S2V-14B"

    # Load config
    cfg = WAN_CONFIGS["s2v-14B"]

    logging.info(f"Initializing Wan2.2-S2V-14B pipeline...")
    logging.info(f"  Image: {image_path}")
    logging.info(f"  Audio: {audio_path}")
    logging.info(f"  Prompt: {prompt}")

    # Initialize S2V pipeline
    wan_s2v = wan.WanS2V(
        config=cfg,
        checkpoint_dir=str(model_dir),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=False,
    )

    # Generate video
    logging.info(f"Generating video with {sample_steps} steps...")
    video = wan_s2v.generate(
        input_prompt=prompt,
        ref_image_path=image_path,
        audio_path=audio_path,
        enable_tts=False,
        tts_prompt_audio=None,
        tts_prompt_text=None,
        tts_text=None,
        num_repeat=None,
        pose_video=None,
        max_area=MAX_AREA_CONFIGS["1280*720"],
        infer_frames=infer_frames,
        shift=cfg.sample_shift,
        sample_solver="unipc",
        sampling_steps=sample_steps,
        guide_scale=guide_scale,
        seed=seed,
        offload_model=offload_model,
        init_first_frame=False,
    )

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save video
    temp_video = output_path.replace(".mp4", "_temp.mp4")
    logging.info(f"Saving video to {temp_video}...")
    save_video(
        tensor=video[None],
        save_file=temp_video,
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )

    # Merge with audio
    logging.info("Merging video with audio...")
    merge_video_audio(video_path=temp_video, audio_path=audio_path)

    # The merged video replaces temp_video with same name minus _temp
    merged_path = temp_video.replace("_temp.mp4", "_temp_with_audio.mp4")
    if os.path.exists(merged_path):
        shutil.move(merged_path, output_path)
    elif os.path.exists(temp_video):
        shutil.move(temp_video, output_path)

    # Cleanup
    for f in [temp_video, merged_path]:
        if os.path.exists(f) and f != output_path:
            os.remove(f)

    logging.info(f"Video saved to: {output_path}")
    return output_path


def resolve_path(path: str, check_exists: bool = True) -> str:
    """Resolve input path - check multiple locations."""
    if os.path.isabs(path) and os.path.exists(path):
        return path

    # Check relative to CVL workspace
    cvl_inputs = os.environ.get("CVL_INPUTS", "/mnt/cvl/workspace")
    cvl_path = os.path.join(cvl_inputs, path)
    if os.path.exists(cvl_path):
        return cvl_path

    # Check relative to /user_data
    user_path = os.path.join("/user_data", path)
    if os.path.exists(user_path):
        return user_path

    # Check relative to local examples
    local_path = os.path.join("/workspace/local/examples", path)
    if os.path.exists(local_path):
        return local_path

    if check_exists:
        raise FileNotFoundError(f"Could not find file: {path}")
    return path


def resolve_output_path(path: str) -> str:
    """Resolve output path."""
    if os.path.isabs(path):
        return path

    cvl_outputs = os.environ.get("CVL_OUTPUTS", "/mnt/cvl/workspace")
    return os.path.join(cvl_outputs, path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate audio-driven talking head video with Wan2.2-S2V-14B"
    )
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image (default: download sample)")
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to input audio WAV/MP3 (default: download sample)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output video path")
    parser.add_argument("--prompt", type=str,
                        default="A person is talking naturally with expressive gestures.",
                        help="Text prompt describing the scene")
    parser.add_argument("--steps", type=int, default=40,
                        help="Sampling steps (default: 40)")
    parser.add_argument("--guidance", type=float, default=4.5,
                        help="Guidance scale (default: 4.5)")
    parser.add_argument("--frames", type=int, default=80,
                        help="Frames per clip (default: 80, must be multiple of 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no-offload", action="store_true",
                        help="Disable model offloading (requires more VRAM)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip model download check")

    args = parser.parse_args()

    # Download models if needed
    if not args.skip_download:
        download_models_if_needed()

    # Handle input paths
    if args.image is None or args.audio is None:
        default_image, default_audio = download_sample_inputs()
        if args.image is None:
            args.image = str(default_image)
            logging.info(f"Using sample image: {args.image}")
        if args.audio is None:
            args.audio = str(default_audio)
            logging.info(f"Using sample audio: {args.audio}")

    # Resolve paths
    image_path = resolve_path(args.image)
    audio_path = resolve_path(args.audio)
    output_path = resolve_output_path(args.output)

    logging.info(f"Image: {image_path}")
    logging.info(f"Audio: {audio_path}")
    logging.info(f"Output: {output_path}")

    # Run inference
    run_inference(
        image_path=image_path,
        audio_path=audio_path,
        output_path=output_path,
        prompt=args.prompt,
        sample_steps=args.steps,
        guide_scale=args.guidance,
        infer_frames=args.frames,
        seed=args.seed,
        offload_model=not args.no_offload,
    )


if __name__ == "__main__":
    main()
