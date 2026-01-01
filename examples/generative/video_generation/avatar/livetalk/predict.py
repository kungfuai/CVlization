#!/usr/bin/env python3
"""
LiveTalk inference wrapper for CVlization.

Real-time multimodal avatar video generation from reference image and audio.
Based on: https://github.com/GAIR-NLP/LiveTalk

License: CC-BY-NC-SA 4.0 (Non-commercial use only)
"""

import argparse
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


def get_example_dir() -> Path:
    return Path(os.environ.get("CVL_EXAMPLE_DIR", os.path.dirname(os.path.abspath(__file__))))


def ensure_model_weights(weights_dir: Path, skip_download: bool) -> dict:
    """Download model weights if not present. Returns paths to checkpoints."""
    from huggingface_hub import snapshot_download

    checkpoints = {
        "wan2_dir": weights_dir / "Wan2.1-T2V-1.3B",
        "livetalk_dir": weights_dir / "LiveTalk-1.3B-V0.1",
        "wav2vec_dir": weights_dir / "wav2vec2",
    }

    # Check if all exist
    all_exist = all(d.exists() for d in checkpoints.values())
    if all_exist:
        logging.info("Model weights found in cache.")
        return checkpoints

    if skip_download:
        missing = [str(d) for d in checkpoints.values() if not d.exists()]
        raise FileNotFoundError(
            f"Model weights not found and --skip-download was set. Missing: {missing}"
        )

    weights_dir.mkdir(parents=True, exist_ok=True)

    # Download Wan2.1-T2V-1.3B
    if not checkpoints["wan2_dir"].exists():
        logging.info("Downloading Wan2.1-T2V-1.3B weights...")
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-T2V-1.3B",
            local_dir=str(checkpoints["wan2_dir"]),
            local_dir_use_symlinks=False,
        )

    # Download LiveTalk-1.3B-V0.1
    if not checkpoints["livetalk_dir"].exists():
        logging.info("Downloading LiveTalk-1.3B-V0.1 weights...")
        snapshot_download(
            repo_id="GAIR/LiveTalk-1.3B-V0.1",
            local_dir=str(checkpoints["livetalk_dir"]),
            local_dir_use_symlinks=False,
        )

    # Download wav2vec2
    if not checkpoints["wav2vec_dir"].exists():
        logging.info("Downloading wav2vec2-base-960h weights...")
        snapshot_download(
            repo_id="facebook/wav2vec2-base-960h",
            local_dir=str(checkpoints["wav2vec_dir"]),
            local_dir_use_symlinks=False,
        )

    return checkpoints


def download_sample_inputs_if_needed(image_path: str, audio_path: str) -> tuple:
    """Get sample inputs, downloading from HuggingFace or using repo defaults."""
    example_dir = get_example_dir()
    test_inputs_dir = example_dir / "test_inputs"
    default_image = test_inputs_dir / "example.jpg"
    default_audio = test_inputs_dir / "example.wav"

    # Check if using defaults
    using_default_image = image_path == str(default_image)
    using_default_audio = audio_path == str(default_audio)

    if not using_default_image and not using_default_audio:
        return image_path, audio_path

    # Try to download from HuggingFace first
    needs_download = False
    if using_default_image and not default_image.exists():
        needs_download = True
    if using_default_audio and not default_audio.exists():
        needs_download = True

    if needs_download:
        try:
            from huggingface_hub import hf_hub_download

            logging.info("Downloading sample inputs from HuggingFace...")
            test_inputs_dir.mkdir(parents=True, exist_ok=True)

            if using_default_image and not default_image.exists():
                hf_hub_download(
                    repo_id="zzsi/cvl",
                    filename="livetalk/example.jpg",
                    repo_type="dataset",
                    local_dir=str(test_inputs_dir),
                )
                src = test_inputs_dir / "livetalk" / "example.jpg"
                if src.exists():
                    src.rename(default_image)

            if using_default_audio and not default_audio.exists():
                hf_hub_download(
                    repo_id="zzsi/cvl",
                    filename="livetalk/example.wav",
                    repo_type="dataset",
                    local_dir=str(test_inputs_dir),
                )
                src = test_inputs_dir / "livetalk" / "example.wav"
                if src.exists():
                    src.rename(default_audio)

            # Clean up subdirectory
            subdir = test_inputs_dir / "livetalk"
            if subdir.exists():
                shutil.rmtree(subdir, ignore_errors=True)
        except Exception as e:
            logging.warning(f"Could not download from HuggingFace: {e}")

    # Fallback to LiveTalk repo examples if HF download failed
    livetalk_examples = Path("/workspace/LiveTalk/examples/inference")
    final_image = image_path
    final_audio = audio_path

    if using_default_image and not default_image.exists():
        fallback_image = livetalk_examples / "example1.jpg"
        if fallback_image.exists():
            logging.info("Using LiveTalk repo example image as fallback")
            final_image = str(fallback_image)

    if using_default_audio and not default_audio.exists():
        fallback_audio = livetalk_examples / "example1.wav"
        if fallback_audio.exists():
            logging.info("Using LiveTalk repo example audio as fallback")
            final_audio = str(fallback_audio)

    return final_image, final_audio


def write_config(config_path: Path, checkpoints: dict, args, internal_output: Path) -> None:
    """Write inference config file."""
    config = f"""# Model paths
dtype: "bf16"
text_encoder_path: {checkpoints['wan2_dir']}/models_t5_umt5-xxl-enc-bf16.pth
dit_path: {checkpoints['livetalk_dir']}/model.safetensors
vae_path: {checkpoints['wan2_dir']}/Wan2.1_VAE.pth
wav2vec_path: {checkpoints['wav2vec_dir']}

# Input data paths
image_path: {args.image}
audio_path: {args.audio}
prompt: "{args.prompt}"
output_path: "{internal_output}"
video_duration: {args.duration}

# Generation parameters
max_hw: {args.max_hw}
image_sizes_720: [[512,512]]
fps: {args.fps}
sample_rate: 16000
num_steps: 4
local_attn_size: 15

# Causal inference parameters
denoising_step_list: [1000, 750, 500, 250]
warp_denoising_step: true
num_transformer_blocks: 30
frame_seq_length: 1024
num_frame_per_block: 3
independent_first_frame: False
"""
    config_path.write_text(config)


def run_inference(args, checkpoints: dict, output_path: Path) -> Path:
    """Run LiveTalk inference."""
    config_path = Path("/workspace/local/inference_config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Write directly to the output path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_config(config_path, checkpoints, args, output_path)

    # Use patched script if --vae-cpu is set
    if getattr(args, 'vae_cpu', False):
        script = "/workspace/patches/inference_patched.py"
        logging.info("Using CPU VAE decode (slower but lower VRAM usage)")
    else:
        script = "/workspace/LiveTalk/scripts/inference_example.py"

    cmd = [
        sys.executable,
        script,
        "--config",
        str(config_path),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace/LiveTalk:/workspace/LiveTalk/OmniAvatar:/workspace/patches"
    if getattr(args, 'vae_cpu', False):
        env["VAE_DEVICE"] = "cpu"

    logging.info("Running LiveTalk inference...")
    subprocess.run(cmd, check=True, cwd="/workspace/LiveTalk", env=env)

    if not output_path.exists():
        raise FileNotFoundError(f"Expected output not found: {output_path}")

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate talking avatar video with LiveTalk"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to reference image (PNG/JPG)"
    )
    parser.add_argument(
        "--audio", type=str, default=None,
        help="Path to driving audio (WAV, 16kHz)"
    )
    parser.add_argument(
        "--output", type=str, default="outputs/output.mp4",
        help="Output video path"
    )
    parser.add_argument(
        "--prompt", type=str,
        default="A realistic video of a person speaking directly to the camera. "
                "The individual maintains steady eye contact with clear, expressive facial features. "
                "Their facial expressions are naturally animated and emotionally engaging, "
                "with precise lip movements perfectly synchronized to their speech.",
        help="Text prompt describing the video"
    )
    parser.add_argument(
        "--duration", type=int, default=5,
        help="Video duration in seconds (must be 3n+2: 5, 8, 11, 14, ...)"
    )
    parser.add_argument(
        "--max-hw", type=int, default=720,
        help="Max height/width (720 for 480p, 1280 for 720p)"
    )
    parser.add_argument(
        "--fps", type=int, default=16,
        help="Output video FPS"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip model download check"
    )
    parser.add_argument(
        "--vae-cpu", action="store_true",
        help="Run VAE decode on CPU to reduce VRAM usage (slower but works on 16GB GPUs)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate duration (must be 3n+2)
    if (args.duration - 2) % 3 != 0:
        logging.warning(
            f"Duration {args.duration} is not 3n+2. "
            f"Recommended: 5, 8, 11, 14, 17, 20..."
        )

    example_dir = get_example_dir()
    default_image = str(example_dir / "test_inputs" / "example.jpg")
    default_audio = str(example_dir / "test_inputs" / "example.wav")

    if args.image is None:
        args.image = default_image
    else:
        args.image = resolve_input_path(args.image)

    if args.audio is None:
        args.audio = default_audio
    else:
        args.audio = resolve_input_path(args.audio)

    args.image, args.audio = download_sample_inputs_if_needed(args.image, args.audio)

    output_path = Path(resolve_output_path(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    weights_dir = Path(os.environ.get("MODEL_BASE", "/workspace/LiveTalk/pretrained_checkpoints"))
    checkpoints = ensure_model_weights(weights_dir, args.skip_download)

    run_inference(args, checkpoints, output_path)

    logging.info("Output saved to %s", output_path)


if __name__ == "__main__":
    main()
