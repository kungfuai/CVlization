#!/usr/bin/env python3
"""
SoulX-FlashTalk inference wrapper.

SoulX-FlashTalk is an audio-driven streaming talking-avatar model: given a
reference image and an audio clip, it generates a lip-synced talking-head
video. It is built on InfiniteTalk / Wan2.1, distilled to 4 sampling steps,
and generates the video chunk-by-chunk (autoregressive, motion-frame
conditioning) so it supports arbitrarily long clips.

This script wraps the upstream `generate_video.py` entry point: it resolves
input/output paths, lazily downloads the model weights from HuggingFace, runs
the upstream generator, and copies the final muxed video to the output path.

Usage:
    python predict.py --audio input.wav --image reference.png --output output.mp4
"""
import os
import sys
import logging
import warnings

# Suppress verbose logging by default (re-enabled with --verbose)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
for _name in ["transformers", "diffusers", "torch", "urllib3"]:
    logging.getLogger(_name).setLevel(logging.ERROR)
logger = logging.getLogger("soulx_flashtalk")

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

# CVL dual-mode execution support
try:
    from cvlization.paths import resolve_input_path, resolve_output_path
    CVL_AVAILABLE = True
except ImportError:
    CVL_AVAILABLE = False

    def resolve_input_path(p):
        return p

    def resolve_output_path(p):
        return p

# Upstream repo location (cloned into the image at build time)
SOULX_DIR = Path(os.environ.get("SOULX_DIR", "/workspace/SoulX-FlashTalk"))

# Model repositories on HuggingFace
HF_DIT_MODEL = "Soul-AILab/SoulX-FlashTalk-14B"
HF_WAV2VEC = "TencentGameMate/chinese-wav2vec2-base"

# Bundled sample inputs (shipped inside the upstream repo)
DEFAULT_IMAGE = SOULX_DIR / "examples" / "man.png"
DEFAULT_AUDIO = SOULX_DIR / "examples" / "cantonese_16k.wav"

DEFAULT_PROMPT = (
    "A person is talking. Only the foreground characters are moving, "
    "the background remains static."
)


def parse_args():
    parser = argparse.ArgumentParser(description="SoulX-FlashTalk avatar video generation")
    parser.add_argument("--audio", type=str, default=None,
                        help="Input audio file (WAV). Defaults to the bundled sample.")
    parser.add_argument("--image", type=str, default=None,
                        help="Reference image (PNG/JPG). Defaults to the bundled sample.")
    parser.add_argument("--output", type=str, default="output.mp4",
                        help="Output video path")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                        help="Text prompt describing the scene")
    parser.add_argument("--seed", type=int, default=9999, help="Random seed")
    parser.add_argument("--audio-encode-mode", choices=["stream", "once"], default="stream",
                        help="stream: encode audio per chunk; once: encode all audio together")
    parser.add_argument("--no-cpu-offload", action="store_true",
                        help="Disable CPU offload. Offload is ON by default and is "
                             "required to fit a 40-48GB GPU; disabling it needs >64GB VRAM.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging from the underlying frameworks")
    return parser.parse_args()


def download_models():
    """Lazily download model weights from HuggingFace; return local cache paths."""
    from huggingface_hub import snapshot_download

    logger.info(f"Resolving DiT weights ({HF_DIT_MODEL})...")
    dit_dir = snapshot_download(repo_id=HF_DIT_MODEL)
    logger.info(f"  -> {dit_dir}")

    logger.info(f"Resolving wav2vec weights ({HF_WAV2VEC})...")
    wav2vec_dir = snapshot_download(repo_id=HF_WAV2VEC)
    logger.info(f"  -> {wav2vec_dir}")

    return dit_dir, wav2vec_dir


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        for _name in ["transformers", "diffusers", "torch"]:
            logging.getLogger(_name).setLevel(logging.INFO)
        os.environ["TRANSFORMERS_VERBOSITY"] = "info"

    import torch
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. SoulX-FlashTalk requires an NVIDIA GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")

    cpu_offload = not args.no_cpu_offload
    if cpu_offload and gpu_mem < 38:
        logger.warning(f"GPU has {gpu_mem:.1f}GB VRAM; SoulX-FlashTalk needs ~40GB even "
                       "with CPU offload.")
    if not cpu_offload and gpu_mem < 64:
        logger.warning(f"GPU has {gpu_mem:.1f}GB VRAM; running without CPU offload needs "
                       ">64GB. Consider dropping --no-cpu-offload.")

    # Resolve inputs: a user-supplied path goes through CVL resolution; if omitted,
    # fall back to the sample bundled inside the upstream repo.
    image = resolve_input_path(args.image) if args.image else str(DEFAULT_IMAGE)
    audio = resolve_input_path(args.audio) if args.audio else str(DEFAULT_AUDIO)
    for label, path in [("image", image), ("audio", audio)]:
        if not Path(path).exists():
            logger.error(f"Input {label} not found: {path}")
            sys.exit(1)

    dit_dir, wav2vec_dir = download_models()

    print("=" * 60)
    print("SoulX-FlashTalk - Avatar Video Generation")
    print("=" * 60)
    print(f"Image:        {image}")
    print(f"Audio:        {audio}")
    print(f"Output:       {args.output}")
    print(f"Encode mode:  {args.audio_encode_mode}")
    print(f"CPU offload:  {cpu_offload}")
    print("=" * 60)

    # Upstream save_video() derives a temp (silent) path by stripping the literal
    # "res_" from the save_file basename, then ffmpeg-muxes audio into save_file.
    # The save_file basename MUST start with "res_" so temp != final; the parent
    # directory must NOT contain the substring "res_".
    out_dir = Path(tempfile.mkdtemp(prefix="soulx_"))
    save_file = out_dir / "res_soulx_output.mp4"

    cmd = [
        sys.executable, "generate_video.py",
        "--ckpt_dir", dit_dir,
        "--wav2vec_dir", wav2vec_dir,
        "--cond_image", image,
        "--audio_path", audio,
        "--input_prompt", args.prompt,
        "--base_seed", str(args.seed),
        "--audio_encode_mode", args.audio_encode_mode,
        "--save_file", str(save_file),
    ]
    if cpu_offload:
        cmd.append("--cpu_offload")

    # generate_video.py loads flash_talk/configs/infer_params.yaml via a relative
    # path, so it must run with cwd set to the upstream repo root.
    logger.info("Running SoulX-FlashTalk generator (this can take several minutes)...")
    subprocess.run(cmd, cwd=str(SOULX_DIR), check=True)

    if not save_file.exists():
        logger.error(f"Generation finished but no output was produced at {save_file}")
        sys.exit(1)

    output_path = Path(resolve_output_path(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(save_file), str(output_path))
    shutil.rmtree(out_dir, ignore_errors=True)

    logger.info(f"Video saved to: {output_path}")
    logger.info("Inference complete!")


if __name__ == "__main__":
    main()
