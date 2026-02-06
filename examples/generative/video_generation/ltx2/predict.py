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
import math
from pathlib import Path

import torch
import torchaudio
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

# LTX imports
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.helpers import cleanup_memory
from ltx_pipelines.utils.media_io import encode_video, load_video_conditioning
from ltx_pipelines.utils.constants import (
    AUDIO_SAMPLE_RATE,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_FRAMES,
)
from ltx_core.conditioning import AudioConditionByLatentSequence, VideoConditionByLatentIndex
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import AudioProcessor
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.types import AudioLatentShape, VideoPixelShape

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
SAMPLE_AUDIO_REPO = "zzsi/cvl"
SAMPLE_AUDIO_PATH = "ltx2/ltx2_tts_30s.wav"

# Default distilled sigma schedules (from ltx_pipelines)
DEFAULT_STAGE1_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
DEFAULT_STAGE2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]


def generate_sigma_schedule(num_steps: int, sigma_max: float = 1.0, sigma_min: float = 0.0) -> list[float]:
    """Generate a sigma schedule for diffusion denoising.

    EXPERIMENTAL: This generates linear sigma schedules for arbitrary step counts.
    The distilled model was trained on specific sigma values, so custom schedules
    may produce suboptimal results. Use at your own risk.

    Args:
        num_steps: Number of denoising steps (will produce num_steps + 1 sigma values)
        sigma_max: Starting sigma (noise level), default 1.0
        sigma_min: Ending sigma (clean), default 0.0

    Returns:
        List of sigma values from sigma_max to sigma_min
    """
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    # Linear schedule from sigma_max to sigma_min
    sigmas = [sigma_max - (sigma_max - sigma_min) * i / num_steps for i in range(num_steps + 1)]
    return sigmas


def get_sigma_schedule(num_steps: int | None, default_sigmas: list[float], sigma_max: float = None) -> list[float] | None:
    """Get sigma schedule - returns None to use default, or generates custom schedule.

    Args:
        num_steps: Number of steps, or None to use default
        default_sigmas: The default sigma schedule to use if num_steps is None
        sigma_max: Starting sigma for custom schedules (uses default_sigmas[0] if None)

    Returns:
        None if using default, or list of sigma values for custom schedule
    """
    if num_steps is None:
        return None  # Use default schedule in pipeline

    default_steps = len(default_sigmas) - 1
    if num_steps == default_steps:
        return None  # Same as default, use built-in

    # Generate custom schedule
    if sigma_max is None:
        sigma_max = default_sigmas[0]
    return generate_sigma_schedule(num_steps, sigma_max=sigma_max, sigma_min=0.0)


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


def resolve_lora_path(path: str) -> str:
    """Resolve LoRA path, downloading from HuggingFace if needed.

    Supports:
    - Local file paths: /path/to/lora.safetensors
    - HuggingFace repo IDs: username/repo-name (auto-downloads .safetensors file)
    - HuggingFace repo IDs with filename: username/repo-name::file.safetensors
    - HuggingFace repo/filename: username/repo-name/path/file.safetensors
    """
    # If it's an existing local file, use it directly
    if os.path.exists(path):
        return path

    repo_id = path
    filename = None

    if "::" in path:
        repo_id, filename = path.split("::", 1)
    elif path.endswith((".safetensors", ".bin", ".pt")) and path.count("/") >= 2:
        parts = path.split("/")
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])

    # Check if it looks like a HuggingFace repo ID (contains "/")
    if "/" in repo_id:
        print(f"Downloading LoRA from HuggingFace: {repo_id}...")

        # Find the .safetensors file in the repo if not specified
        try:
            if filename is None:
                files = list_repo_files(repo_id=repo_id)
                safetensor_files = [f for f in files if f.endswith(".safetensors")]

                if not safetensor_files:
                    raise ValueError(f"No .safetensors file found in {repo_id}")

                # Use the first (or only) safetensors file
                filename = safetensor_files[0]
                if len(safetensor_files) > 1:
                    print(f"  Multiple LoRA files found, using: {filename}")

            return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=get_cache_dir())
        except Exception as e:
            print(f"Warning: Failed to download LoRA from {path}: {e}")
            return path

    return path


def download_sample_audio() -> str:
    """Download the bundled sample audio from HuggingFace datasets."""
    print(f"Downloading sample audio from {SAMPLE_AUDIO_REPO}:{SAMPLE_AUDIO_PATH}...")
    return hf_hub_download(
        repo_id=SAMPLE_AUDIO_REPO,
        filename=SAMPLE_AUDIO_PATH,
        repo_type="dataset",
        cache_dir=get_cache_dir(),
    )


def parse_loras(lora_args: list) -> list:
    """Parse LoRA arguments into LoraPathStrengthAndSDOps objects."""
    if not lora_args:
        return []

    loras = []
    for lora_spec in lora_args:
        path = resolve_lora_path(lora_spec[0])
        strength = float(lora_spec[1]) if len(lora_spec) > 1 else 1.0
        loras.append(LoraPathStrengthAndSDOps(path, strength, LTXV_LORA_COMFY_RENAMING_MAP))
    return loras


def get_audio_duration_seconds(audio_path: str) -> float:
    try:
        info = torchaudio.info(audio_path)
        if info.num_frames and info.sample_rate:
            return info.num_frames / info.sample_rate
    except Exception:
        pass

    waveform, waveform_sample_rate = torchaudio.load(audio_path)
    if waveform.numel() == 0:
        return 0.0
    return waveform.shape[-1] / waveform_sample_rate


def derive_num_frames_from_audio(audio_path: str, frame_rate: float) -> int | None:
    if frame_rate <= 0:
        return None

    duration = get_audio_duration_seconds(audio_path)
    if duration <= 0:
        return None

    raw_frames = max(1, int(math.ceil(duration * frame_rate)))
    return ((raw_frames - 1 + 7) // 8) * 8 + 1


def build_audio_conditionings(
    audio_path: str | None,
    audio_strength: float,
    audio_offset_seconds: float,
    model_ledger,
    num_frames: int,
    frame_rate: float,
    device: torch.device,
    dtype: torch.dtype,
) -> list | None:
    if not audio_path:
        return None

    strength = max(0.0, min(1.0, float(audio_strength)))
    if strength <= 0.0:
        return None

    waveform, waveform_sample_rate = torchaudio.load(audio_path)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)

    if audio_offset_seconds:
        offset_samples = int(round(audio_offset_seconds * waveform_sample_rate))
        if offset_samples > 0:
            pad = torch.zeros(
                (waveform.shape[0], waveform.shape[1], offset_samples),
                dtype=waveform.dtype,
            )
            waveform = torch.cat([pad, waveform], dim=2)
        elif offset_samples < 0:
            offset_samples = abs(offset_samples)
            if offset_samples >= waveform.shape[2]:
                waveform = torch.zeros((waveform.shape[0], waveform.shape[1], 1), dtype=waveform.dtype)
            else:
                waveform = waveform[:, :, offset_samples:]

    audio_encoder = model_ledger.audio_encoder()
    target_channels = int(getattr(audio_encoder, "in_channels", waveform.shape[1]))
    if target_channels <= 0:
        target_channels = waveform.shape[1]

    if waveform.shape[1] != target_channels:
        if waveform.shape[1] == 1 and target_channels > 1:
            waveform = waveform.repeat(1, target_channels, 1)
        elif target_channels == 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        else:
            waveform = waveform[:, :target_channels, :]
            if waveform.shape[1] < target_channels:
                pad_channels = target_channels - waveform.shape[1]
                pad = torch.zeros(
                    (waveform.shape[0], pad_channels, waveform.shape[2]),
                    dtype=waveform.dtype,
                )
                waveform = torch.cat([waveform, pad], dim=1)

    audio_processor = AudioProcessor(
        sample_rate=audio_encoder.sample_rate,
        mel_bins=audio_encoder.mel_bins,
        mel_hop_length=audio_encoder.mel_hop_length,
        n_fft=audio_encoder.n_fft,
    )
    waveform = waveform.to(device="cpu", dtype=torch.float32)
    audio_processor = audio_processor.to(waveform.device)
    mel = audio_processor.waveform_to_mel(waveform, waveform_sample_rate)
    audio_params = next(audio_encoder.parameters(), None)
    audio_device = audio_params.device if audio_params is not None else device
    audio_dtype = audio_params.dtype if audio_params is not None else dtype
    mel = mel.to(device=audio_device, dtype=audio_dtype)
    with torch.inference_mode():
        audio_latent = audio_encoder(mel)

    audio_downsample = getattr(
        getattr(audio_encoder, "patchifier", None),
        "audio_latent_downsample_factor",
        4,
    )
    target_shape = AudioLatentShape.from_video_pixel_shape(
        VideoPixelShape(
            batch=audio_latent.shape[0],
            frames=int(num_frames),
            width=1,
            height=1,
            fps=float(frame_rate),
        ),
        channels=audio_latent.shape[1],
        mel_bins=audio_latent.shape[3],
        sample_rate=audio_encoder.sample_rate,
        hop_length=audio_encoder.mel_hop_length,
        audio_latent_downsample_factor=audio_downsample,
    )
    target_frames = target_shape.frames
    if audio_latent.shape[2] < target_frames:
        pad_frames = target_frames - audio_latent.shape[2]
        pad = torch.zeros(
            (audio_latent.shape[0], audio_latent.shape[1], pad_frames, audio_latent.shape[3]),
            device=audio_latent.device,
            dtype=audio_latent.dtype,
        )
        audio_latent = torch.cat([audio_latent, pad], dim=2)
    elif audio_latent.shape[2] > target_frames:
        audio_latent = audio_latent[:, :, :target_frames, :]

    audio_latent = audio_latent.to(device=device, dtype=dtype)
    del audio_encoder
    cleanup_memory()

    return [AudioConditionByLatentSequence(audio_latent, strength)]


def build_prefix_video_conditionings(
    prefix_video_path: str | None,
    prefix_frames: int,
    prefix_strength: float,
    model_ledger,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list:
    if not prefix_video_path or prefix_frames <= 0:
        return []

    video = load_video_conditioning(
        video_path=prefix_video_path,
        height=height,
        width=width,
        frame_cap=prefix_frames,
        dtype=dtype,
        device=device,
    )
    if video is None or video.shape[2] == 0:
        return []

    video_encoder = model_ledger.video_encoder()
    with torch.inference_mode():
        encoded_video = video_encoder(video)
    del video_encoder
    cleanup_memory()

    return [VideoConditionByLatentIndex(latent=encoded_video, strength=prefix_strength, latent_idx=0)]


def get_local_model_path(filename: str, repo_id: str = LTX2_REPO) -> str | None:
    """Check if model file exists in local/persistent storage.

    Checks these locations in order:
    1. Cerebrium persistent storage: /persistent-storage/.cache/huggingface/hub/...
    2. Standard HF cache: ~/.cache/huggingface/hub/...
    """
    # Convert repo_id to HF cache format (e.g., "Lightricks/LTX-2" -> "models--Lightricks--LTX-2")
    repo_dir = f"models--{repo_id.replace('/', '--')}"

    # Known revision for LTX-2 models
    revision = "1931de987c8e265eb64a9123227b903754e3cc68"

    # Paths to check
    locations = [
        # Cerebrium persistent storage
        f"/persistent-storage/.cache/huggingface/hub/{repo_dir}/snapshots/{revision}/{filename}",
        # Also check blobs with original filename (legacy upload location)
        f"/persistent-storage/.cache/huggingface/hub/{repo_dir}/blobs/{filename}",
        # Standard HF cache
        os.path.expanduser(f"~/.cache/huggingface/hub/{repo_dir}/snapshots/{revision}/{filename}"),
    ]

    for path in locations:
        if os.path.exists(path):
            print(f"Found local model: {path}")
            return path

    return None


def get_local_gemma_path() -> str | None:
    """Check if Gemma encoder exists in local/persistent storage."""
    repo_dir = f"models--{GEMMA_REPO.replace('/', '--')}"

    locations = [
        f"/persistent-storage/.cache/huggingface/hub/{repo_dir}",
        os.path.expanduser(f"~/.cache/huggingface/hub/{repo_dir}"),
    ]

    for path in locations:
        # Check if it's a valid snapshot directory
        snapshots_dir = os.path.join(path, "snapshots")
        if os.path.isdir(snapshots_dir):
            # Get first available snapshot
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                snapshot_path = os.path.join(snapshots_dir, snapshots[0])
                print(f"Found local Gemma encoder: {snapshot_path}")
                return snapshot_path

    return None


def get_model_paths(pipeline: str, fp8: bool) -> dict:
    """Get model paths - check local/persistent storage first, then download."""
    paths = {}

    # Select checkpoint based on pipeline and precision
    if pipeline == "distilled":
        checkpoint_file = CHECKPOINT_DISTILLED_FP8 if fp8 else CHECKPOINT_DISTILLED
    else:  # full (dev) pipeline
        checkpoint_file = CHECKPOINT_DEV_FP8 if fp8 else CHECKPOINT_DEV

    # Check local storage first, fall back to download
    local_checkpoint = get_local_model_path(checkpoint_file)
    paths["checkpoint"] = local_checkpoint if local_checkpoint else download_model_file(checkpoint_file)

    local_upsampler = get_local_model_path(SPATIAL_UPSAMPLER)
    paths["spatial_upsampler"] = local_upsampler if local_upsampler else download_model_file(SPATIAL_UPSAMPLER)

    local_gemma = get_local_gemma_path()
    paths["gemma"] = local_gemma if local_gemma else download_gemma_encoder()

    # full pipeline requires distilled LoRA
    if pipeline == "full":
        local_lora = get_local_model_path(DISTILLED_LORA)
        paths["distilled_lora"] = local_lora if local_lora else download_model_file(DISTILLED_LORA)

    return paths


def run_distilled_pipeline(
    prompt: str,
    output_path: str,
    model_paths: dict,
    image_path: str = None,
    audio_path: str | None = None,
    audio_strength: float = 1.0,
    audio_offset_seconds: float = 0.0,
    prefix_video_path: str | None = None,
    prefix_frames: int = 0,
    prefix_strength: float = 1.0,
    seed: int = 10,
    height: int = 1024,
    width: int = 1536,
    num_frames: int = 121,
    frame_rate: float = 24.0,
    fp8: bool = True,
    enhance_prompt: bool = False,
    loras: list = None,
    stage1_steps: int | None = None,
    stage2_steps: int | None = None,
):
    """Run the distilled pipeline (faster, default 8+3 steps).

    Args:
        stage1_steps: EXPERIMENTAL - Override stage 1 denoising steps (default 8).
                      Custom step counts use linear sigma schedules and may produce
                      suboptimal results since the model was trained on specific sigmas.
        stage2_steps: EXPERIMENTAL - Override stage 2 denoising steps (default 3).
    """
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

    audio_conditionings = build_audio_conditionings(
        audio_path=audio_path,
        audio_strength=audio_strength,
        audio_offset_seconds=audio_offset_seconds,
        model_ledger=pipeline.model_ledger,
        num_frames=num_frames,
        frame_rate=frame_rate,
        device=pipeline.device,
        dtype=pipeline.dtype,
    )
    stage_1_prefix = build_prefix_video_conditionings(
        prefix_video_path=prefix_video_path,
        prefix_frames=prefix_frames,
        prefix_strength=prefix_strength,
        model_ledger=pipeline.model_ledger,
        height=height // 2,
        width=width // 2,
        device=pipeline.device,
        dtype=pipeline.dtype,
    )
    stage_2_prefix = build_prefix_video_conditionings(
        prefix_video_path=prefix_video_path,
        prefix_frames=prefix_frames,
        prefix_strength=prefix_strength,
        model_ledger=pipeline.model_ledger,
        height=height,
        width=width,
        device=pipeline.device,
        dtype=pipeline.dtype,
    )

    # Get sigma schedules (None = use default)
    stage1_sigmas = get_sigma_schedule(stage1_steps, DEFAULT_STAGE1_SIGMAS)
    stage2_sigmas = get_sigma_schedule(stage2_steps, DEFAULT_STAGE2_SIGMAS)

    steps_info = f"{stage1_steps or 8}+{stage2_steps or 3} steps"
    if stage1_steps or stage2_steps:
        steps_info += " (EXPERIMENTAL)"
    print(f"Generating video: {width}x{height}, {num_frames} frames, {steps_info}...")
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
            audio_conditionings=audio_conditionings,
            video_conditionings_stage1=stage_1_prefix,
            video_conditionings_stage2=stage_2_prefix,
            stage1_sigmas=stage1_sigmas,
            stage2_sigmas=stage2_sigmas,
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
    audio_path: str | None = None,
    audio_strength: float = 1.0,
    audio_offset_seconds: float = 0.0,
    prefix_video_path: str | None = None,
    prefix_frames: int = 0,
    prefix_strength: float = 1.0,
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

    audio_conditionings = build_audio_conditionings(
        audio_path=audio_path,
        audio_strength=audio_strength,
        audio_offset_seconds=audio_offset_seconds,
        model_ledger=pipeline.stage_1_model_ledger,
        num_frames=num_frames,
        frame_rate=frame_rate,
        device=pipeline.device,
        dtype=pipeline.dtype,
    )
    stage_1_prefix = build_prefix_video_conditionings(
        prefix_video_path=prefix_video_path,
        prefix_frames=prefix_frames,
        prefix_strength=prefix_strength,
        model_ledger=pipeline.stage_1_model_ledger,
        height=height // 2,
        width=width // 2,
        device=pipeline.device,
        dtype=pipeline.dtype,
    )
    stage_2_prefix = build_prefix_video_conditionings(
        prefix_video_path=prefix_video_path,
        prefix_frames=prefix_frames,
        prefix_strength=prefix_strength,
        model_ledger=pipeline.stage_1_model_ledger,
        height=height,
        width=width,
        device=pipeline.device,
        dtype=pipeline.dtype,
    )

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
            audio_conditionings=audio_conditionings,
            video_conditionings_stage1=stage_1_prefix,
            video_conditionings_stage2=stage_2_prefix,
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
        choices=["distilled", "full", "two_stage"],
        default="distilled",
        help="Pipeline mode: distilled (fast, 8+3 steps) or full (quality, 40 steps). "
             "'two_stage' is deprecated and maps to 'full'.",
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
        "--audio",
        type=str,
        default=None,
        help="Optional input audio for audio-latent conditioning",
    )
    parser.add_argument(
        "--use-sample-audio",
        action="store_true",
        help="Use the bundled sample audio (downloads lazily) for conditioning",
    )
    parser.add_argument(
        "--audio-strength",
        type=float,
        default=1.0,
        help="Strength for audio conditioning (0.0 to 1.0)",
    )
    parser.add_argument(
        "--audio-align-prefix",
        action="store_true",
        help="Pad audio by the prefix duration to align continuation audio",
    )
    parser.add_argument(
        "--prefix-video",
        type=str,
        default=None,
        help="Optional prefix video for continuation conditioning",
    )
    parser.add_argument(
        "--prefix-frames",
        type=int,
        default=0,
        help="Number of prefix frames to condition from the prefix video",
    )
    parser.add_argument(
        "--prefix-strength",
        type=float,
        default=1.0,
        help="Strength for prefix video conditioning (0.0 to 1.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--list-lora-files",
        type=str,
        default=None,
        help="List available .safetensors files in a HuggingFace LoRA repo and exit",
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
        default=None,
        help="Number of frames (must satisfy (F-1) %% 8 == 0). "
             "Defaults to audio length when audio is provided.",
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
        help="Denoising steps (full pipeline only)",
    )
    parser.add_argument(
        "--cfg-guidance-scale",
        type=float,
        default=4.0,
        help="CFG guidance scale (full pipeline only)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt (full pipeline only, uses default if empty)",
    )

    # Distilled pipeline step override (EXPERIMENTAL)
    parser.add_argument(
        "--stage1-steps",
        type=int,
        default=None,
        help="EXPERIMENTAL: Override stage 1 denoising steps for distilled pipeline (default 8). "
             "Custom step counts may produce suboptimal results.",
    )
    parser.add_argument(
        "--stage2-steps",
        type=int,
        default=None,
        help="EXPERIMENTAL: Override stage 2 denoising steps for distilled pipeline (default 3). "
             "Custom step counts may produce suboptimal results.",
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

    if args.list_lora_files:
        try:
            files = list_repo_files(repo_id=args.list_lora_files)
        except Exception as exc:
            print(f"Error: failed to list files for {args.list_lora_files}: {exc}")
            sys.exit(1)

        safetensor_files = [f for f in files if f.endswith(".safetensors")]
        if not safetensor_files:
            print(f"No .safetensors files found in {args.list_lora_files}")
        else:
            for filename in safetensor_files:
                print(filename)
        sys.exit(0)

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

    audio_path = None
    if args.audio and args.use_sample_audio:
        print("Error: --audio and --use-sample-audio are mutually exclusive.")
        sys.exit(1)

    if args.audio:
        audio_path = resolve_input_path(args.audio)
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found: {audio_path}")
            sys.exit(1)
    elif args.use_sample_audio:
        try:
            audio_path = download_sample_audio()
        except Exception as exc:
            print(f"Error: Failed to download sample audio: {exc}")
            sys.exit(1)

    num_frames = args.num_frames
    if num_frames is None and audio_path:
        num_frames = derive_num_frames_from_audio(audio_path, args.frame_rate)
        if num_frames is not None:
            print(f"Derived num_frames={num_frames} from audio length.")
    if num_frames is None:
        num_frames = DEFAULT_NUM_FRAMES

    prefix_video_path = None
    if args.prefix_video:
        prefix_video_path = resolve_input_path(args.prefix_video)
        if not os.path.exists(prefix_video_path):
            print(f"Error: Prefix video file not found: {prefix_video_path}")
            sys.exit(1)
        if args.prefix_frames <= 0:
            print("Error: --prefix-frames must be > 0 when --prefix-video is set.")
            sys.exit(1)
        if args.prefix_frames > num_frames:
            print("Error: --prefix-frames cannot exceed --num-frames.")
            sys.exit(1)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fp8 = not args.no_fp8
    loras = parse_loras(args.lora)

    pipeline = args.pipeline
    if pipeline == "two_stage":
        print("Warning: --pipeline two_stage is deprecated; use --pipeline full instead.")
        pipeline = "full"

    print(f"Pipeline: {pipeline}")
    print(f"FP8 mode: {'enabled' if fp8 else 'disabled'}")
    if loras:
        print(f"LoRAs: {len(loras)}")
    print(f"Output: {output_path}")

    # Download models
    print("\nPreparing models...")
    model_paths = get_model_paths(pipeline, fp8)

    # Run pipeline
    audio_offset_seconds = 0.0
    if args.audio_align_prefix and args.prefix_frames > 0:
        audio_offset_seconds = float(args.prefix_frames) / float(args.frame_rate)

    if pipeline == "distilled":
        run_distilled_pipeline(
            prompt=args.prompt,
            output_path=output_path,
            model_paths=model_paths,
            image_path=image_path,
            audio_path=audio_path,
            audio_strength=args.audio_strength,
            audio_offset_seconds=audio_offset_seconds,
            prefix_video_path=prefix_video_path,
            prefix_frames=args.prefix_frames,
            prefix_strength=args.prefix_strength,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=num_frames,
            frame_rate=args.frame_rate,
            fp8=fp8,
            enhance_prompt=args.enhance_prompt,
            loras=loras,
            stage1_steps=args.stage1_steps,
            stage2_steps=args.stage2_steps,
        )
    else:
        run_two_stage_pipeline(
            prompt=args.prompt,
            output_path=output_path,
            model_paths=model_paths,
            image_path=image_path,
            audio_path=audio_path,
            audio_strength=args.audio_strength,
            audio_offset_seconds=audio_offset_seconds,
            prefix_video_path=prefix_video_path,
            prefix_frames=args.prefix_frames,
            prefix_strength=args.prefix_strength,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=num_frames,
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
