#!/usr/bin/env python3
"""
LongCat-Video-Avatar: Audio-driven character animation using LongCat-Video.

Supports two modes:
- single: Animate one person with one audio track
- multi: Animate two people with separate audio tracks (parallel or conversation)

Model: meituan-longcat/LongCat-Video-Avatar (13.6B parameters)
License: Apache 2.0

References:
- https://github.com/meituan-longcat/LongCat-Video
- https://huggingface.co/meituan-longcat/LongCat-Video-Avatar
"""

import os
import sys
import logging
import warnings

# Suppress verbose logging by default
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")

warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.ERROR)
for logger_name in ["transformers", "diffusers", "torch", "triton"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import numpy as np
from PIL import Image
import soundfile as sf
import imageio


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
        return Path.cwd()

    def get_output_dir():
        return Path.cwd()

    def resolve_input_path(path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return Path.cwd() / p

    def resolve_output_path(path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return Path.cwd() / p


# Model IDs
BASE_MODEL_ID = "meituan-longcat/LongCat-Video"
AVATAR_MODEL_ID = "meituan-longcat/LongCat-Video-Avatar"
MODEL_CACHE_DIR = Path("/models")
SAMPLE_RATE = 16000


def get_model_paths():
    """Get the paths to the model directories."""
    from huggingface_hub import snapshot_download

    cache_dir = MODEL_CACHE_DIR if MODEL_CACHE_DIR.exists() else None

    # Download base model (for tokenizer, text_encoder, vae, scheduler)
    print(f"Downloading base model: {BASE_MODEL_ID}...")
    base_path = snapshot_download(
        repo_id=BASE_MODEL_ID,
        cache_dir=cache_dir,
        allow_patterns=["tokenizer/*", "text_encoder/*", "vae/*", "scheduler/*", "*.json"]
    )

    # Download avatar model (for transformer and audio)
    print(f"Downloading avatar model: {AVATAR_MODEL_ID}...")
    avatar_path = snapshot_download(
        repo_id=AVATAR_MODEL_ID,
        cache_dir=cache_dir
    )

    return base_path, avatar_path


def load_pipeline(device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Load the LongCat-Video-Avatar pipeline."""
    from transformers import AutoTokenizer, UMT5EncoderModel, Wav2Vec2FeatureExtractor
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
    from longcat_video.audio_process.wav2vec2 import Wav2Vec2ModelWrapper
    from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline

    # Get model paths
    base_path, avatar_path = get_model_paths()

    print("Loading models...")

    # Load tokenizer and text encoder from BASE model
    tokenizer = AutoTokenizer.from_pretrained(
        base_path,
        subfolder="tokenizer",
        local_files_only=True
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        base_path,
        subfolder="text_encoder",
        torch_dtype=dtype,
        local_files_only=True
    ).to(device)

    # Load VAE from BASE model
    vae = AutoencoderKLWan.from_pretrained(
        base_path,
        subfolder="vae",
        torch_dtype=dtype,
        local_files_only=True
    ).to(device)

    # Load scheduler from BASE model
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        base_path,
        subfolder="scheduler",
        local_files_only=True
    )

    # Load DiT transformer from AVATAR model (avatar_single subfolder)
    # cp_split_hw is for context parallelism - set to (1, 1) for single GPU
    dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(
        avatar_path,
        subfolder="avatar_single",
        cp_split_hw=(1, 1),
        torch_dtype=dtype,
        local_files_only=True
    ).to(device)

    # Load audio encoder from AVATAR model (chinese-wav2vec2-base directory)
    wav2vec_path = os.path.join(avatar_path, "chinese-wav2vec2-base")
    audio_encoder = Wav2Vec2ModelWrapper(wav2vec_path, device=device).to(device)
    audio_encoder.feature_extractor._freeze_parameters()

    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        wav2vec_path,
        local_files_only=True
    )

    pipeline = LongCatVideoAvatarPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
        audio_encoder=audio_encoder,
        wav2vec_feature_extractor=wav2vec_feature_extractor
    )

    print("Model loaded successfully!")
    return pipeline


def load_audio(audio_path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load and resample audio to target sample rate."""
    import librosa

    audio, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    return audio.astype(np.float32)


def create_face_mask(image: Image.Image, person_index: int = 0) -> Optional[torch.Tensor]:
    """
    Create a simple face mask for multi-person mode.
    For multi mode, we split the image horizontally.
    Returns a mask tensor of shape [3, H, W].
    """
    w, h = image.size
    mask = np.zeros((h, w, 3), dtype=np.float32)

    if person_index == 0:
        # Left half
        mask[:, :w//2, :] = 1.0
    else:
        # Right half
        mask[:, w//2:, :] = 1.0

    # Convert to tensor [3, H, W]
    mask_tensor = torch.from_numpy(mask).permute(2, 0, 1)
    return mask_tensor


def prepare_audio_embedding(full_audio_emb: torch.Tensor, num_frames: int, audio_stride: int = 2, device: str = "cuda"):
    """
    Prepare audio embedding for the DiT model.

    The DiT expects audio embedding with specific indexing based on frame-audio alignment.
    """
    # Create indices for audio-frame alignment (5 neighbors: 2 before, current, 2 after)
    indices = torch.arange(2 * 2 + 1) - 2  # [-2, -1, 0, 1, 2]

    # Audio indices for each frame
    audio_start_idx = 0
    audio_end_idx = audio_start_idx + audio_stride * num_frames

    # Center indices with neighbors
    center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)

    # Index the full audio embedding and add batch dimension
    audio_emb = full_audio_emb[center_indices][None, ...].to(device)

    return audio_emb


def run_single_mode(
    pipeline,
    image_path: Path,
    audio_path: Path,
    prompt: str,
    negative_prompt: str,
    output_path: Path,
    resolution: str = "480p",
    num_frames: int = 93,
    num_segments: int | None = None,
    num_steps: int = 50,
    text_guidance: float = 4.0,
    audio_guidance: float = 4.0,
    offload_kv_cache: bool = False,
    seed: int = 42,
    device: str = "cuda",
    verbose: bool = False
):
    """Run single-person avatar generation with optional multi-segment long video support."""

    print(f"Mode: single")
    print(f"Image: {image_path}")
    print(f"Audio: {audio_path}")
    # Constants for audio-video alignment
    save_fps = 16
    audio_stride = 2
    num_cond_frames = 13  # Overlap frames between segments

    # Load inputs
    image = Image.open(image_path).convert("RGB")
    audio = load_audio(audio_path)

    audio_duration = len(audio) / SAMPLE_RATE

    if verbose:
        print(f"Image size: {image.size}")
        print(f"Audio duration: {audio_duration:.2f}s")

    if num_segments is None:
        frames_needed = math.ceil(audio_duration * save_fps)
        if frames_needed <= num_frames:
            num_segments = 1
        else:
            overlap = num_frames - num_cond_frames
            num_segments = math.ceil((frames_needed - num_frames) / overlap) + 1

    if num_segments > 1:
        print(f"Segments: {num_segments} (long video mode)")

    # Calculate total duration for all segments
    # First segment: num_frames, subsequent segments add (num_frames - num_cond_frames) each
    total_new_frames = num_frames + (num_segments - 1) * (num_frames - num_cond_frames)
    generate_duration = total_new_frames / save_fps

    source_duration = len(audio) / SAMPLE_RATE
    added_sample_nums = math.ceil((generate_duration - source_duration) * SAMPLE_RATE)
    if added_sample_nums > 0:
        audio = np.append(audio, np.zeros(added_sample_nums, dtype=np.float32))

    if verbose:
        print(f"Total frames to generate: {total_new_frames}")
        print(f"Total duration: {generate_duration:.2f}s")

    # Get full audio embedding for entire duration
    full_audio_emb = pipeline.get_audio_embedding(
        audio,
        fps=save_fps * audio_stride,
        device=device,
        sample_rate=SAMPLE_RATE
    )

    # Set seed
    generator = torch.Generator(device=device).manual_seed(seed)

    # === SEGMENT 1: Initial generation with ai2v ===
    print(f"Generating segment 1/{num_segments}...")

    # Prepare audio embedding for first segment
    audio_emb = prepare_audio_embedding(full_audio_emb, num_frames, audio_stride, device)

    if num_segments == 1:
        # Simple single-segment generation
        if offload_kv_cache:
            print("Note: --offload-kv-cache only applies to continuation segments.")

        output_video = pipeline.generate_ai2v(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=resolution,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            text_guidance_scale=text_guidance,
            audio_guidance_scale=audio_guidance,
            generator=generator,
            audio_emb=audio_emb,
            output_type="np"
        )
        return output_video

    # Multi-segment: need latent for continuation
    output_tuple = pipeline.generate_ai2v(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        resolution=resolution,
        num_frames=num_frames,
        num_inference_steps=num_steps,
        text_guidance_scale=text_guidance,
        audio_guidance_scale=audio_guidance,
        generator=generator,
        audio_emb=audio_emb,
        output_type="both"  # Returns (video, latent)
    )
    output, latent = output_tuple
    output = output[0]  # Remove batch dimension

    # Convert to PIL images for continuation
    video_frames = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
    current_video = [Image.fromarray(img) for img in video_frames]

    # Save reference latent from first frame for temporal consistency
    ref_latent = latent[:, :, :1].clone()

    # Collect all frames
    all_frames = list(current_video)

    # === SEGMENTS 2+: Continue with avc ===
    audio_start_idx = 0
    indices = torch.arange(2 * 2 + 1) - 2  # For audio embedding indexing

    # Get actual video dimensions from generated frames (may differ from requested resolution)
    width, height = current_video[0].size

    for segment_idx in range(1, num_segments):
        print(f"Generating segment {segment_idx + 1}/{num_segments}...")

        # Advance audio position (skip overlap frames worth of audio)
        audio_start_idx = audio_start_idx + audio_stride * (num_frames - num_cond_frames)
        audio_end_idx = audio_start_idx + audio_stride * num_frames

        # Prepare audio embedding for this segment
        center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
        audio_emb = full_audio_emb[center_indices][None, ...].to(device)

        # Generate continuation
        output_tuple = pipeline.generate_avc(
            video=current_video,
            video_latent=latent,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            num_inference_steps=num_steps,
            text_guidance_scale=text_guidance,
            audio_guidance_scale=audio_guidance,
            generator=generator,
            output_type="both",
            use_kv_cache=True,
            offload_kv_cache=offload_kv_cache,
            enhance_hf=True,
            audio_emb=audio_emb,
            ref_latent=ref_latent,
            ref_img_index=10,
            mask_frame_range=3
        )
        output, latent = output_tuple
        output = output[0]

        # Convert to PIL and update current_video
        new_frames = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        current_video = [Image.fromarray(img) for img in new_frames]

        # Append only new frames (skip overlap)
        all_frames.extend(current_video[num_cond_frames:])

        if verbose:
            print(f"  Segment {segment_idx + 1}: added {len(current_video) - num_cond_frames} new frames, total: {len(all_frames)}")

    # Convert back to numpy array format
    output_video = np.stack([np.array(f) for f in all_frames], axis=0)

    print(f"Long video complete: {len(all_frames)} frames ({len(all_frames)/save_fps:.1f}s)")

    return output_video


def run_multi_mode(
    pipeline,
    image_path: Path,
    audio1_path: Path,
    audio2_path: Path,
    audio_type: str,  # "para" (parallel) or "conv" (conversation)
    prompt: str,
    negative_prompt: str,
    output_path: Path,
    resolution: str = "480p",
    num_frames: int = 93,
    num_steps: int = 50,
    text_guidance: float = 4.0,
    audio_guidance: float = 4.0,
    seed: int = 42,
    device: str = "cuda",
    verbose: bool = False
):
    """Run multi-person avatar generation (two people)."""
    print(f"Mode: multi ({audio_type})")
    print(f"Image: {image_path}")
    print(f"Audio 1: {audio1_path}")
    print(f"Audio 2: {audio2_path}")

    # Constants for audio-video alignment
    import math
    save_fps = 16
    audio_stride = 2

    # Load inputs
    image = Image.open(image_path).convert("RGB")
    audio1 = load_audio(audio1_path)
    audio2 = load_audio(audio2_path)

    if verbose:
        print(f"Image size: {image.size}")
        print(f"Audio 1 duration: {len(audio1)/16000:.2f}s")
        print(f"Audio 2 duration: {len(audio2)/16000:.2f}s")

    # Calculate required audio duration and pad if necessary
    generate_duration = num_frames / save_fps
    for i, audio in enumerate([audio1, audio2]):
        source_duration = len(audio) / 16000
        added_sample_nums = math.ceil((generate_duration - source_duration) * 16000)
        if added_sample_nums > 0:
            if i == 0:
                audio1 = np.append(audio1, np.zeros(added_sample_nums, dtype=np.float32))
            else:
                audio2 = np.append(audio2, np.zeros(added_sample_nums, dtype=np.float32))

    # Get full audio embeddings for both persons
    full_audio_emb1 = pipeline.get_audio_embedding(
        audio1,
        fps=save_fps * audio_stride,
        device=device,
        sample_rate=16000
    )
    full_audio_emb2 = pipeline.get_audio_embedding(
        audio2,
        fps=save_fps * audio_stride,
        device=device,
        sample_rate=16000
    )

    # Prepare audio embeddings for DiT (both persons)
    audio_emb1 = prepare_audio_embedding(full_audio_emb1, num_frames, audio_stride, device)
    audio_emb2 = prepare_audio_embedding(full_audio_emb2, num_frames, audio_stride, device)

    # Stack audio embeddings: [2, T, 5, 12, 768] -> expected by multi-mode
    audio_emb = torch.cat([audio_emb1, audio_emb2], dim=0)

    # Create face masks for the two persons
    ref_target_masks = create_face_mask(image, person_index=0)
    ref_target_masks = ref_target_masks.to(device)

    # Set seed
    generator = torch.Generator(device=device).manual_seed(seed)

    # Generate video
    print(f"Generating video with {num_steps} steps, {num_frames} frames...")

    output_video = pipeline.generate_ai2v(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        resolution=resolution,
        num_frames=num_frames,
        num_inference_steps=num_steps,
        text_guidance_scale=text_guidance,
        audio_guidance_scale=audio_guidance,
        generator=generator,
        audio_emb=audio_emb,
        ref_target_masks=ref_target_masks,
        output_type="np"
    )

    return output_video


def save_video(video: np.ndarray, output_path: Path, fps: int = 15, audio_path: Optional[Path] = None):
    """Save video frames to MP4, optionally muxing with audio."""
    if video.ndim == 5:  # [B, T, H, W, C]
        video = video[0]  # Take first batch

    # Ensure uint8
    if video.dtype != np.uint8:
        video = (video * 255).clip(0, 255).astype(np.uint8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if audio_path is None:
        # No audio - just save video
        imageio.mimwrite(str(output_path), video, fps=fps)
    else:
        # Save video to temp file, then mux with audio
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_video = tmp.name

        imageio.mimwrite(temp_video, video, fps=fps)

        # Mux video with audio using ffmpeg
        print(f"Muxing audio from {audio_path}...")
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_video,
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            "-loglevel", "error",
            str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to mux audio: {e}")
            # Fall back to video-only
            Path(temp_video).rename(output_path)
        finally:
            if Path(temp_video).exists():
                Path(temp_video).unlink()

    print(f"Video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="LongCat-Video-Avatar: Audio-driven character animation"
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "multi"],
        default="single",
        help="Generation mode: single (one person) or multi (two persons)"
    )

    # Input arguments
    parser.add_argument(
        "--image",
        type=str,
        default="sample",
        help="Input image path (use 'sample' for default sample)"
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="sample",
        help="Audio file for person 1 (use 'sample' for default)"
    )
    parser.add_argument(
        "--audio2",
        type=str,
        default=None,
        help="Audio file for person 2 (multi mode only)"
    )
    parser.add_argument(
        "--audio-type",
        type=str,
        choices=["para", "conv"],
        default="para",
        help="Audio type for multi mode: para (parallel) or conv (conversation)"
    )

    # Generation parameters
    parser.add_argument(
        "--prompt",
        type=str,
        default="A person speaking naturally with clear lip movements and subtle head motion.",
        help="Text prompt describing the video"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="Blurry, distorted face, unnatural motion, low quality",
        help="Negative prompt"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Output video path"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        choices=["480p", "720p"],
        default="480p",
        help="Output resolution"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=93,
        help="Number of frames per segment"
    )
    parser.add_argument(
        "--num-segments",
        type=int,
        default=None,
        help="Number of segments for long video generation (defaults to auto based on audio length)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--text-guidance",
        type=float,
        default=4.0,
        help="Text guidance scale"
    )
    parser.add_argument(
        "--audio-guidance",
        type=float,
        default=4.0,
        help="Audio guidance scale"
    )
    parser.add_argument(
        "--offload-kv-cache",
        action="store_true",
        help="Offload KV cache to CPU to reduce VRAM usage (slower)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Re-enable verbose output if requested
    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)
        for logger_name in ["transformers", "diffusers", "torch", "triton"]:
            logging.getLogger(logger_name).setLevel(logging.INFO)
        os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

    # Resolve paths
    output_path = resolve_output_path(args.output)

    # Handle sample inputs
    sample_dir = Path("/workspace/local/examples")

    if args.image == "sample":
        # Use HuggingFace sample or local sample
        try:
            from huggingface_hub import hf_hub_download
            image_path = Path(hf_hub_download(
                repo_id="zzsi/cvl",
                filename="longcat_video_avatar/man.png",
                repo_type="dataset"
            ))
        except Exception:
            image_path = sample_dir / "man.png"
    else:
        image_path = resolve_input_path(args.image)

    if args.audio == "sample":
        try:
            from huggingface_hub import hf_hub_download
            audio_path = Path(hf_hub_download(
                repo_id="zzsi/cvl",
                filename="longcat_video_avatar/man.mp3",
                repo_type="dataset"
            ))
        except Exception:
            audio_path = sample_dir / "man.mp3"
    else:
        audio_path = resolve_input_path(args.audio)

    # Validate inputs
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    # For multi mode, validate audio2
    audio2_path = None
    if args.mode == "multi":
        if args.audio2 is None:
            print("Error: Multi mode requires --audio2 argument")
            sys.exit(1)
        audio2_path = resolve_input_path(args.audio2)
        if not audio2_path.exists():
            print(f"Error: Audio2 file not found: {audio2_path}")
            sys.exit(1)

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: Running on CPU. This will be very slow.")

    # Load pipeline
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipeline = load_pipeline(device=device, dtype=dtype)

    # Run generation
    if args.mode == "single":
        output_video = run_single_mode(
            pipeline=pipeline,
            image_path=image_path,
            audio_path=audio_path,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            output_path=output_path,
            resolution=args.resolution,
            num_frames=args.frames,
            num_segments=args.num_segments,
            num_steps=args.steps,
            text_guidance=args.text_guidance,
            audio_guidance=args.audio_guidance,
            offload_kv_cache=args.offload_kv_cache,
            seed=args.seed,
            device=device,
            verbose=args.verbose
        )
    else:
        output_video = run_multi_mode(
            pipeline=pipeline,
            image_path=image_path,
            audio1_path=audio_path,
            audio2_path=audio2_path,
            audio_type=args.audio_type,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            output_path=output_path,
            resolution=args.resolution,
            num_frames=args.frames,
            num_steps=args.steps,
            text_guidance=args.text_guidance,
            audio_guidance=args.audio_guidance,
            seed=args.seed,
            device=device,
            verbose=args.verbose
        )

    # Save output with audio
    # For multi mode, use the first audio track (could mix both in future)
    save_video(output_video, output_path, fps=16, audio_path=audio_path)

    print("Done!")


if __name__ == "__main__":
    main()
