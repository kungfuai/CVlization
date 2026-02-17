#!/usr/bin/env python3
"""
DOVE (NeurIPS 2025) one-step diffusion video super-resolution.

Adapted from DOVE inference_script.py for CVlization conventions.
Uses CogVideoX1.5-5B fine-tuned for single-step VSR.
"""
import os
import sys
import logging
import warnings
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

# Suppress verbose logging by default
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")

warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

logging.basicConfig(level=logging.ERROR)
for logger_name in ["transformers", "diffusers", "torch"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import numpy as np
from PIL import Image
import imageio.v3 as iio
import cv2
import torch
from tqdm import tqdm
from safetensors.torch import load_file

# Must import after torch to avoid potential segfaults (see decord issues)
import decord  # isort:skip

decord.bridge.set_bridge("torch")

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
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def resolve_input_path(path, base_dir):
        if os.path.isabs(path):
            return path
        return os.path.join(base_dir, path)

    def resolve_output_path(path, base_dir):
        if os.path.isabs(path):
            return path
        return os.path.join(base_dir, path)


# ---------------------------------------------------------------------------
# Video I/O helpers
# ---------------------------------------------------------------------------

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def is_video_file(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith(VIDEO_EXTS)


def read_video_frames(video_path: str) -> torch.Tensor:
    """Read video using OpenCV, return [F, C, H, W] float tensor in [0, 1]."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        frames.append(t)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")
    return torch.stack(frames)


def get_video_fps(video_path: str) -> int:
    """Get FPS from video file."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return max(1, int(round(fps))) if fps > 0 else 16


def preprocess_video(video_path: str) -> Tuple[torch.Tensor, int, int, int, Tuple]:
    """
    Load and preprocess video for DOVE inference.

    Returns:
        frames: [F, C, H, W] float tensor in [0, 1] (padded to align)
        pad_f: number of padded frames
        pad_h: height padding
        pad_w: width padding
        original_shape: (F, H, W, C) original dimensions
    """
    video_reader = decord.VideoReader(uri=video_path)
    video_num_frames = len(video_reader)
    frames = video_reader.get_batch(list(range(video_num_frames)))
    F, H, W, C = frames.shape
    original_shape = (F, H, W, C)

    # Pad frames to (8n+1)
    pad_f = 0
    remainder = (F - 1) % 8
    if remainder != 0:
        last_frame = frames[-1:]
        pad_f = 8 - remainder
        repeated_frames = last_frame.repeat(pad_f, 1, 1, 1)
        frames = torch.cat([frames, repeated_frames], dim=0)

    # Pad spatial dims to multiple of 16
    pad_h = (16 - H % 16) % 16
    pad_w = (16 - W % 16) % 16
    if pad_h > 0 or pad_w > 0:
        frames = torch.nn.functional.pad(frames, pad=(0, 0, 0, pad_w, 0, pad_h))

    # [F, C, H, W]
    return frames.float().permute(0, 3, 1, 2).contiguous(), pad_f, pad_h, pad_w, original_shape


def remove_padding(video: torch.Tensor, pad_f: int, pad_h: int, pad_w: int) -> torch.Tensor:
    """Remove temporal and spatial padding from [B, C, F, H, W] tensor."""
    if pad_f > 0:
        video = video[:, :, :-pad_f, :, :]
    if pad_h > 0:
        video = video[:, :, :, :-pad_h, :]
    if pad_w > 0:
        video = video[:, :, :, :, :-pad_w]
    return video


def save_video(video: torch.Tensor, output_path: str, fps: int = 16):
    """
    Save video tensor to mp4.

    Args:
        video: [B, C, F, H, W] float tensor in [0, 1]
        output_path: path to save
        fps: frames per second
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    video = video[0]  # remove batch dim
    video = video.permute(1, 2, 3, 0)  # [F, H, W, C]
    frames = (video * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    iio.imwrite(
        output_path,
        frames,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_params=["-crf", "10"],
    )


# ---------------------------------------------------------------------------
# Temporal chunking and spatial tiling
# ---------------------------------------------------------------------------

def make_temporal_chunks(F: int, chunk_len: int, overlap_t: int = 8):
    if chunk_len == 0:
        return [(0, F)]

    effective_stride = chunk_len - overlap_t
    if effective_stride <= 0:
        raise ValueError("chunk_len must be greater than overlap")

    chunk_starts = list(range(0, F - overlap_t, effective_stride))
    if chunk_starts[-1] + chunk_len < F:
        chunk_starts.append(F - chunk_len)

    time_chunks = []
    for t_start in chunk_starts:
        t_end = min(t_start + chunk_len, F)
        time_chunks.append((t_start, t_end))

    if len(time_chunks) >= 2 and time_chunks[-1][1] - time_chunks[-1][0] < chunk_len:
        last = time_chunks.pop()
        prev_start, _ = time_chunks[-1]
        time_chunks[-1] = (prev_start, last[1])

    return time_chunks


def make_spatial_tiles(H: int, W: int, tile_size_hw: Tuple[int, int],
                       overlap_hw: Tuple[int, int] = (32, 32)):
    tile_height, tile_width = tile_size_hw
    overlap_h, overlap_w = overlap_hw

    if tile_height == 0 or tile_width == 0:
        return [(0, H, 0, W)]

    tile_stride_h = tile_height - overlap_h
    tile_stride_w = tile_width - overlap_w

    if tile_stride_h <= 0 or tile_stride_w <= 0:
        raise ValueError("Tile size must be greater than overlap")

    h_tiles = list(range(0, H - overlap_h, tile_stride_h))
    if not h_tiles or h_tiles[-1] + tile_height < H:
        h_tiles.append(H - tile_height)
    if len(h_tiles) >= 2 and h_tiles[-1] + tile_height > H:
        h_tiles.pop()

    w_tiles = list(range(0, W - overlap_w, tile_stride_w))
    if not w_tiles or w_tiles[-1] + tile_width < W:
        w_tiles.append(W - tile_width)
    if len(w_tiles) >= 2 and w_tiles[-1] + tile_width > W:
        w_tiles.pop()

    spatial_tiles = []
    for h_start in h_tiles:
        h_end = min(h_start + tile_height, H)
        if h_end + tile_stride_h > H:
            h_end = H
        for w_start in w_tiles:
            w_end = min(w_start + tile_width, W)
            if w_end + tile_stride_w > W:
                w_end = W
            spatial_tiles.append((h_start, h_end, w_start, w_end))
    return spatial_tiles


def get_valid_tile_region(t_start, t_end, h_start, h_end, w_start, w_end,
                          video_shape, overlap_t, overlap_h, overlap_w):
    _, _, F, H, W = video_shape

    t_len = t_end - t_start
    h_len = h_end - h_start
    w_len = w_end - w_start

    valid_t_start = 0 if t_start == 0 else overlap_t // 2
    valid_t_end = t_len if t_end == F else t_len - overlap_t // 2
    valid_h_start = 0 if h_start == 0 else overlap_h // 2
    valid_h_end = h_len if h_end == H else h_len - overlap_h // 2
    valid_w_start = 0 if w_start == 0 else overlap_w // 2
    valid_w_end = w_len if w_end == W else w_len - overlap_w // 2

    out_t_start = t_start + valid_t_start
    out_t_end = t_start + valid_t_end
    out_h_start = h_start + valid_h_start
    out_h_end = h_start + valid_h_end
    out_w_start = w_start + valid_w_start
    out_w_end = w_start + valid_w_end

    return {
        "valid_t_start": valid_t_start, "valid_t_end": valid_t_end,
        "valid_h_start": valid_h_start, "valid_h_end": valid_h_end,
        "valid_w_start": valid_w_start, "valid_w_end": valid_w_end,
        "out_t_start": out_t_start, "out_t_end": out_t_end,
        "out_h_start": out_h_start, "out_h_end": out_h_end,
        "out_w_start": out_w_start, "out_w_end": out_w_end,
    }


# ---------------------------------------------------------------------------
# Rotary positional embeddings
# ---------------------------------------------------------------------------

def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    transformer_config: Dict,
    vae_scale_factor_spatial: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from diffusers.models.embeddings import get_3d_rotary_pos_embed

    grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
    grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

    if transformer_config.patch_size_t is None:
        base_num_frames = num_frames
    else:
        base_num_frames = (
            num_frames + transformer_config.patch_size_t - 1
        ) // transformer_config.patch_size_t

    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=transformer_config.attention_head_dim,
        crops_coords=None,
        grid_size=(grid_height, grid_width),
        temporal_size=base_num_frames,
        grid_type="slice",
        max_size=(grid_height, grid_width),
        device=device,
    )
    return freqs_cos, freqs_sin


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def process_video(
    pipe,
    video: torch.Tensor,
    noise_step: int = 0,
    sr_noise_step: int = 399,
    empty_prompt_embedding: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Run one-step diffusion VSR on a video chunk.

    Args:
        pipe: CogVideoXPipeline
        video: [B, C, F, H, W] input tensor in [-1, 1]
        noise_step: noise level to add (0 = no noise)
        sr_noise_step: timestep for SR denoising
        empty_prompt_embedding: pre-computed empty prompt embedding

    Returns:
        [B, C, F, H, W] output tensor in [0, 1]
    """
    video = video.to(pipe.vae.device, dtype=pipe.vae.dtype)
    latent_dist = pipe.vae.encode(video).latent_dist
    latent = latent_dist.sample() * pipe.vae.config.scaling_factor

    patch_size_t = pipe.transformer.config.patch_size_t
    ncopy = 0
    if patch_size_t is not None:
        ncopy = latent.shape[2] % patch_size_t
        if ncopy > 0:
            first_frame = latent[:, :, :1, :, :]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
        assert latent.shape[2] % patch_size_t == 0

    batch_size, num_channels, num_frames, height, width = latent.shape

    # Get prompt embeddings
    if empty_prompt_embedding is not None:
        prompt_embedding = empty_prompt_embedding.to(latent.device, dtype=latent.dtype)
        if prompt_embedding.shape[0] != batch_size:
            prompt_embedding = prompt_embedding.repeat(batch_size, 1, 1)
    else:
        prompt_token_ids = pipe.tokenizer(
            "",
            padding="max_length",
            max_length=pipe.transformer.config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_embedding = pipe.text_encoder(
            prompt_token_ids.input_ids.to(latent.device)
        )[0]
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

    latent = latent.permute(0, 2, 1, 3, 4)

    # Add noise
    if noise_step != 0:
        noise = torch.randn_like(latent)
        add_timesteps = torch.full(
            (batch_size,), fill_value=noise_step, dtype=torch.long, device=latent.device
        )
        latent = pipe.scheduler.add_noise(latent, noise, add_timesteps)

    timesteps = torch.full(
        (batch_size,), fill_value=sr_noise_step, dtype=torch.long, device=latent.device
    )

    # Rotary embeddings
    vae_scale_factor_spatial = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    transformer_config = pipe.transformer.config
    rotary_emb = None
    if transformer_config.use_rotary_positional_embeddings:
        rotary_emb = prepare_rotary_positional_embeddings(
            height=height * vae_scale_factor_spatial,
            width=width * vae_scale_factor_spatial,
            num_frames=num_frames,
            transformer_config=transformer_config,
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            device=latent.device,
        )

    # One-step denoising
    predicted_noise = pipe.transformer(
        hidden_states=latent,
        encoder_hidden_states=prompt_embedding,
        timestep=timesteps,
        image_rotary_emb=rotary_emb,
        return_dict=False,
    )[0]

    latent_generate = pipe.scheduler.get_velocity(predicted_noise, latent, timesteps)

    if patch_size_t is not None and ncopy > 0:
        latent_generate = latent_generate[:, ncopy:, :, :, :]

    video_generate = pipe.decode_latents(latent_generate)
    video_generate = (video_generate * 0.5 + 0.5).clamp(0.0, 1.0)

    return video_generate


# ---------------------------------------------------------------------------
# Sample input download
# ---------------------------------------------------------------------------

def maybe_download_sample_input(input_arg: str, token: Optional[str]) -> Optional[Path]:
    if input_arg != "sample":
        return None
    from huggingface_hub import hf_hub_download

    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    local_dir = cache_root / "cvl_samples" / "dove"
    local_dir.mkdir(parents=True, exist_ok=True)
    sample_path = hf_hub_download(
        repo_id="zzsi/cvl",
        repo_type="dataset",
        filename="dove/001.mp4",
        cache_dir=str(cache_root),
        local_dir=str(local_dir),
        token=token,
    )
    return Path(sample_path).resolve()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="DOVE one-step video super-resolution")
    parser.add_argument(
        "--input", default="sample",
        help="Input video file or 'sample' for demo (default: sample)",
    )
    parser.add_argument("--output", default="outputs/dove_out.mp4", help="Output video path")
    parser.add_argument(
        "--model_path", default=None,
        help="Path to DOVE model (local dir or HF repo ID). "
             "Default: downloads zzsi/DOVE from HuggingFace.",
    )
    parser.add_argument(
        "--model_subfolder", default=None,
        help="Subfolder within HF repo containing model_index.json (default: stage2 for zzsi/DOVE)",
    )
    parser.add_argument("--upscale", type=int, default=4, help="Upscale factor (default: 4)")
    parser.add_argument(
        "--upscale_mode", type=str, default="bilinear",
        help="Interpolation mode for upscaling (default: bilinear)",
    )
    parser.add_argument("--noise_step", type=int, default=0, help="Noise level to add (default: 0)")
    parser.add_argument("--sr_noise_step", type=int, default=399, help="SR denoising timestep (default: 399)")
    parser.add_argument("--fps", type=int, default=None, help="Output FPS (default: from input video)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--is_cpu_offload", action="store_true", help="Enable sequential CPU offload (saves VRAM)")
    parser.add_argument("--is_vae_st", action="store_true", help="Enable VAE slicing and tiling (saves VRAM)")
    parser.add_argument("--tile_size_hw", type=int, nargs=2, default=(0, 0), help="Spatial tile size (H W)")
    parser.add_argument("--overlap_hw", type=int, nargs=2, default=(32, 32), help="Spatial tile overlap (H W)")
    parser.add_argument("--chunk_len", type=int, default=0, help="Temporal chunk length (0 = no chunking)")
    parser.add_argument("--overlap_t", type=int, default=8, help="Temporal chunk overlap")
    parser.add_argument("--hf_token", default=None, help="HuggingFace token (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup and exit")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)
        for name in ["transformers", "diffusers", "torch"]:
            logging.getLogger(name).setLevel(logging.INFO)

    # Resolve dtype
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Resolve model path
    model_path = args.model_path or "zzsi/DOVE"
    model_subfolder = args.model_subfolder
    # Default subfolder for the zzsi/DOVE HF repo
    if model_subfolder is None and model_path == "zzsi/DOVE":
        model_subfolder = "stage2"
    if args.model_path is None:
        print(f"Using HF model: {model_path}")

    if args.dry_run:
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Model: {model_path}")
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        print(f"Dtype: {args.dtype}")
        print("Dry run successful. Configuration validated.")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("DOVE requires CUDA GPU support.")

    input_base = get_input_dir()
    output_base = get_output_dir()

    # Load empty prompt embedding
    empty_prompt_embedding = None
    embed_path = script_dir / "prompt_embeddings" / "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors"
    if embed_path.exists():
        try:
            empty_prompt_embedding = load_file(str(embed_path))["prompt_embedding"]
            print(f"Loaded empty prompt embedding from {embed_path.name}")
        except Exception as e:
            print(f"Warning: Failed to load empty prompt embedding: {e}")

    # Resolve input
    if args.input == "sample":
        sample_path = maybe_download_sample_input(args.input, hf_token)
        if sample_path is None:
            raise FileNotFoundError("No sample input available")
        input_path = str(sample_path)
    else:
        input_path = resolve_input_path(args.input, input_base)

    output_path = resolve_output_path(args.output, output_base)

    from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
    from transformers import set_seed

    set_seed(args.seed)

    # Get input video FPS
    input_fps = get_video_fps(input_path)
    fps = args.fps if args.fps is not None else input_fps

    # Preprocess video
    print(f"Reading input video: {input_path}")
    video, pad_f, pad_h, pad_w, original_shape = preprocess_video(input_path)
    F_frames, H_, W_ = video.shape[0], video.shape[2], video.shape[3]
    print(f"Original: {original_shape[0]} frames, {original_shape[2]}x{original_shape[1]}")
    print(f"Padded: {F_frames} frames, {W_}x{H_}")

    # Upscale with interpolation
    video = torch.nn.functional.interpolate(
        video, size=(H_ * args.upscale, W_ * args.upscale),
        mode=args.upscale_mode, align_corners=False,
    )
    # Normalize to [-1, 1]
    video = video / 255.0 * 2.0 - 1.0
    # [B, C, F, H, W]
    video = video.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()

    _B, _C, _F, _H, _W = video.shape
    print(f"Upscaled: {_F} frames, {_W}x{_H} (x{args.upscale})")

    # Setup chunking/tiling
    overlap_t = args.overlap_t if args.chunk_len > 0 else 0
    overlap_hw = tuple(args.overlap_hw) if args.tile_size_hw != (0, 0) else (0, 0)
    time_chunks = make_temporal_chunks(_F, args.chunk_len, overlap_t)
    spatial_tiles = make_spatial_tiles(_H, _W, tuple(args.tile_size_hw), overlap_hw)

    if args.chunk_len > 0:
        print(f"Temporal chunking: {len(time_chunks)} chunks (len={args.chunk_len}, overlap={overlap_t})")
    if args.tile_size_hw != (0, 0):
        print(f"Spatial tiling: {len(spatial_tiles)} tiles (size={args.tile_size_hw}, overlap={list(overlap_hw)})")

    # Load model — resolve HF repo with subfolder if needed
    resolved_model_path = model_path
    if not os.path.isdir(model_path):
        from huggingface_hub import snapshot_download
        cache_dir = os.environ.get("HF_HOME", None)
        repo_path = snapshot_download(
            repo_id=model_path, cache_dir=cache_dir, token=hf_token,
        )
        if model_subfolder:
            resolved_model_path = os.path.join(repo_path, model_subfolder)
        else:
            resolved_model_path = repo_path
    elif model_subfolder:
        resolved_model_path = os.path.join(model_path, model_subfolder)

    print(f"Loading model: {resolved_model_path}")
    pipe = CogVideoXPipeline.from_pretrained(resolved_model_path, torch_dtype=dtype)

    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    if args.is_cpu_offload:
        print("Enabling sequential CPU offload")
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")

    if args.is_vae_st:
        print("Enabling VAE slicing and tiling")
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    # Run inference with chunking/tiling
    output_video = torch.zeros_like(video)
    write_count = torch.zeros_like(video, dtype=torch.int)

    total_chunks = len(time_chunks) * len(spatial_tiles)
    print(f"Processing {total_chunks} chunk(s)...")

    chunk_idx = 0
    for t_start, t_end in time_chunks:
        for h_start, h_end, w_start, w_end in spatial_tiles:
            chunk_idx += 1
            video_chunk = video[:, :, t_start:t_end, h_start:h_end, w_start:w_end]
            print(f"  Chunk {chunk_idx}/{total_chunks}: t=[{t_start}:{t_end}] h=[{h_start}:{h_end}] w=[{w_start}:{w_end}]")

            _video_generate = process_video(
                pipe=pipe,
                video=video_chunk,
                noise_step=args.noise_step,
                sr_noise_step=args.sr_noise_step,
                empty_prompt_embedding=empty_prompt_embedding,
            )

            region = get_valid_tile_region(
                t_start, t_end, h_start, h_end, w_start, w_end,
                video_shape=video.shape,
                overlap_t=overlap_t,
                overlap_h=overlap_hw[0],
                overlap_w=overlap_hw[1],
            )
            output_video[:, :,
                         region["out_t_start"]:region["out_t_end"],
                         region["out_h_start"]:region["out_h_end"],
                         region["out_w_start"]:region["out_w_end"]] = \
                _video_generate[:, :,
                                region["valid_t_start"]:region["valid_t_end"],
                                region["valid_h_start"]:region["valid_h_end"],
                                region["valid_w_start"]:region["valid_w_end"]]
            write_count[:, :,
                        region["out_t_start"]:region["out_t_end"],
                        region["out_h_start"]:region["out_h_end"],
                        region["out_w_start"]:region["out_w_end"]] += 1

    if (write_count == 0).any():
        raise RuntimeError("Some regions were not written during tiled inference")
    if (write_count > 1).any():
        raise RuntimeError("Overlapping write detected during tiled inference")

    # Remove padding (scaled by upscale factor for spatial dims)
    video_generate = remove_padding(output_video, pad_f, pad_h * args.upscale, pad_w * args.upscale)

    # Save output
    save_video(video_generate, output_path, fps=fps)
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    main()
