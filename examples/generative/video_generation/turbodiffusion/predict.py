#!/usr/bin/env python3
"""
TurboDiffusion inference wrapper for CVlization.

Generates videos using the TurboWan2.1-T2V-1.3B model with 100x+ speedup
over the original Wan2.1 model.
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# Add TurboDiffusion to path
sys.path.insert(0, "/workspace/TurboDiffusion/turbodiffusion")

from imaginaire.utils.io import save_image_or_video
from imaginaire.utils import log
from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from inference.modify_model import tensor_kwargs, create_model

torch._dynamo.config.suppress_errors = True

CHECKPOINT_DIR = Path("/workspace/checkpoints")

# HuggingFace model repos
HF_REPOS = {
    "vae": "Wan-AI/Wan2.1-T2V-1.3B",
    "text_encoder": "Wan-AI/Wan2.1-T2V-1.3B",
    "dit_1.3b": "TurboDiffusion/TurboWan2.1-T2V-1.3B-480P",
}

# Checkpoint filenames
CHECKPOINTS = {
    "vae": "Wan2.1_VAE.pth",
    "text_encoder": "models_t5_umt5-xxl-enc-bf16.pth",
    "dit_1.3b": "TurboWan2.1-T2V-1.3B-480P-quant.pth",
}


def download_checkpoints():
    """Download required checkpoints from HuggingFace if not present."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for key, filename in CHECKPOINTS.items():
        local_path = CHECKPOINT_DIR / filename
        if local_path.exists():
            log.info(f"Checkpoint already exists: {local_path}")
            continue

        repo_key = key.replace("_quant", "")
        repo_id = HF_REPOS.get(repo_key, HF_REPOS.get(key.split("_")[0]))

        log.info(f"Downloading {filename} from {repo_id}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(CHECKPOINT_DIR),
            local_dir_use_symlinks=False,
        )
        log.success(f"Downloaded: {local_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TurboDiffusion video generation (Wan2.1-1.3B)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cat playing piano in a jazz club, cinematic lighting",
        help="Text prompt for video generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/outputs/generated_video.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=77,
        help="Number of frames (77 = ~5 seconds at 16fps)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        choices=[1, 2, 3, 4],
        default=4,
        help="Sampling steps (1-4, more = better quality)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="480p",
        choices=["480p", "720p"],
        help="Output resolution",
    )
    parser.add_argument(
        "--aspect_ratio",
        type=str,
        default="16:9",
        help="Aspect ratio (width:height)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--attention_type",
        type=str,
        default="sagesla",
        choices=["original", "sla", "sagesla"],
        help="Attention type (sagesla recommended for speed)",
    )
    parser.add_argument(
        "--sla_topk",
        type=float,
        default=0.15,
        help="Top-k ratio for sparse attention (0.15 for quality, 0.1 for speed)",
    )
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=80,
        help="Initial sigma for rCM (higher = less diverse but higher quality)",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip checkpoint download (assume already present)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Download checkpoints if needed
    if not args.skip_download:
        download_checkpoints()

    # Paths
    dit_path = str(CHECKPOINT_DIR / CHECKPOINTS["dit_1.3b"])
    vae_path = str(CHECKPOINT_DIR / CHECKPOINTS["vae"])
    text_encoder_path = str(CHECKPOINT_DIR / CHECKPOINTS["text_encoder"])

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create args namespace for model creation
    model_args = argparse.Namespace(
        model="Wan2.1-1.3B",
        attention_type=args.attention_type,
        sla_topk=args.sla_topk,
        quant_linear=True,  # Using quantized checkpoint
        default_norm=False,
    )

    # Encode text prompt
    log.info(f"Encoding prompt: {args.prompt}")
    text_emb = get_umt5_embedding(
        checkpoint_path=text_encoder_path, prompts=args.prompt
    ).to(**tensor_kwargs)
    clear_umt5_memory()

    # Load model
    log.info(f"Loading DiT model from {dit_path}")
    net = create_model(dit_path=dit_path, args=model_args).cpu()
    torch.cuda.empty_cache()
    log.success("Model loaded successfully")

    # Load VAE
    tokenizer = Wan2pt1VAEInterface(vae_pth=vae_path)

    # Get resolution
    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
    log.info(f"Generating {args.num_frames} frames at {w}x{h} ({args.resolution})")

    # Prepare condition
    condition = {"crossattn_emb": repeat(text_emb, "b l d -> (k b) l d", k=1)}

    # State shape for latents
    state_shape = [
        tokenizer.latent_ch,
        tokenizer.get_latent_num_frames(args.num_frames),
        h // tokenizer.spatial_compression_factor,
        w // tokenizer.spatial_compression_factor,
    ]

    # Generate initial noise
    generator = torch.Generator(device=tensor_kwargs["device"])
    generator.manual_seed(args.seed)

    init_noise = torch.randn(
        1,
        *state_shape,
        dtype=torch.float32,
        device=tensor_kwargs["device"],
        generator=generator,
    )

    # Timesteps for sampling
    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]
    t_steps = torch.tensor(
        [math.atan(args.sigma_max), *mid_t, 0],
        dtype=torch.float64,
        device=init_noise.device,
    )
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    # Sampling loop
    log.info(f"Sampling with {args.num_steps} steps...")
    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)

    net.cuda()
    for t_cur, t_next in tqdm(
        list(zip(t_steps[:-1], t_steps[1:])),
        desc="Sampling",
        total=args.num_steps,
    ):
        with torch.no_grad():
            v_pred = net(
                x_B_C_T_H_W=x.to(**tensor_kwargs),
                timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                **condition,
            ).to(torch.float64)

            x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                *x.shape,
                dtype=torch.float32,
                device=tensor_kwargs["device"],
                generator=generator,
            )

    samples = x.float()

    # Decode to video
    log.info("Decoding latents to video...")
    video = tokenizer.decode(samples)
    video = (1.0 + video.clamp(-1, 1)) / 2.0

    # Save video
    save_image_or_video(
        rearrange(video, "b c t h w -> c t h (b w)"),
        str(output_path),
        fps=16,
    )
    log.success(f"Video saved to: {output_path}")


if __name__ == "__main__":
    main()
