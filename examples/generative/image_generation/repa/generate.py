#!/usr/bin/env python3
"""
REPA: Representation Alignment for Generation
Image generation script using pretrained SiT model.

Adapted from https://github.com/sihyun-yu/REPA
Paper: https://arxiv.org/abs/2410.06940
"""

import os
import sys
import logging
import warnings

# Suppress verbose logging by default
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
logging.basicConfig(level=logging.ERROR)
for logger_name in ["transformers", "diffusers", "torch"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from models.sit import SiT_models
from samplers import euler_sampler, euler_maruyama_sampler
from utils import download_model, load_legacy_checkpoints
from diffusers.models import AutoencoderKL


# ImageNet class labels (subset for display)
IMAGENET_CLASSES = {
    0: "tench",
    1: "goldfish",
    88: "macaw",
    207: "golden retriever",
    279: "arctic fox",
    291: "lion",
    360: "otter",
    387: "lesser panda",
    388: "giant panda",
    417: "balloon",
    971: "bubble",
    972: "cliff",
    973: "coral reef",
    974: "geyser",
    975: "lakeside",
    980: "volcano",
    985: "daisy",
    992: "agaric",
}


def detect_device():
    """Auto-detect cuda/mps/cpu with appropriate dtype."""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    else:
        return torch.device("cpu"), torch.float32


def load_model(args, device):
    """Load SiT model with pretrained weights."""
    latent_size = args.resolution // 8
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}

    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=True,
        z_dims=[int(z_dim) for z_dim in args.projector_embed_dims.split(',')],
        encoder_depth=args.encoder_depth,
        **block_kwargs,
    ).to(device)

    # Load checkpoint
    if args.ckpt is None:
        # Auto-download pretrained model
        print("Downloading pretrained SiT-XL/2 model...")
        state_dict = download_model('last.pt')
    else:
        state_dict = torch.load(args.ckpt, map_location=device)
        if 'ema' in state_dict:
            state_dict = state_dict['ema']

    if args.legacy:
        state_dict = load_legacy_checkpoints(state_dict, args.encoder_depth)

    model.load_state_dict(state_dict)
    model.eval()

    return model


def generate_images(model, vae, args, device, dtype):
    """Generate images using the model."""
    latent_size = args.resolution // 8

    # Setup class labels
    if args.class_ids:
        class_ids = [int(c) for c in args.class_ids.split(',')]
        n = len(class_ids)
        y = torch.tensor(class_ids, device=device)
    else:
        n = args.num_samples
        y = torch.randint(0, args.num_classes, (n,), device=device)

    # Create sampling noise
    z = torch.randn(n, 4, latent_size, latent_size, device=device, dtype=dtype)

    print(f"Generating {n} images with cfg_scale={args.cfg_scale}...")

    # Sample
    sampling_kwargs = dict(
        model=model,
        latents=z,
        y=y,
        num_steps=args.num_steps,
        heun=args.heun,
        cfg_scale=args.cfg_scale,
        guidance_low=args.guidance_low,
        guidance_high=args.guidance_high,
        path_type=args.path_type,
    )

    with torch.no_grad():
        if args.mode == "sde":
            samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
        elif args.mode == "ode":
            samples = euler_sampler(**sampling_kwargs).to(torch.float32)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        # Decode latents
        latents_scale = torch.tensor([0.18215] * 4).view(1, 4, 1, 1).to(device)
        latents_bias = torch.tensor([0.0] * 4).view(1, 4, 1, 1).to(device)

        samples = vae.decode((samples - latents_bias) / latents_scale).sample
        samples = (samples + 1) / 2.0  # [-1, 1] -> [0, 1]
        samples = torch.clamp(samples, 0, 1)

    return samples, y


def save_images(samples, labels, output_dir, args):
    """Save generated images."""
    os.makedirs(output_dir, exist_ok=True)

    samples_np = samples.permute(0, 2, 3, 1).cpu().numpy()
    samples_np = (samples_np * 255).astype(np.uint8)

    saved_paths = []
    for i, (img_np, label) in enumerate(zip(samples_np, labels)):
        label_id = label.item()
        class_name = IMAGENET_CLASSES.get(label_id, f"class_{label_id}")
        filename = f"{i:04d}_{class_name}.png"
        filepath = os.path.join(output_dir, filename)
        Image.fromarray(img_np).save(filepath)
        saved_paths.append(filepath)
        print(f"  Saved: {filename}")

    return saved_paths


def create_grid(samples, nrow=4):
    """Create a grid of images."""
    from torchvision.utils import make_grid
    grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    grid_np = (grid_np * 255).astype(np.uint8)
    return Image.fromarray(grid_np)


def main(args):
    """Main generation function."""
    device, dtype = detect_device()
    print(f"Using device: {device}, dtype: {dtype}")

    # Override for TF32
    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    # Load models
    print("Loading SiT model...")
    model = load_model(args, device)
    print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    # Generate
    samples, labels = generate_images(model, vae, args, device, dtype)

    # Save
    output_dir = args.output_dir
    print(f"\nSaving images to {output_dir}/")
    saved_paths = save_images(samples, labels, output_dir, args)

    # Create and save grid
    if args.save_grid and len(samples) > 1:
        nrow = min(4, len(samples))
        grid = create_grid(samples, nrow=nrow)
        grid_path = os.path.join(output_dir, "grid.png")
        grid.save(grid_path)
        print(f"  Saved grid: grid.png")

    print(f"\nGeneration complete! {len(saved_paths)} images saved.")


def parse_args():
    parser = argparse.ArgumentParser(description="REPA Image Generation")

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/generated",
                       help="Output directory for generated images")
    parser.add_argument("--save-grid", action="store_true", default=True,
                       help="Save a grid of all generated images")

    # Model
    parser.add_argument("--model", type=str, default="SiT-XL/2",
                       choices=list(SiT_models.keys()))
    parser.add_argument("--ckpt", type=str, default=None,
                       help="Path to checkpoint (auto-downloads if not specified)")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--fused-attn", action="store_true", default=False)
    parser.add_argument("--qk-norm", action="store_true", default=False)
    parser.add_argument("--projector-embed-dims", type=str, default="768")

    # VAE
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")

    # Generation
    parser.add_argument("--num-samples", type=int, default=4,
                       help="Number of images to generate")
    parser.add_argument("--class-ids", type=str, default=None,
                       help="Comma-separated class IDs (e.g., '207,388,971,985')")

    # Sampling
    parser.add_argument("--mode", type=str, default="ode", choices=["ode", "sde"])
    parser.add_argument("--cfg-scale", type=float, default=1.8,
                       help="Classifier-free guidance scale")
    parser.add_argument("--path-type", type=str, default="linear",
                       choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=250,
                       help="Number of sampling steps")
    parser.add_argument("--heun", action="store_true", default=False,
                       help="Use Heun's method (ODE only)")
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=0.7)

    # Precision
    parser.add_argument("--tf32", action="store_true", default=True,
                       help="Use TF32 on Ampere GPUs")
    parser.add_argument("--no-tf32", action="store_false", dest="tf32")

    # Legacy
    parser.add_argument("--legacy", action="store_true", default=False)

    # Verbose
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)
        for logger_name in ["transformers", "diffusers", "torch"]:
            logging.getLogger(logger_name).setLevel(logging.INFO)

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
