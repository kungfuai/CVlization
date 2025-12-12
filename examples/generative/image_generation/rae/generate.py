"""
Generate images using RAE (Representation Autoencoders) with pretrained models.

Downloads pretrained models from HuggingFace (nyu-visionx/RAE-collections) and
generates images conditioned on ImageNet class labels.
"""
import os
import sys
import math
import argparse
from pathlib import Path
from time import time

import torch
from torchvision.utils import save_image
from huggingface_hub import hf_hub_download

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage1 import RAE
from stage2.models.DDT import DiTwDDTHead
from stage2.transport import create_transport, Sampler


def download_pretrained_models() -> dict:
    """Download pretrained models from HuggingFace using centralized cache."""
    repo_id = "nyu-visionx/RAE-collections"

    # Files to download (uses HF_HOME for caching)
    files = {
        "decoder": "decoders/dinov2/wReg_base/ViTXL_n08/model.pt",
        "stats": "stats/dinov2/wReg_base/imagenet1k/stat.pt",
        "dit": "DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt",
    }

    local_paths = {}
    for name, remote_path in files.items():
        print(f"Downloading/loading {name} from {repo_id}...")
        local_paths[name] = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
        )
        print(f"  -> {local_paths[name]}")

    return local_paths


def build_rae(decoder_path: str, stats_path: str, device: torch.device) -> RAE:
    """Build RAE autoencoder with pretrained decoder."""
    # Use local decoder config for ViT-XL
    script_dir = Path(__file__).parent
    decoder_config_path = str(script_dir / "configs" / "decoder" / "ViTXL")

    rae = RAE(
        encoder_cls='Dinov2withNorm',
        encoder_config_path='facebook/dinov2-with-registers-base',
        encoder_input_size=224,
        encoder_params={'dinov2_path': 'facebook/dinov2-with-registers-base', 'normalize': True},
        decoder_config_path=decoder_config_path,
        decoder_patch_size=16,
        pretrained_decoder_path=decoder_path,
        noise_tau=0.0,
        reshape_to_2d=True,
        normalization_stat_path=stats_path,
    )
    rae = rae.to(device)
    rae.eval()
    return rae


def build_dit(dit_path: str, device: torch.device) -> DiTwDDTHead:
    """Build DiT diffusion transformer with pretrained weights."""
    model = DiTwDDTHead(
        input_size=16,
        patch_size=1,
        in_channels=768,
        hidden_size=[1152, 2048],
        depth=[28, 2],
        num_heads=[16, 16],
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        use_pos_embed=True,
    )

    # Load pretrained weights
    state_dict = torch.load(dit_path, map_location="cpu")
    if "ema" in state_dict:
        state_dict = state_dict["ema"]
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded DiT from {dit_path}")

    model = model.to(device)
    model.eval()
    return model


def generate_images(
    rae: RAE,
    model: DiTwDDTHead,
    class_labels: list,
    num_samples: int,
    cfg_scale: float,
    num_steps: int,
    mode: str,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate images using RAE + DiT pipeline."""
    torch.manual_seed(seed)

    # Latent space parameters
    latent_size = (768, 16, 16)
    num_classes = 1000

    # Create transport and sampler
    time_dist_shift = math.sqrt((768 * 16 * 16) / 4096)
    transport = create_transport(
        path_type='Linear',
        prediction='velocity',
        time_dist_type='uniform',
        time_dist_shift=time_dist_shift,
    )
    sampler = Sampler(transport)

    if mode.upper() == "ODE":
        sample_fn = sampler.sample_ode(
            sampling_method='euler',
            num_steps=num_steps,
            atol=1e-6,
            rtol=1e-3,
            reverse=False,
        )
    else:
        sample_fn = sampler.sample_sde(
            sampling_method='Euler',
            diffusion_form='SBDM',
            diffusion_norm=1.0,
            last_step='Mean',
            last_step_size=0.04,
            num_steps=num_steps,
        )

    # Expand class labels to num_samples
    if len(class_labels) < num_samples:
        class_labels = class_labels * (num_samples // len(class_labels) + 1)
    class_labels = class_labels[:num_samples]

    # Create sampling noise
    n = len(class_labels)
    z = torch.randn(n, *latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)

    model_kwargs = dict(
        y=y,
        cfg_scale=cfg_scale,
        cfg_interval=(0.0, 1.0),
    )
    model_fn = model.forward_with_cfg

    # Sample
    print(f"Generating {n} images with {num_steps} steps...")
    start_time = time()

    with torch.no_grad():
        samples = sample_fn(z, model_fn, **model_kwargs)[-1]
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = rae.decode(samples)

    print(f"Generation took {time() - start_time:.2f} seconds")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate images with RAE")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory to save generated images")
    parser.add_argument("--class-ids", type=str, default="207,360,387,971,88,979,417,279",
                        help="Comma-separated ImageNet class IDs")
    parser.add_argument("--num-samples", type=int, default=8,
                        help="Number of images to generate")
    parser.add_argument("--cfg-scale", type=float, default=1.8,
                        help="Classifier-free guidance scale")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of sampling steps")
    parser.add_argument("--mode", type=str, choices=["ode", "sde"], default="ode",
                        help="Sampling mode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download pretrained models (uses centralized HF cache)
    print("\n=== Downloading pretrained models ===")
    model_paths = download_pretrained_models()

    # Build models
    print("\n=== Building models ===")
    rae = build_rae(model_paths["decoder"], model_paths["stats"], device)
    dit = build_dit(model_paths["dit"], device)

    # Parse class IDs
    class_ids = [int(x.strip()) for x in args.class_ids.split(",")]

    # Generate images
    print("\n=== Generating images ===")
    samples = generate_images(
        rae=rae,
        model=dit,
        class_labels=class_ids,
        num_samples=args.num_samples,
        cfg_scale=args.cfg_scale,
        num_steps=args.num_steps,
        mode=args.mode,
        seed=args.seed,
        device=device,
    )

    # Save images
    output_path = output_dir / f"samples_cfg{args.cfg_scale}_steps{args.num_steps}.png"
    save_image(samples, output_path, nrow=4, normalize=True, value_range=(0, 1))
    print(f"\nSaved {len(samples)} images to {output_path}")

    # Also save individual images
    for i, img in enumerate(samples):
        individual_path = output_dir / f"sample_{i:03d}_class{class_ids[i % len(class_ids)]}.png"
        save_image(img, individual_path, normalize=True, value_range=(0, 1))


if __name__ == "__main__":
    main()
