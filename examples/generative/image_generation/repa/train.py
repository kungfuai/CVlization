#!/usr/bin/env python3
"""
REPA: Representation Alignment for Generation
Training script with CIFAR-10 support and on-the-fly VAE encoding.

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
import copy
from copy import deepcopy
import json
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from datasets import load_dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.sit import SiT_models
from loss import SILoss
from utils import load_encoders

from diffusers.models import AutoencoderKL
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from torchvision.utils import make_grid
import math

logger = get_logger(__name__)

# Normalization constants
CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


def preprocess_raw_image(x, enc_type, resolution=256):
    """Preprocess images for the visual encoder."""
    if 'clip' in enc_type:
        x = x / 255. if x.max() > 1 else x
        x = F.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255. if x.max() > 1 else x
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = F.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    else:
        x = x / 255. if x.max() > 1 else x
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    return x


def array2grid(x):
    """Convert batch of images to grid for logging."""
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def encode_to_latent(vae, images, latents_scale, latents_bias):
    """Encode images to VAE latent space."""
    # images should be in [-1, 1] range
    posterior = vae.encode(images).latent_dist
    z = posterior.sample()
    z = z * latents_scale + latents_bias
    return z


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Update EMA model weights."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """Create a logger that writes to a log file and stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    return logging.getLogger(__name__)


def requires_grad(model, flag=True):
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


class HFImageNetDataset(Dataset):
    """Wrapper for HuggingFace ImageNet dataset."""
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataset(dataset_name, resolution, data_dir=None):
    """Get dataset with appropriate transforms."""
    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Scale to [-1, 1] for VAE
    ])

    # Also get raw images for encoder (without normalization)
    raw_transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])

    if dataset_name == "cifar10":
        root = data_dir or "./data/cifar10"
        dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        raw_dataset = datasets.CIFAR10(root=root, train=True, download=False, transform=raw_transform)
        num_classes = 10
    elif dataset_name == "cifar100":
        root = data_dir or "./data/cifar100"
        dataset = datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
        raw_dataset = datasets.CIFAR100(root=root, train=True, download=False, transform=raw_transform)
        num_classes = 100
    elif dataset_name == "imagenet":
        print("Loading ImageNet from HuggingFace (requires login and terms acceptance)...")
        print("If this fails, run: huggingface-cli login")
        print("Then accept terms at: https://huggingface.co/datasets/ILSVRC/imagenet-1k")
        hf_dataset = load_dataset("ILSVRC/imagenet-1k", split="train", trust_remote_code=True)
        dataset = HFImageNetDataset(hf_dataset, transform=transform)
        raw_dataset = HFImageNetDataset(hf_dataset, transform=raw_transform)
        num_classes = 1000
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset, raw_dataset, num_classes


class PairedDataset(torch.utils.data.Dataset):
    """Dataset that returns both normalized and raw images."""
    def __init__(self, norm_dataset, raw_dataset):
        self.norm_dataset = norm_dataset
        self.raw_dataset = raw_dataset

    def __len__(self):
        return len(self.norm_dataset)

    def __getitem__(self, idx):
        norm_img, label = self.norm_dataset[idx]
        raw_img, _ = self.raw_dataset[idx]
        return norm_img, raw_img, label


def main(args):
    """Main training loop."""
    # Setup accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to if args.report_to != "none" else None,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        train_logger = create_logger(save_dir)
        train_logger.info(f"Experiment directory created at {save_dir}")

    device = accelerator.device

    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Create model
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    # Determine num_classes from dataset if not specified
    if args.num_classes is None:
        dataset_num_classes = {"cifar10": 10, "cifar100": 100, "imagenet": 1000}
        args.num_classes = dataset_num_classes.get(args.dataset, 1000)

    # Load visual encoders for REPA
    if args.enc_type:
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
        )
        z_dims = [encoder.embed_dim for encoder in encoders]
    else:
        encoders, encoder_types, architectures = [], [], []
        z_dims = [768]  # Default

    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=(args.cfg_prob > 0),
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
        **block_kwargs
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    requires_grad(ema, False)
    requires_grad(vae, False)

    latents_scale = torch.tensor([0.18215] * 4).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor([0.0] * 4).view(1, 4, 1, 1).to(device)

    # Create loss function
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type,
        encoders=encoders,
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting
    )

    if accelerator.is_main_process:
        train_logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Setup data
    norm_dataset, raw_dataset, detected_num_classes = get_dataset(
        args.dataset, args.resolution, args.data_dir
    )
    if args.num_classes is None:
        args.num_classes = detected_num_classes

    dataset = PairedDataset(norm_dataset, raw_dataset)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    if accelerator.is_main_process:
        train_logger.info(f"Dataset '{args.dataset}' contains {len(dataset):,} images")
        train_logger.info(f"Number of classes: {args.num_classes}")

    # Prepare models for training
    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()

    # Resume from checkpoint
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) + '.pt'
        ckpt_path = f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}'
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']
        if accelerator.is_main_process:
            train_logger.info(f"Resumed from checkpoint: {ckpt_path}")

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if accelerator.is_main_process and args.report_to == "wandb":
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="REPA",
            config=tracker_config,
            init_kwargs={"wandb": {"name": f"{args.exp_name}"}},
        )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        for norm_images, raw_images, labels in train_dataloader:
            norm_images = norm_images.to(device)
            raw_images = raw_images.to(device)
            labels = labels.to(device)

            # Encode images to latent space on-the-fly
            with torch.no_grad():
                x = encode_to_latent(vae, norm_images, latents_scale, latents_bias)

                # Get encoder features for REPA loss
                zs = []
                if encoders:
                    with accelerator.autocast():
                        for encoder, encoder_type in zip(encoders, encoder_types):
                            raw_img = preprocess_raw_image(raw_images, encoder_type, args.resolution)
                            z = encoder.forward_features(raw_img)
                            if 'dinov2' in encoder_type:
                                z = z['x_norm_patchtokens']
                            zs.append(z)

            with accelerator.accumulate(model):
                model_kwargs = dict(y=labels)
                loss, proj_loss = loss_fn(model, x, model_kwargs, zs=zs)
                loss_mean = loss.mean()
                proj_loss_mean = proj_loss.mean() if isinstance(proj_loss, torch.Tensor) else proj_loss
                total_loss = loss_mean + proj_loss_mean * args.proj_coeff

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            # Logging
            if global_step % args.log_every == 0:
                logs = {
                    "loss": accelerator.gather(loss_mean).mean().detach().item(),
                    "proj_loss": accelerator.gather(torch.tensor(proj_loss_mean)).mean().detach().item() if isinstance(proj_loss_mean, (int, float)) else accelerator.gather(proj_loss_mean).mean().detach().item(),
                }
                if accelerator.sync_gradients:
                    logs["grad_norm"] = accelerator.gather(grad_norm).mean().detach().item()
                progress_bar.set_postfix(**logs)
                if args.report_to == "wandb":
                    accelerator.log(logs, step=global_step)

            # Save checkpoint
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    train_logger.info(f"Saved checkpoint to {checkpoint_path}")

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    model.eval()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        train_logger.info("Training complete!")
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="REPA Training")

    # Logging
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--exp-name", type=str, default="repa-cifar10")
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="none", choices=["none", "wandb"])
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume-step", type=int, default=0)

    # Model
    parser.add_argument("--model", type=str, default="SiT-B/2",
                       choices=list(SiT_models.keys()))
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action="store_true", default=True)
    parser.add_argument("--no-fused-attn", action="store_false", dest="fused_attn")
    parser.add_argument("--qk-norm", action="store_true", default=False)

    # Dataset
    parser.add_argument("--dataset", type=str, default="cifar10",
                       choices=["cifar10", "cifar100", "imagenet"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=32)

    # Precision
    parser.add_argument("--allow-tf32", action="store_true", default=True)
    parser.add_argument("--mixed-precision", type=str, default="fp16",
                       choices=["no", "fp16", "bf16"])

    # Optimization
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-train-steps", type=int, default=10000)
    parser.add_argument("--checkpointing-steps", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=0.)
    parser.add_argument("--adam-epsilon", type=float, default=1e-08)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)

    # Seed
    parser.add_argument("--seed", type=int, default=0)

    # Workers
    parser.add_argument("--num-workers", type=int, default=4)

    # REPA loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"])
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default="dinov2-vit-b")
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str)

    # Verbose
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # Re-enable verbose if requested
    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)
        for logger_name in ["transformers", "diffusers", "torch"]:
            logging.getLogger(logger_name).setLevel(logging.INFO)

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
