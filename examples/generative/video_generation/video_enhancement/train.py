"""
Training Script for Video Artifact Removal (Step-based)

Usage:
    python train.py --dummy                     # Train with dummy data
    python train.py --vimeo --steps 50000       # Train with Vimeo dataset
    python train.py --resume checkpoint.pt      # Resume training
    python train.py --val-every 500             # Validate every 500 steps
    python train.py --artifacts "corner_logo,gaussian_noise"  # Specify artifact types
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
import time
from datetime import datetime
from typing import Dict, Optional
import json

import torchvision.utils as vutils

from config import Config, get_config
from model import TemporalNAFUNet, NAFUNetLite
from losses import ArtifactRemovalLoss, PSNRMetric, SSIMMetric
from dataset import get_dataloaders, get_vimeo_dataloaders


def log_sample_images(
    writer: SummaryWriter,
    model: nn.Module,
    val_loader,
    device: torch.device,
    step: int,
    num_samples: int = 4,
):
    """Log sample images to TensorBoard for visual inspection.

    Creates a grid showing: degraded (input) | predicted (output) | clean (ground truth)
    for each frame in the sequence.
    """
    model.eval()

    # Get first batch
    batch = next(iter(val_loader))
    clean = batch["clean"].to(device)  # [B, T, C, H, W]
    degraded = batch["degraded"].to(device)

    B, T, C, H, W = clean.shape
    num_samples = min(num_samples, B)

    with torch.no_grad():
        pred = model(degraded)
        if isinstance(pred, tuple):
            pred, pred_mask = pred

    # Log samples for first few items in batch
    for b in range(num_samples):
        # For each sample, create a grid showing all frames
        # Layout: 3 rows (degraded, predicted, clean) x T columns (frames)
        frames_degraded = degraded[b]  # [T, C, H, W]
        frames_pred = pred[b].clamp(0, 1)  # [T, C, H, W]
        frames_clean = clean[b]  # [T, C, H, W]

        # Concatenate: all degraded, then all predicted, then all clean
        # This gives us [3*T, C, H, W] in row-major order
        all_frames = torch.cat([frames_degraded, frames_pred, frames_clean], dim=0)
        grid = vutils.make_grid(all_frames, nrow=T, padding=2, normalize=False)

        writer.add_image(f"samples/sample_{b}", grid, step)

    # Also log a single frame comparison in higher detail
    # Take middle frame from first sample
    mid_t = T // 2
    single_comparison = torch.stack([
        degraded[0, mid_t],
        pred[0, mid_t].clamp(0, 1),
        clean[0, mid_t],
    ], dim=0)
    grid_single = vutils.make_grid(single_comparison, nrow=3, padding=4, normalize=False)
    writer.add_image("comparison/degraded_pred_clean", grid_single, step)


def setup_device(config: Config) -> torch.device:
    """Setup compute device"""
    device = config.training.get_device()
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif device.type == "mps":
        print("  Apple Silicon GPU (MPS)")
    
    return device


def setup_model(config: Config, device: torch.device) -> nn.Module:
    """Create and setup model"""
    model = TemporalNAFUNet(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        encoder_channels=config.model.encoder_channels,
        use_temporal_attention=config.model.use_temporal_attention,
        num_frames=config.model.num_frames,
        attention_heads=config.model.attention_heads,
        residual_learning=config.model.residual_learning,
        predict_mask=config.model.predict_mask,
        mask_guidance=config.model.mask_guidance,
    )

    model = model.to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: TemporalNAFUNet, parameters: {params:,}")
    if config.model.predict_mask:
        print(f"  Multi-task: mask prediction enabled (w_mask={config.training.w_mask})")
        if config.model.mask_guidance != "none":
            print(f"  Mask guidance: {config.model.mask_guidance}")

    return model


def train_step(
    model: nn.Module,
    batch: Dict,
    optimizer: optim.Optimizer,
    loss_fn: ArtifactRemovalLoss,
    device: torch.device,
    config: Config,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    """Execute a single training step"""
    model.train()

    # Move to device
    clean = batch["clean"].to(device)  # [B, T, C, H, W]
    degraded = batch["degraded"].to(device)

    optimizer.zero_grad()

    # Forward pass with optional mixed precision
    use_amp = scaler is not None and device.type == "cuda"

    with autocast(enabled=use_amp):
        # Predict clean frames from degraded
        pred = model(degraded)

        # Handle mask output if model predicts it
        pred_mask = None
        if isinstance(pred, tuple):
            pred, pred_mask = pred

        # Compute losses
        losses = loss_fn(pred, clean, is_video=True)

        # Mask loss (only for overlay artifacts where mask is non-zero)
        if pred_mask is not None:
            gt_mask = batch["mask"].to(device)
            mask_loss = torch.nn.functional.mse_loss(pred_mask, gt_mask)
            losses["mask"] = config.training.w_mask * mask_loss
            losses["total"] = losses["total"] + losses["mask"]

    # Backward pass
    if use_amp:
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return {k: v.item() for k, v in losses.items()}


def infinite_dataloader(dataloader):
    """Infinite iterator over dataloader"""
    while True:
        for batch in dataloader:
            yield batch


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    loss_fn: ArtifactRemovalLoss,
    device: torch.device,
    w_mask: float = 0.3,
) -> Dict[str, float]:
    """Validation loop"""
    model.eval()

    total_losses = {}
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0

    for batch in val_loader:
        clean = batch["clean"].to(device)
        degraded = batch["degraded"].to(device)

        B, T, C, H, W = clean.shape

        # Forward
        pred = model(degraded)

        # Handle mask output if model predicts it
        pred_mask = None
        if isinstance(pred, tuple):
            pred, pred_mask = pred

        # Losses
        losses = loss_fn(pred, clean, is_video=True)

        # Mask loss (only for overlay artifacts where mask is non-zero)
        if pred_mask is not None:
            gt_mask = batch["mask"].to(device)
            mask_loss = torch.nn.functional.mse_loss(pred_mask, gt_mask)
            losses["mask"] = w_mask * mask_loss
            losses["total"] = losses["total"] + losses["mask"]

        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()
        
        # Metrics
        pred_flat = pred.view(B * T, C, H, W)
        clean_flat = clean.view(B * T, C, H, W)
        
        total_psnr += PSNRMetric.compute(pred_flat, clean_flat)
        total_ssim += SSIMMetric.compute(pred_flat, clean_flat)
        
        num_batches += 1
    
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    avg_losses["psnr"] = total_psnr / num_batches
    avg_losses["ssim"] = total_ssim / num_batches
    
    return avg_losses


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    step: int,
    best_loss: float,
    config: Config,
    path: Path,
):
    """Save training checkpoint"""
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_loss": best_loss,
        "config": {
            "model": config.model.__dict__,
            "data": config.data.__dict__,
            "training": {k: v for k, v in config.training.__dict__.items() if k != "device"},
        },
    }
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
) -> int:
    """Load training checkpoint"""
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Support both old (epoch) and new (step) checkpoints
    step = checkpoint.get("step", checkpoint.get("epoch", 0) * 1000)
    best_loss = checkpoint.get("best_loss", float("inf"))

    print(f"Loaded checkpoint from step {step}")
    return step, best_loss


def train(config: Config, args):
    """Main training function"""
    # Setup
    device = setup_device(config)
    
    # Set seed
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(config.seed)
    
    # Create directories
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Use --run-name if provided, otherwise use experiment_name with timestamp
    run_name = args.run_name or f"{config.experiment_name}_{datetime.now():%Y%m%d_%H%M%S}"
    log_dir = Path(f"./logs/{run_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Model
    print("\nSetting up model...")
    model = setup_model(config, device)
    
    # Data
    print("\nSetting up data...")
    # Convert artifact list to set if specified
    enabled_artifacts = None
    if config.data.enabled_artifacts:
        enabled_artifacts = set(config.data.enabled_artifacts)

    # Select data source
    if args.vimeo:
        print("Using Vimeo Septuplet dataset")
        train_loader, val_loader = get_vimeo_dataloaders(
            batch_size=config.training.batch_size,
            frame_size=config.data.frame_size,
            num_frames=config.data.num_frames,
            num_workers=args.workers,
            enabled_artifacts=enabled_artifacts,
            data_dir=args.data,
        )
    else:
        train_loader, val_loader = get_dataloaders(
            data_path=args.data if args.data else "./data",
            batch_size=config.training.batch_size,
            frame_size=config.data.frame_size,
            num_frames=config.data.num_frames,
            num_workers=args.workers,
            use_dummy=args.dummy,
            enabled_artifacts=enabled_artifacts,
        )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Loss
    loss_fn = ArtifactRemovalLoss(
        w_pixel=config.training.w_pixel,
        w_perceptual=config.training.w_perceptual,
        w_temporal=config.training.w_temporal,
        w_fft=config.training.w_fft,
        use_lpips=not args.no_lpips,
    ).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
    ) if config.training.use_scheduler else None
    
    # Mixed precision scaler (CUDA only)
    scaler = GradScaler() if config.training.use_amp and device.type == "cuda" else None
    
    # Tensorboard
    writer = SummaryWriter(log_dir)
    
    # Resume if checkpoint provided
    start_step = 0
    best_loss = float("inf")

    if args.resume:
        start_step, best_loss = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, device
        )

    # Save config to both log_dir and checkpoint_dir
    config_dict = {
        "run_name": run_name,
        "experiment_name": config.experiment_name,
        "model_class": "TemporalNAFUNet",
        "model": config.model.__dict__,
        "data": config.data.__dict__,
        "training": {k: v for k, v in config.training.__dict__.items()},
    }
    for dir_path in [log_dir, checkpoint_dir]:
        with open(dir_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

    # Log hyperparameters to TensorBoard
    hparams = {
        "model/class": "TemporalNAFUNet",
        "model/channels": config.model.encoder_channels[0],
        "model/depth": len(config.model.encoder_channels),
        "model/temporal_attn": config.model.use_temporal_attention,
        "model/predict_mask": config.model.predict_mask,
        "model/mask_guidance": config.model.mask_guidance,
        "model/residual": config.model.residual_learning,
        "train/batch_size": config.training.batch_size,
        "train/lr": config.training.learning_rate,
        "train/num_steps": config.training.num_steps,
        "loss/w_pixel": config.training.w_pixel,
        "loss/w_perceptual": config.training.w_perceptual,
        "loss/w_temporal": config.training.w_temporal,
        "loss/w_mask": config.training.w_mask,
    }
    writer.add_hparams(hparams, {"placeholder": 0}, run_name=".")

    # Training loop (step-based)
    print("\n" + "=" * 60)
    print(f"Starting training for {config.training.num_steps} steps...")
    print(f"  Validate every {config.training.val_every} steps")
    print(f"  Save checkpoint every {config.training.save_every} steps")
    print("=" * 60)

    train_iter = infinite_dataloader(train_loader)
    running_losses = {}
    steps_since_val = 0
    start_time = time.time()

    for step in range(start_step, config.training.num_steps):
        # Get next batch
        batch = next(train_iter)

        # Train step
        losses = train_step(model, batch, optimizer, loss_fn, device, config, scaler)

        # Accumulate running losses
        for k, v in losses.items():
            if k not in running_losses:
                running_losses[k] = 0.0
            running_losses[k] += v
        steps_since_val += 1

        # Log to TensorBoard
        if (step + 1) % config.training.log_every == 0:
            for k, v in losses.items():
                writer.add_scalar(f"train/{k}", v, step)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], step)

        # Progress
        if (step + 1) % 10 == 0:
            elapsed = time.time() - start_time
            steps_done = step - start_step + 1
            steps_per_sec = steps_done / elapsed
            eta = (config.training.num_steps - step - 1) / steps_per_sec
            print(f"\r  Step {step+1}/{config.training.num_steps} | "
                  f"Loss: {losses['total']:.4f} | "
                  f"{steps_per_sec:.1f} steps/s | "
                  f"ETA: {eta/60:.1f}m", end="", flush=True)

        # Validate
        if (step + 1) % config.training.val_every == 0:
            print()  # New line

            # Average training losses since last validation
            avg_train = {k: v / steps_since_val for k, v in running_losses.items()}

            # Validate
            val_losses = validate(model, val_loader, loss_fn, device, config.training.w_mask)

            # Print summary
            print(f"  [Step {step+1}] Train: {avg_train['total']:.4f} | "
                  f"Val: {val_losses['total']:.4f} | "
                  f"PSNR: {val_losses['psnr']:.2f} dB | "
                  f"SSIM: {val_losses['ssim']:.4f}")

            # Log validation to TensorBoard
            for k, v in val_losses.items():
                writer.add_scalar(f"val/{k}", v, step)

            # Log sample images
            log_sample_images(writer, model, val_loader, device, step)

            # Scheduler step
            if scheduler:
                scheduler.step(val_losses["total"])

            # Save best checkpoint
            if val_losses["total"] < best_loss:
                best_loss = val_losses["total"]
                save_checkpoint(
                    model, optimizer, scheduler, step, best_loss, config,
                    checkpoint_dir / "best.pt"
                )

            # Reset running losses
            running_losses = {}
            steps_since_val = 0

        # Save periodic checkpoint
        if (step + 1) % config.training.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, step, best_loss, config,
                checkpoint_dir / f"step_{step + 1}.pt"
            )

    # Save final model
    print()
    save_checkpoint(
        model, optimizer, scheduler, config.training.num_steps - 1, best_loss, config,
        checkpoint_dir / "final.pt"
    )

    writer.close()
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train artifact removal model")

    # Data
    parser.add_argument("--data", type=str, default=None,
                        help="Path to data directory")
    parser.add_argument("--dummy", action="store_true",
                        help="Use dummy data for testing")
    parser.add_argument("--vimeo", action="store_true",
                        help="Use Vimeo Septuplet dataset (run prepare_data.sh first)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--artifacts", type=str, default=None,
                        help="Comma-separated artifact types to enable (e.g., 'corner_logo,gaussian_noise')")

    # Training
    parser.add_argument("--steps", type=int, default=None,
                        help="Total training steps (default: 50000)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--val-every", type=int, default=None,
                        help="Validate every N steps (default: 500)")
    parser.add_argument("--save-every", type=int, default=None,
                        help="Save checkpoint every N steps (default: 2000)")
    parser.add_argument("--log-every", type=int, default=None,
                        help="Log to TensorBoard every N steps (default: 50)")

    # Model
    parser.add_argument("--no-temporal", action="store_true",
                        help="Disable temporal attention")
    parser.add_argument("--lite", action="store_true",
                        help="Use lightweight model")
    parser.add_argument("--residual", action="store_true",
                        help="Use residual learning (predict residual, add to input)")
    parser.add_argument("--predict-mask", action="store_true",
                        help="Also predict artifact mask (multi-task learning)")
    parser.add_argument("--mask-guidance", type=str, default="none",
                        choices=["none", "modulation"],
                        help="How predicted mask guides inpainting (default: none)")
    parser.add_argument("--channels", type=int, default=None,
                        help="Base channel width (default: 32). Model uses [c, 2c, 4c, 8c]")
    parser.add_argument("--depth", type=int, default=None,
                        help="Number of encoder/decoder stages (default: 4)")

    # Loss
    parser.add_argument("--no-lpips", action="store_true",
                        help="Use VGG instead of LPIPS for perceptual loss")
    parser.add_argument("--pixel-only", action="store_true",
                        help="Use only pixel loss (disable perceptual, good for dummy data)")

    # Checkpoints and logging
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for this run (used in TensorBoard logs)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory for saving checkpoints (default: ./checkpoints)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device to use")
    parser.add_argument("--amp", action="store_true",
                        help="Enable automatic mixed precision (faster but may cause inf loss)")

    args = parser.parse_args()
    
    # Get config and apply overrides
    config = get_config()

    if args.steps:
        config.training.num_steps = args.steps
    if args.val_every:
        config.training.val_every = args.val_every
    if args.save_every:
        config.training.save_every = args.save_every
    if args.log_every:
        config.training.log_every = args.log_every
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.no_temporal:
        config.model.use_temporal_attention = False
    if args.device:
        config.training.device = args.device
    if args.residual:
        config.model.residual_learning = True
    if args.predict_mask:
        config.model.predict_mask = True
    if args.mask_guidance != "none":
        config.model.mask_guidance = args.mask_guidance
        config.model.predict_mask = True  # mask_guidance requires predict_mask
    if args.artifacts:
        config.data.enabled_artifacts = args.artifacts.split(",")
    if args.amp:
        config.training.use_amp = True
    if args.checkpoint_dir:
        config.training.checkpoint_dir = args.checkpoint_dir
    if args.pixel_only:
        config.training.w_perceptual = 0.0
        config.training.w_fft = 0.0
    if args.channels:
        # Build channel progression: [c, 2c, 4c, 8c, ...]
        depth = args.depth or len(config.model.encoder_channels)
        config.model.encoder_channels = [args.channels * (2 ** i) for i in range(depth)]
    elif args.depth:
        # Keep base channels, adjust depth
        base = config.model.encoder_channels[0]
        config.model.encoder_channels = [base * (2 ** i) for i in range(args.depth)]

    # Print config
    print("\n" + "=" * 60)
    print("Video Artifact Removal Training (Step-based)")
    print("=" * 60)
    print(f"Experiment: {config.experiment_name}")
    print(f"Total steps: {config.training.num_steps}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Validate every: {config.training.val_every} steps")
    print(f"Save every: {config.training.save_every} steps")
    print(f"Frame size: {config.data.frame_size}")
    print(f"Num frames: {config.data.num_frames}")
    print(f"Encoder channels: {config.model.encoder_channels}")
    print(f"Temporal attention: {config.model.use_temporal_attention}")
    print(f"Residual learning: {config.model.residual_learning}")
    print(f"Predict mask: {config.model.predict_mask}")
    print(f"Mask guidance: {config.model.mask_guidance}")
    print(f"Enabled artifacts: {config.data.enabled_artifacts or 'overlay types (default)'}")
    print(f"Dataset: {'Vimeo Septuplet' if args.vimeo else ('dummy' if args.dummy else 'custom')}")
    
    # Train
    train(config, args)


if __name__ == "__main__":
    main()
