"""
Training Script for Video Artifact Removal

Usage:
    python train.py                    # Train with dummy data
    python train.py --data ./videos    # Train with real data
    python train.py --resume checkpoint.pt  # Resume training
    python train.py --residual         # Use residual learning
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

from config import Config, get_config
from model import ArtifactRemovalNet, ArtifactRemovalNetLite
from losses import ArtifactRemovalLoss, PSNRMetric, SSIMMetric
from dataset import get_dataloaders, get_vimeo_dataloaders


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
    model = ArtifactRemovalNet(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        encoder_channels=config.model.encoder_channels,
        use_temporal_attention=config.model.use_temporal_attention,
        num_frames=config.model.num_frames,
        attention_heads=config.model.attention_heads,
        residual_learning=config.model.residual_learning,
        predict_mask=config.model.predict_mask,
    )

    model = model.to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")

    return model


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: optim.Optimizer,
    loss_fn: ArtifactRemovalLoss,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    epoch: int = 0,
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    
    total_losses = {}
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        clean = batch["clean"].to(device)  # [B, T, C, H, W]
        degraded = batch["degraded"].to(device)

        B, T, C, H, W = clean.shape

        optimizer.zero_grad()

        # Forward pass with optional mixed precision
        use_amp = scaler is not None and device.type == "cuda"

        with autocast(enabled=use_amp):
            # Predict clean frames from degraded
            pred = model(degraded)

            # Handle mask output if model predicts it
            if isinstance(pred, tuple):
                pred, pred_mask = pred

            # Compute losses
            losses = loss_fn(pred, clean, is_video=True)
        
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
        
        # Accumulate losses
        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()
        num_batches += 1
        
        # Progress
        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (batch_idx + 1) * (len(train_loader) - batch_idx - 1)
            print(f"\r  Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {losses['total'].item():.4f} | "
                  f"ETA: {eta:.0f}s", end="", flush=True)
    
    print()  # New line after progress
    
    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    avg_losses["time"] = time.time() - start_time
    
    return avg_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    loss_fn: ArtifactRemovalLoss,
    device: torch.device,
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
        if isinstance(pred, tuple):
            pred, pred_mask = pred

        # Losses
        losses = loss_fn(pred, clean, is_video=True)
        
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
    epoch: int,
    best_loss: float,
    config: Config,
    path: Path,
):
    """Save training checkpoint"""
    checkpoint = {
        "epoch": epoch,
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
    
    epoch = checkpoint["epoch"]
    best_loss = checkpoint.get("best_loss", float("inf"))
    
    print(f"Loaded checkpoint from epoch {epoch}")
    return epoch, best_loss


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
    
    log_dir = Path(f"./logs/{config.experiment_name}_{datetime.now():%Y%m%d_%H%M%S}")
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
        verbose=True,
    ) if config.training.use_scheduler else None
    
    # Mixed precision scaler (CUDA only)
    scaler = GradScaler() if config.training.use_amp and device.type == "cuda" else None
    
    # Tensorboard
    writer = SummaryWriter(log_dir)
    
    # Resume if checkpoint provided
    start_epoch = 0
    best_loss = float("inf")
    
    if args.resume:
        start_epoch, best_loss = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, device
        )
        start_epoch += 1
    
    # Save config
    with open(log_dir / "config.json", "w") as f:
        json.dump({
            "model": config.model.__dict__,
            "data": config.data.__dict__,
            "training": {k: v for k, v in config.training.__dict__.items()},
        }, f, indent=2, default=str)
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    for epoch in range(start_epoch, config.training.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")
        print("-" * 40)
        
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler, epoch
        )
        
        print(f"  Train Loss: {train_losses['total']:.4f} | "
              f"Pixel: {train_losses['pixel']:.4f} | "
              f"Perceptual: {train_losses.get('perceptual', 0):.4f} | "
              f"Time: {train_losses['time']:.1f}s")
        
        # Validate
        val_losses = validate(model, val_loader, loss_fn, device)
        
        print(f"  Val Loss: {val_losses['total']:.4f} | "
              f"PSNR: {val_losses['psnr']:.2f} dB | "
              f"SSIM: {val_losses['ssim']:.4f}")
        
        # Scheduler step
        if scheduler:
            scheduler.step(val_losses["total"])
        
        # Tensorboard logging
        for k, v in train_losses.items():
            if k != "time":
                writer.add_scalar(f"train/{k}", v, epoch)
        
        for k, v in val_losses.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        
        # Save checkpoint
        is_best = val_losses["total"] < best_loss
        if is_best:
            best_loss = val_losses["total"]
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_loss, config,
                checkpoint_dir / "best.pt"
            )
        
        if (epoch + 1) % config.training.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_loss, config,
                checkpoint_dir / f"epoch_{epoch + 1}.pt"
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, config.training.num_epochs - 1, best_loss, config,
        checkpoint_dir / "final.pt"
    )
    
    writer.close()
    print("\n" + "=" * 60)
    print("Training complete!")
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
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")

    # Model
    parser.add_argument("--no-temporal", action="store_true",
                        help="Disable temporal attention")
    parser.add_argument("--lite", action="store_true",
                        help="Use lightweight model")
    parser.add_argument("--residual", action="store_true",
                        help="Use residual learning (predict residual, add to input)")
    parser.add_argument("--predict-mask", action="store_true",
                        help="Also predict artifact mask")

    # Loss
    parser.add_argument("--no-lpips", action="store_true",
                        help="Use VGG instead of LPIPS for perceptual loss")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device to use")

    args = parser.parse_args()
    
    # Get config and apply overrides
    config = get_config()

    if args.epochs:
        config.training.num_epochs = args.epochs
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
    if args.artifacts:
        config.data.enabled_artifacts = args.artifacts.split(",")

    # Print config
    print("\n" + "=" * 60)
    print("Video Artifact Removal Training")
    print("=" * 60)
    print(f"Experiment: {config.experiment_name}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Frame size: {config.data.frame_size}")
    print(f"Num frames: {config.data.num_frames}")
    print(f"Temporal attention: {config.model.use_temporal_attention}")
    print(f"Residual learning: {config.model.residual_learning}")
    print(f"Enabled artifacts: {config.data.enabled_artifacts or 'overlay types (default)'}")
    print(f"Dataset: {'Vimeo Septuplet' if args.vimeo else ('dummy' if args.dummy else 'custom')}")
    
    # Train
    train(config, args)


if __name__ == "__main__":
    main()
