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
import torch.nn.functional as F
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
from model import TemporalNAFUNet, NAFUNetLite, ExplicitCompositeNet
from model_lama import LamaWithMask
from model_elir import ElirWithMask
from losses import ArtifactRemovalLoss, PSNRMetric, SSIMMetric
from dataset import get_dataloaders, get_vimeo_dataloaders
from pretrained import get_pretrained_path


class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine annealing."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self, val_loss=None):
        """Update learning rate. val_loss is ignored (for API compatibility)."""
        self.current_step += 1
        lr = self._get_lr()
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group["lr"] = lr

    def _get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lrs[0] * self.current_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            import math
            return self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict):
        self.current_step = state_dict["current_step"]


def mask_loss_fn(
    pred: torch.Tensor,
    target: torch.Tensor,
    dice_weight: float = 1.0,
    l1_weight: float = 1.0,
    dice_threshold: float = 0.1,
    eps: float = 1e-6,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.9,  # High alpha for rare positive class (artifacts ~1-2%)
) -> torch.Tensor:
    """
    Combined Dice + L1 loss for soft masks, with optional Focal loss.

    Default (use_focal=False):
    - Dice: Uses thresholded (binary) gt_mask for "where" (location)
    - L1: Uses soft gt_mask for "how much" (intensity)

    With use_focal=True:
    - Dice: Same as above
    - Focal: Down-weights easy negatives, better for class imbalance

    Args:
        pred: Predicted mask [B, T, 1, H, W] or [B*T, 1, H, W]
        target: Ground truth soft mask, same shape
        dice_weight: Weight for Dice loss
        l1_weight: Weight for L1/Focal loss
        dice_threshold: Threshold to binarize gt_mask for Dice
        eps: Small value for numerical stability
        use_focal: If True, use Focal loss instead of L1 (better for class imbalance)
        focal_gamma: Focusing parameter (higher = more focus on hard examples)
        focal_alpha: Weight for positive class (higher for rare class)
    """
    if use_focal:
        # Focal loss for class imbalance
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target > 0.5, pred, 1 - pred)
        focal_mod = (1 - pt) ** focal_gamma
        alpha_t = torch.where(target > 0.5, focal_alpha, 1 - focal_alpha)
        secondary_loss = (alpha_t * focal_mod * bce).mean()
    else:
        # L1 loss on soft mask
        secondary_loss = torch.abs(pred - target).mean()

    # Dice loss on binary mask
    target_binary = (target > dice_threshold).float()

    # Only compute Dice if there are positive pixels (overlay artifacts)
    if target_binary.sum() > 0:
        intersection = (pred * target_binary).sum()
        dice = 2 * intersection / (pred.sum() + target_binary.sum() + eps)
        dice_loss = 1 - dice
    else:
        # No artifact in mask (degradation type) - skip Dice
        dice_loss = torch.tensor(0.0, device=pred.device)

    return dice_weight * dice_loss + l1_weight * secondary_loss


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
    for each frame in the sequence. Also logs masks if model predicts them.
    """
    model.eval()

    # Get first batch
    batch = next(iter(val_loader))
    clean = batch["clean"].to(device)  # [B, T, C, H, W]
    degraded = batch["degraded"].to(device)
    gt_mask = batch["mask"].to(device)  # [B, T, 1, H, W]

    B, T, C, H, W = clean.shape
    num_samples = min(num_samples, B)

    with torch.no_grad():
        pred = model(degraded)
        pred_mask = None
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

    # Log mask comparison (gt | pred) if model predicts masks
    if pred_mask is not None:
        gt_mask_vis = gt_mask[0, mid_t].expand(3, -1, -1)  # [3, H, W]
        pred_mask_vis = pred_mask[0, mid_t].clamp(0, 1).expand(3, -1, -1)  # [3, H, W]
        mask_comparison = torch.stack([gt_mask_vis, pred_mask_vis], dim=0)
        grid_masks = vutils.make_grid(mask_comparison, nrow=2, padding=4, normalize=False)
        writer.add_image("masks/gt_vs_pred", grid_masks, step)


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


def setup_model(config: Config, device: torch.device, model_type: str = "temporal_nafunet", pretrained_path: str = None, use_mask_decoder: bool = False, use_mask_unet: bool = False, detach_mask_features: bool = True) -> nn.Module:
    """Create and setup model"""
    if model_type == "temporal_nafunet":
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
        model_name = "TemporalNAFUNet"
    elif model_type == "composite":
        model = ExplicitCompositeNet(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            encoder_channels=config.model.encoder_channels,
            use_temporal_attention=config.model.use_temporal_attention,
            num_frames=config.model.num_frames,
            attention_heads=config.model.attention_heads,
        )
        model_name = "ExplicitCompositeNet"
    elif model_type == "lama":
        # LaMa uses base_channels instead of encoder_channels
        base_channels = config.model.encoder_channels[0] if config.model.encoder_channels else 64
        model = LamaWithMask(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            base_channels=base_channels,
            num_downs=3,
            num_ffc_blocks=9,
            use_temporal_attention=config.model.use_temporal_attention,
            num_frames=config.model.num_frames,
            attention_heads=config.model.attention_heads,
        )
        model_name = "LamaWithMask"

        # Load pretrained weights if provided
        if pretrained_path:
            # Handle "auto" - download from centralized cache
            if pretrained_path == "auto":
                pretrained_path = get_pretrained_path("lama", download=True)
            if pretrained_path:
                print(f"Loading pretrained LaMa weights from: {pretrained_path}")
                model.load_pretrained(str(pretrained_path), strict=False)
    elif model_type == "elir":
        # ELIR uses latent channels and flow steps
        base_channels = config.model.encoder_channels[0] if config.model.encoder_channels else 64
        model = ElirWithMask(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            latent_channels=16,  # Standard for TAESD
            hidden_channels=base_channels,
            flow_hidden_channels=base_channels * 2,
            k_steps=3,  # 3 NFE like original ELIR
            use_temporal_attention=config.model.use_temporal_attention,
            num_frames=config.model.num_frames,
            attention_heads=config.model.attention_heads,
            use_mask_decoder=use_mask_decoder,  # Option B: encoder features for mask
            use_mask_unet=use_mask_unet,  # Option C: full NAFNet UNet for mask
            detach_mask_features=detach_mask_features,  # False for training from scratch
        )
        if use_mask_unet:
            model_name = "ElirWithMask (MaskUNet)"
        elif use_mask_decoder:
            model_name = "ElirWithMask (MaskDecoder)"
            if not detach_mask_features:
                model_name += " (joint training)"
        else:
            model_name = "ElirWithMask"

        # Load pretrained ELIR weights if provided
        if pretrained_path:
            # Handle "auto" - download from centralized cache
            if pretrained_path == "auto":
                pretrained_path = get_pretrained_path("elir", download=True)
            if pretrained_path:
                print(f"Loading pretrained ELIR weights from: {pretrained_path}")
                model.load_elir_weights(str(pretrained_path), strict=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name}, parameters: {params:,}")
    if model_type == "temporal_nafunet" and config.model.predict_mask:
        print(f"  Multi-task: mask prediction enabled (w_mask={config.training.w_mask})")
        if config.model.mask_guidance != "none":
            print(f"  Mask guidance: {config.model.mask_guidance}")
    elif model_type == "composite":
        print(f"  Explicit composite: output = input*(1-mask) + inpainted*mask")

    return model


def train_step(
    model: nn.Module,
    batch: Dict,
    optimizer: optim.Optimizer,
    loss_fn: ArtifactRemovalLoss,
    device: torch.device,
    config: Config,
    scaler: Optional[GradScaler] = None,
    mask_weight: float = 1.0,
    is_composite: bool = False,
    is_elir: bool = False,
    use_focal_mask_loss: bool = False,
) -> Dict[str, float]:
    """Execute a single training step"""
    model.train()

    # Move to device
    clean = batch["clean"].to(device)  # [B, T, C, H, W]
    degraded = batch["degraded"].to(device)
    gt_mask = batch["mask"].to(device)  # [B, T, 1, H, W]

    optimizer.zero_grad()

    # Forward pass with optional mixed precision
    use_amp = scaler is not None and device.type == "cuda"

    with autocast(enabled=use_amp):
        # Predict clean frames from degraded
        # ELIR has special forward that takes clean for flow matching loss
        if is_elir:
            result = model(degraded, clean=clean)
            pred = result['output']
            pred_mask = result['pred_mask']
            inpainted = result['restored']
            flow_loss = result['flow_loss']
        elif is_composite:
            pred, pred_mask, inpainted = model(degraded, return_inpainted=True)
            flow_loss = None
        else:
            pred = model(degraded)
            inpainted = None
            flow_loss = None

            # Handle mask output if model predicts it
            pred_mask = None
            if isinstance(pred, tuple):
                pred, pred_mask = pred

        # Compute losses (with optional mask weighting)
        losses = loss_fn(pred, clean, is_video=True, mask=gt_mask, mask_weight=mask_weight)

        # Mask loss (Dice + L1 or Dice + Focal)
        if pred_mask is not None:
            mask_loss = mask_loss_fn(pred_mask, gt_mask, use_focal=use_focal_mask_loss)
            losses["mask"] = config.training.w_mask * mask_loss
            losses["total"] = losses["total"] + losses["mask"]

        # Flow matching loss for ELIR (with weight and NaN check)
        if flow_loss is not None:
            # Check for NaN/inf
            if torch.isnan(flow_loss) or torch.isinf(flow_loss):
                print(f"WARNING: flow_loss is {flow_loss.item()}, skipping")
                flow_loss = torch.tensor(0.0, device=device)
            # Weight flow loss (can be tuned via config if needed)
            losses["flow"] = 0.1 * flow_loss  # Scale down to balance with other losses
            losses["total"] = losses["total"] + losses["flow"]

        # Auxiliary inpaint loss for composite/ELIR model
        # This supervises inpainted directly in mask regions to prevent degenerate solutions
        if inpainted is not None:
            # For ELIR: detach inpainted so flow model only gets gradients from flow_loss
            # (inpainted goes through flow_loop, we don't want inpaint_loss to train flow)
            inpainted_for_loss = inpainted.detach() if is_elir else inpainted
            # Loss: inpainted should match clean in artifact (mask) regions
            # Weight by mask so we only care about artifact regions
            inpaint_diff = (inpainted_for_loss - clean).abs() * gt_mask
            inpaint_loss = inpaint_diff.sum() / (gt_mask.sum() + 1e-6)
            losses["inpaint"] = 0.5 * inpaint_loss  # Weight for auxiliary loss
            losses["total"] = losses["total"] + losses["inpaint"]

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
    mask_weight: float = 1.0,
    use_focal_mask_loss: bool = False,
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
        gt_mask = batch["mask"].to(device)

        B, T, C, H, W = clean.shape

        # Forward
        pred = model(degraded)

        # Handle mask output if model predicts it
        pred_mask = None
        if isinstance(pred, tuple):
            pred, pred_mask = pred

        # Losses (with optional mask weighting)
        losses = loss_fn(pred, clean, is_video=True, mask=gt_mask, mask_weight=mask_weight)

        # Mask loss (Dice + L1 or Dice + Focal)
        if pred_mask is not None:
            mask_loss = mask_loss_fn(pred_mask, gt_mask, use_focal=use_focal_mask_loss)
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
    detach_mask_features = not getattr(args, 'no_detach_mask_features', False)
    use_mask_unet = getattr(args, 'mask_unet', False)
    model = setup_model(config, device, args.model, args.pretrained, args.use_mask_decoder, use_mask_unet, detach_mask_features)
    
    # Data
    print("\nSetting up data...")
    # Convert artifact list to set if specified
    enabled_artifacts = None
    if config.data.enabled_artifacts:
        enabled_artifacts = set(config.data.enabled_artifacts)

    # Parse multi-scale if provided
    multi_scale = None
    if args.multi_scale:
        multi_scale = [int(s.strip()) for s in args.multi_scale.split(",")]
        print(f"Multi-scale training enabled: {multi_scale}")

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
            preserve_aspect_ratio=config.data.preserve_aspect_ratio,
            size_scale=args.size_scale,
            multi_scale=multi_scale,
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
            size_scale=args.size_scale,
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
    
    # Handle pretrained model finetuning
    if args.freeze_pretrained and args.model == "elir":
        # Freeze all except mask_head
        for name, param in model.named_parameters():
            if "mask_head" not in name:
                param.requires_grad = False
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"Frozen pretrained weights. Training only mask_head ({sum(p.numel() for p in trainable_params):,} params)")
    else:
        trainable_params = model.parameters()

    # Use lower learning rate for finetuning pretrained models
    lr = config.training.learning_rate
    if args.finetune_lr is not None:
        lr = args.finetune_lr
        print(f"Using finetune learning rate: {lr}")
    elif args.pretrained and args.model == "elir" and not args.freeze_pretrained:
        # Default to lower LR for ELIR finetuning
        lr = 1e-5
        print(f"Auto-reduced learning rate for ELIR finetuning: {lr}")

    # Optimizer
    optimizer = optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=config.training.weight_decay,
    )
    
    # Scheduler
    if args.warmup_cosine:
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=args.warmup_steps,
            total_steps=config.training.num_steps,
        )
        print(f"Using warmup ({args.warmup_steps} steps) + cosine annealing scheduler")
    elif config.training.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience,
        )
    else:
        scheduler = None
    
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
    model_class = {
        "temporal_nafunet": "TemporalNAFUNet",
        "composite": "ExplicitCompositeNet",
        "lama": "LamaWithMask",
        "elir": "ElirWithMask",
    }.get(args.model, args.model)
    config_dict = {
        "run_name": run_name,
        "experiment_name": config.experiment_name,
        "model_type": args.model,
        "model_class": model_class,
        "model": config.model.__dict__,
        "data": config.data.__dict__,
        "training": {k: v for k, v in config.training.__dict__.items()},
        "mask_weight": args.mask_weight,
    }
    for dir_path in [log_dir, checkpoint_dir]:
        with open(dir_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

    # Log hyperparameters to TensorBoard
    hparams = {
        "model/type": args.model,
        "model/class": model_class,
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
        "loss/mask_weight": args.mask_weight,
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
        is_composite = args.model in ("composite", "lama")
        is_elir = args.model == "elir"
        use_focal = getattr(args, 'focal_mask_loss', False)
        losses = train_step(model, batch, optimizer, loss_fn, device, config, scaler, args.mask_weight, is_composite, is_elir, use_focal)

        # Step-based scheduler (warmup+cosine)
        if scheduler and isinstance(scheduler, WarmupCosineScheduler):
            scheduler.step()

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
            val_losses = validate(model, val_loader, loss_fn, device, config.training.w_mask, args.mask_weight, use_focal)

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

            # Scheduler step (only for ReduceLROnPlateau, warmup+cosine steps every iteration)
            if scheduler and not isinstance(scheduler, WarmupCosineScheduler):
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
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Number of frames per sample (default: 5, use 7 for Vimeo Septuplet)")
    parser.add_argument("--artifacts", type=str, default=None,
                        help="Comma-separated artifact types to enable (e.g., 'corner_logo,gaussian_noise')")
    parser.add_argument("--size-scale", type=float, default=1.0,
                        help="Scale factor for artifact sizes (default: 1.0). "
                             "Use 0.5 for smaller artifacts, 2.0 for larger.")
    parser.add_argument("--preserve-aspect", action="store_true",
                        help="Preserve aspect ratio (resize to max side, no crop)")
    parser.add_argument("--multi-scale", type=str, default=None,
                        help="Comma-separated max sizes for multi-scale training (e.g., '256,320,384'). "
                             "Randomly samples a scale per example, resizes preserving aspect ratio. "
                             "Forces batch_size=1 during training due to variable sizes.")
    parser.add_argument("--frame-size", type=int, default=None,
                        help="Frame size (square, e.g., 256 or 384). Default: 256")

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
                        choices=["none", "modulation", "concat", "residual", "skip_gate", "attn_gate"],
                        help="How predicted mask guides inpainting: "
                             "none (no guidance), "
                             "modulation (feature modulation), "
                             "concat (concatenate mask to features), "
                             "residual (apply residual only in mask regions), "
                             "skip_gate (suppress encoder skip features in mask regions), "
                             "attn_gate (boost attention to artifact regions)")
    parser.add_argument("--model", type=str, default="temporal_nafunet",
                        choices=["temporal_nafunet", "composite", "lama", "elir"],
                        help="Model architecture: "
                             "temporal_nafunet (default, with optional mask guidance), "
                             "composite (explicit alpha blending, always predicts mask), "
                             "lama (LaMa with FFC blocks, temporal attention, mask prediction), "
                             "elir (ELIR flow matching in latent space, 3-step inference)")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained weights (for LaMa or ELIR model)")
    parser.add_argument("--freeze-pretrained", action="store_true",
                        help="Freeze pretrained weights, only train mask_head (for ELIR)")
    parser.add_argument("--finetune-lr", type=float, default=None,
                        help="Learning rate for finetuning pretrained model (default: 1e-5)")
    parser.add_argument("--use-mask-decoder", action="store_true",
                        help="Use MaskDecoder with encoder features instead of simple MaskHead (for ELIR).")
    parser.add_argument("--mask-unet", action="store_true",
                        help="Use full NAFNet-style UNet for mask prediction (for ELIR). "
                             "Same architecture as ExplicitCompositeNet's encoder-decoder. "
                             "More powerful than MaskHead but adds ~2.3M params.")
    parser.add_argument("--no-detach-mask-features", action="store_true",
                        help="Don't detach encoder features for MaskDecoder (for ELIR). "
                             "Use when training from scratch so mask loss can train encoder. "
                             "With pretrained encoder, detaching is fine since encoder already has good features.")
    parser.add_argument("--channels", type=int, default=None,
                        help="Base channel width (default: 32). Model uses [c, 2c, 4c, 8c]")
    parser.add_argument("--depth", type=int, default=None,
                        help="Number of encoder/decoder stages (default: 4)")

    # Loss
    parser.add_argument("--no-lpips", action="store_true",
                        help="Use VGG instead of LPIPS for perceptual loss")
    parser.add_argument("--pixel-only", action="store_true",
                        help="Use only pixel loss (disable perceptual, good for dummy data)")
    parser.add_argument("--mask-weight", type=float, default=1.0,
                        help="Weight multiplier for loss in artifact (mask) regions. "
                             "1.0 = equal weighting everywhere (default). "
                             "5.0 = artifact regions contribute 5x more to loss. "
                             "Use higher values (3-10) to focus learning on artifact removal "
                             "when artifacts cover small portion of image (<15%%).")
    parser.add_argument("--focal-mask-loss", action="store_true",
                        help="Use Focal loss instead of L1 for mask prediction. "
                             "Better for class imbalance when mask coverage is low (<5%%).")

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

    # Scheduler
    parser.add_argument("--warmup-cosine", action="store_true",
                        help="Use warmup + cosine annealing scheduler instead of ReduceLROnPlateau")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Warmup steps for --warmup-cosine (default: 1000)")

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
    if args.num_frames:
        config.data.num_frames = args.num_frames
        config.model.num_frames = args.num_frames
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
    if args.preserve_aspect:
        config.data.preserve_aspect_ratio = True
    if args.frame_size:
        config.data.frame_size = (args.frame_size, args.frame_size)
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
    # Composite/LaMa/ELIR models need higher mask_weight by default to prevent degenerate solutions
    if args.model in ("composite", "lama", "elir") and args.mask_weight == 1.0:
        args.mask_weight = 5.0
        print(f"Note: Using mask_weight=5.0 for {args.model} model (required for stable training)")

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
    print(f"Model: {args.model}")
    print(f"Encoder channels: {config.model.encoder_channels}")
    print(f"Temporal attention: {config.model.use_temporal_attention}")
    if args.model == "temporal_nafunet":
        print(f"Residual learning: {config.model.residual_learning}")
        print(f"Predict mask: {config.model.predict_mask}")
        print(f"Mask guidance: {config.model.mask_guidance}")
    elif args.model == "composite":
        print(f"Architecture: explicit composite (input*(1-mask) + inpainted*mask)")
    elif args.model == "lama":
        print(f"Architecture: LaMa with FFC blocks + temporal attention + mask prediction")
        if args.pretrained:
            print(f"Pretrained weights: {args.pretrained}")
    elif args.model == "elir":
        print(f"Architecture: ELIR flow matching (3-step latent ODE)")
        if args.pretrained:
            print(f"Pretrained weights: {args.pretrained}")
    if args.mask_weight != 1.0:
        print(f"Mask weight: {args.mask_weight}x (artifact regions weighted more in loss)")
    if args.size_scale != 1.0:
        print(f"Artifact size scale: {args.size_scale}x")
    print(f"Enabled artifacts: {config.data.enabled_artifacts or 'overlay types (default)'}")
    print(f"Dataset: {'Vimeo Septuplet' if args.vimeo else ('dummy' if args.dummy else 'custom')}")
    
    # Train
    train(config, args)


if __name__ == "__main__":
    main()
