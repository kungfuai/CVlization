"""SAM LoRA Fine-tuning -- training session orchestrator.

Run with: python -m src.training.training_session [args]
"""

from pathlib import Path
from statistics import mean

import monai
import numpy as np
import torch
from torch.optim import Adam

from src.data.dataset_builder import DatasetBuilder
from src.models.lora import LoRA_sam
from src.training.trainer import Trainer
from src.training.training_config import TrainingConfig


def main():
    config = TrainingConfig.parse_args()

    print("=" * 70)
    print("SAM LoRA Fine-tuning")
    print("=" * 70)

    # ---- 1. Model ----
    print("\n[1/4] Loading SAM ViT-B and applying LoRA...")
    from segment_anything import build_sam_vit_b

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config.checkpoint:
        sam = build_sam_vit_b(checkpoint=config.checkpoint)
    else:
        # Download default checkpoint to centralized cache
        import os

        cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        ckpt_path = Path(cache_home) / "cvlization" / "models" / "sam" / "sam_vit_b_01ec64.pth"
        if not ckpt_path.exists():
            print("  Downloading SAM ViT-B checkpoint...")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            import urllib.request

            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            urllib.request.urlretrieve(url, str(ckpt_path))
            print(f"  Cached to {ckpt_path}")
        else:
            print(f"  Using cached checkpoint: {ckpt_path}")
        sam = build_sam_vit_b(checkpoint=str(ckpt_path))

    sam_lora = LoRA_sam(sam, config.rank)
    model = sam_lora.sam
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    print(f"  LoRA rank: {config.rank}")

    # ---- 2. Dataset ----
    print("\n[2/4] Building data loaders...")
    builder = DatasetBuilder(config, model)
    train_loader, val_loader = builder.build()

    # ---- 3. Training setup ----
    print("\n[3/4] Setting up training...")

    optimizer = Adam(model.image_encoder.parameters(), lr=config.lr, weight_decay=0)
    loss_fn = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

    trainer = Trainer(model, optimizer, loss_fn, device)

    wandb_run = None
    if config.wandb:
        import wandb

        wandb_run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "rank": config.rank,
                "lr": config.lr,
                "hf_dataset": config.hf_dataset,
                "train_samples": len(train_loader.dataset),
            },
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 4. Training loop ----
    print(f"\n[4/4] Training for {config.epochs} epochs...")

    best_val_loss = float("inf")

    # Log step-0 val images (before any training)
    if wandb_run and val_loader is not None:
        np.random.seed(0)
        panels = trainer.visualize_predictions(val_loader.dataset)
        wandb_run.log({
            "val/predictions": [
                wandb.Image(p, caption=f"sample {i}: input | GT | pred")
                for i, p in enumerate(panels)
            ],
        })

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")

        # Train
        train_losses = trainer.train_epoch(train_loader)
        avg_train = mean(train_losses)
        print(f"  Train loss: {avg_train:.4f}")

        log = {"epoch": epoch + 1, "train/loss": avg_train}

        # Validate
        if val_loader is not None:
            val_losses = trainer.validate_epoch(val_loader)
            avg_val = mean(val_losses)
            print(f"  Val loss:   {avg_val:.4f}")
            log["val/loss"] = avg_val

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_path = output_dir / f"lora_rank{config.rank}_best.safetensors"
                sam_lora.save_lora_parameters(str(best_path))
                print(f"  Saved best model -> {best_path}")

        # Visualize predictions on validation set
        if wandb_run and val_loader is not None:
            np.random.seed(0)
            panels = trainer.visualize_predictions(val_loader.dataset)
            log["val/predictions"] = [
                wandb.Image(p, caption=f"sample {i}: input | GT | pred")
                for i, p in enumerate(panels)
            ]

        if wandb_run:
            wandb_run.log(log)

    # Save final weights
    final_path = output_dir / f"lora_rank{config.rank}.safetensors"
    sam_lora.save_lora_parameters(str(final_path))
    print(f"\nSaved final LoRA weights -> {final_path}")

    if wandb_run:
        wandb_run.finish()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
