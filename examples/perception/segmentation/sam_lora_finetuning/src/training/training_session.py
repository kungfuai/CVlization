"""SAM LoRA Fine-tuning -- training session orchestrator.

Run with: python -m src.training.training_session [args]
"""

import os
from pathlib import Path
from statistics import mean

import monai
import torch
from torch.optim import Adam

from src.data.dataset_builder import DatasetBuilder
from src.models.lora import LoRA_sam
from src.training.trainer import Trainer
from src.training.training_config import TrainingConfig


class TrainingSession:
    """Orchestrates model loading, data, optimizer, logging, and the train loop."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.wandb_run = None
        self.sam_lora = None
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.trainer = None

    def start(self):
        self.configure_device()
        self.create_model()
        self.create_dataloaders()
        self.create_optimizer()
        self.configure_logging()
        self.create_trainer()
        self.run_training_loop()
        self.finish()

    def configure_device(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def create_model(self):
        print("\n[1/4] Loading SAM ViT-B and applying LoRA...")
        from segment_anything import build_sam_vit_b

        config = self.config

        if config.checkpoint:
            sam = build_sam_vit_b(checkpoint=config.checkpoint)
        else:
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

        self.sam_lora = LoRA_sam(sam, config.rank)
        self.model = self.sam_lora.sam
        self.model.to(self.device)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        print(f"  LoRA rank: {config.rank}")

    def create_dataloaders(self):
        print("\n[2/4] Building data loaders...")
        builder = DatasetBuilder(self.config, self.model)
        self.train_loader, self.val_loader = builder.build()

    def create_optimizer(self):
        print("\n[3/4] Setting up training...")
        self.optimizer = Adam(
            self.model.image_encoder.parameters(), lr=self.config.lr, weight_decay=0
        )

    def create_trainer(self):
        loss_fn = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
        self.trainer = Trainer(
            self.model, self.optimizer, loss_fn, self.device,
            self.train_loader, self.val_loader, self.wandb_run,
        )

    def configure_logging(self):
        if self.config.wandb:
            import wandb

            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config={
                    "epochs": self.config.epochs,
                    "batch_size": self.config.batch_size,
                    "rank": self.config.rank,
                    "lr": self.config.lr,
                    "hf_dataset": self.config.hf_dataset,
                    "train_samples": len(self.train_loader.dataset),
                },
            )

    def run_training_loop(self):
        config = self.config
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[4/4] Training for {config.epochs} epochs...")

        # Log step-0 val images (before any training)
        self.trainer.log_val_predictions()

        best_val_loss = float("inf")

        for epoch in range(config.epochs):
            print(f"\nEpoch {epoch + 1}/{config.epochs}")

            train_losses = self.trainer.train_epoch()
            avg_train = mean(train_losses)
            print(f"  Train loss: {avg_train:.4f}")

            log = {"epoch": epoch + 1, "train/loss": avg_train}

            if self.val_loader is not None:
                val_losses = self.trainer.validate_epoch()
                avg_val = mean(val_losses)
                print(f"  Val loss:   {avg_val:.4f}")
                log["val/loss"] = avg_val

                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    best_path = output_dir / f"lora_rank{config.rank}_best.safetensors"
                    self.sam_lora.save_lora_parameters(str(best_path))
                    print(f"  Saved best model -> {best_path}")

            self.trainer.log_val_predictions(log=log)

            if self.wandb_run:
                self.wandb_run.log(log)

        # Save final weights
        final_path = output_dir / f"lora_rank{config.rank}.safetensors"
        self.sam_lora.save_lora_parameters(str(final_path))
        print(f"\nSaved final LoRA weights -> {final_path}")

    def finish(self):
        if self.wandb_run:
            self.wandb_run.finish()
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)


def main():
    config = TrainingConfig.parse_args()
    print("=" * 70)
    print("SAM LoRA Fine-tuning")
    print("=" * 70)
    session = TrainingSession(config)
    session.start()


if __name__ == "__main__":
    main()
