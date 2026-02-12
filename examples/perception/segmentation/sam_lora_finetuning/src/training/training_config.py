"""Training configuration for SAM LoRA fine-tuning."""

from typing import Optional

from pydantic import Field

from src.config.base_config import BaseConfig


class TrainingConfig(BaseConfig):
    """All CLI fields for SAM LoRA fine-tuning, mapped 1:1 from the old argparse."""

    hf_dataset: Optional[str] = Field(
        default=None, description="HuggingFace dataset to download"
    )
    dataset_dir: Optional[str] = Field(
        default=None, description="Path to COCO dataset root"
    )
    output_dir: str = Field(default="outputs", description="Output directory")
    epochs: int = Field(default=50, description="Number of training epochs")
    batch_size: int = Field(default=1, description="Batch size")
    rank: int = Field(default=512, description="LoRA rank")
    lr: float = Field(default=1e-4, description="Learning rate")
    checkpoint: Optional[str] = Field(
        default=None, description="SAM ViT-B checkpoint path"
    )
    wandb: bool = Field(default=False, description="Enable wandb logging")
    wandb_project: str = Field(
        default="sam-lora-finetuning", description="Wandb project name"
    )
    wandb_run_name: Optional[str] = Field(
        default=None, description="Wandb run name"
    )
