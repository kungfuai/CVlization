"""DatasetBuilder -- resolves dataset source, detects format, builds dataloaders."""

import sys
from pathlib import Path

from torch.utils.data import DataLoader

from src.data.dataset import (
    CocoSamDataset,
    ImageMaskDataset,
    collate_fn,
    download_hf_dataset,
    download_ring_dataset,
)


class DatasetBuilder:
    """Resolve dataset source, detect format, and build train/val dataloaders."""

    def __init__(self, config, sam_model):
        self.config = config
        self.sam_model = sam_model

    def build(self):
        """Download if needed, detect format, return (train_loader, val_loader).

        Returns:
            tuple: (train_loader, val_loader | None)
        """
        config = self.config

        dataset_format = None
        dataset_dir = config.dataset_dir

        if config.hf_dataset:
            print(f"  Downloading dataset: {config.hf_dataset}")
            dataset_dir = download_hf_dataset(config.hf_dataset)
            dataset_format = "coco"
        elif dataset_dir is None:
            print("  No dataset specified -- downloading default ring dataset...")
            dataset_dir = download_ring_dataset()
        else:
            print(f"  Using dataset: {dataset_dir}")

        dataset_path = Path(dataset_dir)

        # Auto-detect format if not already set
        if dataset_format is None:
            train_json = dataset_path / "train" / "_annotations.coco.json"
            if train_json.exists():
                dataset_format = "coco"
            elif (dataset_path / "train" / "images").is_dir() and (
                dataset_path / "train" / "masks"
            ).is_dir():
                dataset_format = "image_mask"
            else:
                print(f"ERROR: Cannot detect dataset format in {dataset_path}.")
                print("  Expected either train/_annotations.coco.json (COCO)")
                print("  or train/images/ + train/masks/ (image+mask)")
                sys.exit(1)

        print(f"  Dataset format: {dataset_format}")

        # Build train loader
        if dataset_format == "coco":
            train_json = dataset_path / "train" / "_annotations.coco.json"
            train_images = dataset_path / "train" / "images"
            if not train_json.exists():
                print(f"ERROR: Train annotations not found: {train_json}")
                sys.exit(1)
            train_ds = CocoSamDataset(
                str(train_json), str(train_images), self.sam_model
            )
        else:
            train_ds = ImageMaskDataset(
                str(dataset_path / "train" / "images"),
                str(dataset_path / "train" / "masks"),
                self.sam_model,
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        print(f"  Train: {len(train_ds)} samples")

        # Build val loader
        val_loader = None
        if dataset_format == "coco":
            val_json = dataset_path / "valid" / "_annotations.coco.json"
            val_images = dataset_path / "valid" / "images"
            if val_json.exists():
                val_ds = CocoSamDataset(
                    str(val_json), str(val_images), self.sam_model
                )
                val_loader = DataLoader(
                    val_ds,
                    batch_size=config.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                )
                print(f"  Val:   {len(val_ds)} samples")
        else:
            test_images = dataset_path / "test" / "images"
            test_masks = dataset_path / "test" / "masks"
            if test_images.is_dir() and test_masks.is_dir():
                val_ds = ImageMaskDataset(
                    str(test_images), str(test_masks), self.sam_model
                )
                val_loader = DataLoader(
                    val_ds,
                    batch_size=config.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                )
                print(f"  Val:   {len(val_ds)} samples")

        return train_loader, val_loader
