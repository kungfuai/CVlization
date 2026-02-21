#!/usr/bin/env python3
"""
SMT Fine-tuning — Domain adaptation of Sheet Music Transformer.

Fine-tunes a pretrained SMT checkpoint on the GrandStaff system-level dataset.
Uses PyTorch Lightning for training and Weights & Biases for logging.

Default config (config.yaml) is set for a quick smoke test (<10 min).
Increase max_steps and remove max_samples_train limit for full training.

Model: antoniorv6/smt-grandstaff (pretrained, system-level piano)
Dataset: antoniorv6/grandstaff (HuggingFace, ~14k system images)
Repository: https://github.com/antoniorv6/SMT
License: MIT
"""

import os
import sys
import logging
import warnings

# Suppress verbose logging by default
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("WANDB_SILENT", "true")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
for _logger_name in ["transformers", "torch", "lightning"]:
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

import argparse
import random
import re
import yaml
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


class OMRDataset(Dataset):
    """
    System-level OMR dataset backed by a HuggingFace dataset split.

    Tokenizes transcriptions using the pretrained model's vocabulary (w2i).
    Images are converted to grayscale tensors. Unknown tokens are dropped.
    """

    def __init__(self, hf_dataset, w2i: dict, augment: bool = False, max_samples: int = None):
        self.data = hf_dataset
        if max_samples is not None:
            self.data = self.data.select(range(min(max_samples, len(self.data))))
        self.w2i = w2i
        # Build reverse map with string keys (JSON serialization converts int keys to str)
        self.i2w = {str(v): k for k, v in w2i.items()}
        self.padding_token = w2i.get("<pad>", 0)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        from data_augmentation.data_augmentation import augment as aug_fn, convert_img_to_tensor

        sample = self.data[idx]
        img = np.array(sample["image"])

        # Limit width for memory efficiency (matches upstream GrandStaffSingleSystem)
        if img.shape[1] > 3056:
            img = cv2.resize(img, (3056, max(img.shape[0], 256)))

        x = aug_fn(img) if self.augment else convert_img_to_tensor(img)

        # Tokenize transcription to bekern token sequence
        gt = sample["transcription"].strip("\n ")
        gt = re.sub(r"(?<=\=)\d+", "", gt)
        gt = gt.replace(" ", " <s> ").replace("\t", " <t> ").replace("\n", " <b> ")
        tokens = ["<bos>"] + gt.split(" ") + ["<eos>"]

        # Filter tokens not in vocabulary (safety for domain-adapted models)
        tokens = [t for t in tokens if t in self.w2i]
        if not tokens or tokens[0] != "<bos>":
            tokens.insert(0, "<bos>")
        if tokens[-1] != "<eos>":
            tokens.append("<eos>")

        y = torch.tensor([self.w2i[t] for t in tokens], dtype=torch.long)
        return x, y

    @staticmethod
    def collate_fn(batch):
        """Pad images and label sequences to batch-max dimensions."""
        images, labels = zip(*batch)

        # Pad images to max H and W in batch (white padding = 1.0)
        max_h = max(max(img.shape[1] for img in images), 256)
        max_w = max(max(img.shape[2] for img in images), 128)
        X = torch.ones(len(images), 1, max_h, max_w, dtype=torch.float32)
        for i, img in enumerate(images):
            _, h, w = img.shape
            X[i, :, :h, :w] = img

        # Pad label sequences; decoder input = y[:-1], targets = y[1:]
        max_len = max(len(y) for y in labels)
        dec_in = torch.zeros(len(labels), max_len - 1, dtype=torch.long)
        y_out = torch.zeros(len(labels), max_len - 1, dtype=torch.long)
        for i, y in enumerate(labels):
            L = len(y)
            dec_in[i, :L - 1] = y[:-1]
            y_out[i, :L - 1] = y[1:]

        return X, dec_in, y_out


class SMTFinetuner(L.LightningModule):
    """LightningModule wrapping a pretrained SMTModelForCausalLM for fine-tuning."""

    def __init__(self, model_id: str, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        from smt_model import SMTModelForCausalLM
        print(f"Loading pretrained model: {model_id}")
        self.model = SMTModelForCausalLM.from_pretrained(model_id)
        self.lr = lr
        self._preds: list[str] = []
        self._gts: list[str] = []

    def training_step(self, batch, batch_idx):
        x, dec_in, y = batch
        out = self.model(encoder_input=x, decoder_input=dec_in, labels=y)
        self.log("train_loss", out.loss, prog_bar=True, on_step=True, on_epoch=True)
        return out.loss

    def validation_step(self, batch, batch_idx):
        x, _, y = batch
        # predict() with convert_to_str=True handles string-keyed i2w from JSON
        preds, _ = self.model.predict(x, convert_to_str=True)

        # Decode ground truth using the same i2w (string-keyed after JSON round-trip)
        gt = [
            self.model.i2w.get(str(t.item()), "<unk>")
            for t in y.squeeze(0)[:-1]  # skip <eos>
        ]

        def decode(tokens):
            return (
                " ".join(tokens)
                .replace("<t>", "\t")
                .replace("<b>", "\n")
                .replace("<s>", " ")
            )

        self._preds.append(decode(preds))
        self._gts.append(decode(gt))

    def on_validation_epoch_end(self):
        if not self._preds:
            return

        # Symbol Error Rate (SER): fraction of sequences with any error
        def tokenize(s):
            return s.replace("\n", " <b> ").replace("\t", " <t> ").split()

        from utils import levenshtein
        total_ed = sum(
            levenshtein(tokenize(p), tokenize(g))
            for p, g in zip(self._preds, self._gts)
        )
        total_len = sum(len(tokenize(g)) for g in self._gts)
        ser = 100.0 * total_ed / max(total_len, 1)

        self.log("val_SER", ser, prog_bar=True)

        # Print a random example for visual inspection
        i = random.randint(0, len(self._preds) - 1)
        print(f"\n[pred] {self._preds[i][:120]}")
        print(f"[gt]   {self._gts[i][:120]}")

        self._preds.clear()
        self._gts.clear()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def main():
    parser = argparse.ArgumentParser(
        description="SMT fine-tuning on GrandStaff dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smoke test (10 steps, 50 samples)
  python train.py

  # Custom config
  python train.py --config my_config.yaml

  # Full training (edit config.yaml: max_steps: 10000, max_samples_train: null)
  python train.py --config config.yaml
        """,
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)
        for _n in ["transformers", "torch", "lightning"]:
            logging.getLogger(_n).setLevel(logging.INFO)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_id = cfg["model"]["name"]
    data_path = cfg["dataset"]["path"]
    max_samples_train = cfg["dataset"].get("max_samples_train")  # None = use all
    max_samples_val = cfg["dataset"].get("max_samples_val", 20)
    lr = float(cfg["training"]["learning_rate"])
    max_steps = int(cfg["training"]["max_steps"])
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["training"].get("num_workers", 0))
    output_dir = cfg["training"]["output_dir"]
    val_check_interval = int(cfg["training"].get("val_check_interval", max(1, max_steps // 2)))
    wandb_offline = cfg["training"].get("wandb_offline", True)

    if wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset
    import datasets as hf_datasets
    print(f"Loading dataset: {data_path}")
    train_ds = hf_datasets.load_dataset(data_path, split="train", trust_remote_code=False)
    val_ds = hf_datasets.load_dataset(data_path, split="val", trust_remote_code=False)
    print(f"  Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Load model (which carries its own vocabulary in config)
    finetuner = SMTFinetuner(model_id=model_id, lr=lr)
    w2i = finetuner.model.w2i
    print(f"  Vocabulary size: {len(w2i)} tokens")

    train_set = OMRDataset(train_ds, w2i, augment=True, max_samples=max_samples_train)
    val_set = OMRDataset(val_ds, w2i, augment=False, max_samples=max_samples_val)
    print(f"  Using {len(train_set)} train / {len(val_set)} val samples")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=OMRDataset.collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=OMRDataset.collate_fn
    )

    # Callbacks and logger
    ckpt_cb = ModelCheckpoint(
        dirpath=output_dir,
        filename="smt-finetune-{step:04d}-{val_SER:.2f}",
        monitor="val_SER",
        mode="min",
        save_top_k=1,
    )
    wandb_logger = WandbLogger(
        project="SMT_CVlization",
        name="smt-finetune",
    )

    trainer = Trainer(
        max_steps=max_steps,
        val_check_interval=val_check_interval,
        logger=wandb_logger,
        callbacks=[ckpt_cb],
        precision="16-mixed",
        accelerator="auto",
        log_every_n_steps=1,
    )

    print(f"\nStarting fine-tuning for {max_steps} steps...")
    print(f"  Model: {model_id}")
    print(f"  LR: {lr}  |  Batch size: {batch_size}  |  val every {val_check_interval} steps")
    print(f"  Output dir: {output_dir}\n")

    trainer.fit(finetuner, train_loader, val_loader)

    best = ckpt_cb.best_model_path
    print(f"\nBest checkpoint: {best or output_dir}")
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
