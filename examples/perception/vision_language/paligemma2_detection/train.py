#!/usr/bin/env python3
"""
Finetune PaliGemma2 on object detection (Roboflow paligemma JSONL format).

Derived from the Roboflow Colab "how-to-finetune-paligemma2-on-detection-dataset".
VRAM: the notebook requires an A100 40GB GPU to train.
"""
import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Trainer,
    TrainingArguments,
)


class JSONLDataset(Dataset):
    """Minimal loader for the Roboflow paligemma JSONL format."""

    def __init__(self, jsonl_file: str, image_dir: str):
        self.jsonl_file = jsonl_file
        self.image_dir = image_dir
        self.entries = self._load_entries()

    def _load_entries(self) -> List[dict]:
        entries = []
        with open(self.jsonl_file, "r") as f:
            for line in f:
                entries.append(json.loads(line))
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, dict]:
        entry = self.entries[idx]
        image_path = Path(self.image_dir) / entry["image"]
        image = Image.open(image_path)
        return image, entry


def augment_suffix(suffix: str) -> str:
    parts = suffix.split(" ; ")
    random.shuffle(parts)
    return " ; ".join(parts)


@dataclass
class Collator:
    processor: PaliGemmaProcessor
    device: torch.device
    dtype: torch.dtype

    def __call__(self, batch):
        images, labels = zip(*batch)
        prefixes = ["<image>" + label["prefix"] for label in labels]
        suffixes = [augment_suffix(label["suffix"]) for label in labels]

        inputs = self.processor(
            text=prefixes,
            images=images,
            return_tensors="pt",
            suffix=suffixes,
            padding="longest",
        ).to(self.dtype).to(self.device)
        return inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune PaliGemma2 for object detection")
    parser.add_argument("--model-id", default="google/paligemma2-3b-pt-448", help="Base model id")
    parser.add_argument("--train-jsonl", required=True, help="Train annotations JSONL")
    parser.add_argument("--val-jsonl", required=True, help="Validation annotations JSONL")
    parser.add_argument("--image-dir", required=True, help="Directory containing images referenced by JSONL")
    parser.add_argument("--output-dir", default="paligemma2_object_detection", help="HF trainer output dir")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    processor = PaliGemmaProcessor.from_pretrained(args.model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=dtype).to(device)

    # Freeze vision encoder and projector (per notebook)
    for param in model.vision_tower.parameters():
        param.requires_grad = False
    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False

    train_ds = JSONLDataset(args.train_jsonl, args.image_dir)
    val_ds = JSONLDataset(args.val_jsonl, args.image_dir)

    collate = Collator(processor=processor, device=device, dtype=dtype)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        remove_unused_columns=False,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        adam_beta2=0.999,
        logging_steps=args.logging_steps,
        optim="adamw_hf",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        bf16=(dtype == torch.bfloat16),
        report_to=["tensorboard"],
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
        args=training_args,
    )

    trainer.train()


if __name__ == "__main__":
    main()
