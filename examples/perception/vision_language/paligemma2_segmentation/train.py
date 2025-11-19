#!/usr/bin/env python3
"""
Finetune PaliGemma2 on instance segmentation (COCO masks -> codebook indices).
Derived from the Roboflow Colab. VRAM: notebook calls for an A100 40GB GPU.
"""
import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import supervision as sv
import tensorflow as tf
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Trainer,
    TrainingArguments,
)


def ensure_big_vision(repo_dir: Path) -> None:
    if (repo_dir / ".git").exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    os.system(f"git clone --quiet --branch=main --depth=1 https://github.com/google-research/big_vision {repo_dir}")


def load_big_vision(repo_dir: Path):
    import sys

    if str(repo_dir) not in sys.path:
        sys.path.append(str(repo_dir))
    from big_vision.pp.proj.paligemma.segmentation import encode_to_codebook_indices, get_checkpoint

    return encode_to_codebook_indices, get_checkpoint


def convert_coco_dataset(dataset_path: str, checkpoint) -> List[Dict]:
    ds = sv.DetectionDataset.from_coco(
        images_directory_path=f"{dataset_path}",
        annotations_path=f"{dataset_path}/_annotations.coco.json",
        force_masks=True,
    )

    seg_tokens = tf.constant(["<seg%03d>" % i for i in range(128)])
    loc_tokens = tf.constant(["<loc%04d>" % i for i in range(1024)])

    labels = []
    prefix = "segment " + " ; ".join(ds.classes)

    for image_path, image, annotations in ds:
        h, w, _ = image.shape
        image_name = os.path.basename(image_path)
        suffix_components = []

        if annotations.xyxy is None or annotations.mask is None or annotations.class_id is None:
            continue

        for xyxy, mask, class_id in zip(annotations.xyxy, annotations.mask, annotations.class_id):
            y1 = tf.cast(tf.round(xyxy[1]), tf.int32)
            x1 = tf.cast(tf.round(xyxy[0]), tf.int32)
            y2 = tf.cast(tf.round(xyxy[3]), tf.int32)
            x2 = tf.cast(tf.round(xyxy[2]), tf.int32)

            mask = tf.convert_to_tensor(mask.astype(np.uint8), dtype=tf.uint8)
            mask = tf.image.resize(
                mask[None, y1:y2, x1:x2, None],
                [64, 64],
                method="bilinear",
                antialias=True,
            )

            mask_indices = encode_to_codebook_indices(checkpoint, mask)[0]
            mask_string = tf.strings.reduce_join(tf.gather(seg_tokens, mask_indices))

            bbox = xyxy[[1, 0, 3, 2]] / np.array([h, w, h, w])
            binned_loc = tf.cast(tf.round(bbox * 1023), tf.int32)
            binned_loc = tf.clip_by_value(binned_loc, 0, 1023)
            loc_string = tf.strings.reduce_join(tf.gather(loc_tokens, binned_loc))
            suffix = tf.strings.join([loc_string, mask_string])
            suffix = f"{suffix.numpy().decode('utf-8')} {ds.classes[class_id]}"
            suffix_components.append(suffix)

        suffix = " ; ".join(suffix_components)
        labels.append({"image": image_name, "prefix": prefix, "suffix": suffix})

    return labels


def copy_images(src_dir: str, dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                shutil.copy(os.path.join(root, file), os.path.join(dest_dir, file))


def save_jsonl(data: List[Dict], path: str) -> None:
    with open(path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


class JSONLDataset(Dataset):
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
    import random

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
    parser = argparse.ArgumentParser(description="Finetune PaliGemma2 for instance segmentation")
    parser.add_argument("--model-id", default="google/paligemma2-3b-pt-224", help="Base model id")
    parser.add_argument("--dataset-dir", required=True, help="COCO dataset root containing train/valid/test with _annotations.coco.json")
    parser.add_argument("--output-dir", default="paligemma2_instance_segmentation", help="Trainer output dir")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--big-vision-dir", default="/workspace/big_vision_repo", help="Path to clone big_vision for codebook utils")
    return parser.parse_args()


def main():
    args = parse_args()

    repo_dir = Path(args.big_vision_dir)
    ensure_big_vision(repo_dir)
    global encode_to_codebook_indices
    encode_to_codebook_indices, get_checkpoint = load_big_vision(repo_dir)
    checkpoint = get_checkpoint(model="oi")

    converted_root = Path(f"{args.dataset_dir}-converted")
    converted_root.mkdir(exist_ok=True)

    for split in ["train", "valid", "test"]:
        src_split = Path(args.dataset_dir) / split
        dest_split = converted_root / split
        copy_images(str(src_split), str(dest_split))
        labels = convert_coco_dataset(str(src_split), checkpoint)
        save_jsonl(labels, dest_split / "annotations.jsonl")

    train_ds = JSONLDataset(str(converted_root / "train" / "annotations.jsonl"), str(Path(args.dataset_dir) / "train"))
    val_ds = JSONLDataset(str(converted_root / "valid" / "annotations.jsonl"), str(Path(args.dataset_dir) / "valid"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    processor = PaliGemmaProcessor.from_pretrained(args.model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=dtype).to(device)

    for param in model.vision_tower.parameters():
        param.requires_grad = False
    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False

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
