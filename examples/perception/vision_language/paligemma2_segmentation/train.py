#!/usr/bin/env python3
"""
Finetune PaliGemma2 on instance segmentation (COCO masks -> codebook indices).
Derived from the Roboflow Colab. VRAM: notebook calls for an A100 40GB GPU.
"""
import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import supervision as sv

# Force TensorFlow to use CPU only for dataset conversion
# TensorFlow 2.18 doesn't support Blackwell GPUs (compute capability 12.0)
# Must be set before importing tensorflow
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from torchmetrics.detection import MeanAveragePrecision

# Vendored VQ-VAE encoding functions (originally from big_vision)
from vqvae_encode import encode_to_codebook_indices, get_checkpoint


# PaliGemma uses 0-1024 coordinate system for location tokens
PALIGEMMA_LOCATION_RANGE = 1024


def convert_hf_dataset(hf_dataset, checkpoint, category_names: List[str]) -> List[Dict]:
    """Convert HuggingFace dataset with segmentation polygons to PaliGemma format.

    Args:
        hf_dataset: HuggingFace dataset split with 'image' and 'objects' fields
        checkpoint: VAE checkpoint for encoding masks
        category_names: List of category names in order

    Returns:
        List of dicts with 'image', 'prefix', 'suffix' for training
    """
    from pycocotools import mask as mask_utils

    seg_tokens = tf.constant(["<seg%03d>" % i for i in range(128)])
    loc_tokens = tf.constant(["<loc%04d>" % i for i in range(1024)])

    labels = []
    prefix = "segment " + " ; ".join(category_names)

    for idx, example in enumerate(hf_dataset):
        image = example["image"]
        if isinstance(image, dict):  # Some datasets store as dict
            image = image["bytes"]
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(image))

        w, h = image.size
        image_np = np.array(image)

        objects = example["objects"]
        bboxes = objects["bbox"]
        segmentations = objects["segmentation"]
        categories = objects["category"]

        suffix_components = []

        for bbox, segmentation, category in zip(bboxes, segmentations, categories):
            # bbox is [x_min, y_min, x_max, y_max]
            x1, y1, x2, y2 = bbox

            # Convert polygon segmentation to binary mask
            # segmentation is a list of polygons, each polygon is a list of coordinates
            # For HuggingFace datasets, it's typically [[x1, y1, x2, y2, ...]]
            if isinstance(segmentation, list) and len(segmentation) > 0:
                # If it's a nested list, take the first polygon
                poly = segmentation[0] if isinstance(segmentation[0], list) else segmentation
            else:
                poly = segmentation

            rle = mask_utils.frPyObjects([poly], h, w)
            binary_mask = mask_utils.decode(rle)[..., 0].astype(np.uint8)

            # Crop mask to bbox
            y1_int, y2_int = int(y1), int(y2)
            x1_int, x2_int = int(x1), int(x2)

            # Ensure bbox is within image bounds
            y1_int = max(0, min(y1_int, h - 1))
            y2_int = max(0, min(y2_int, h))
            x1_int = max(0, min(x1_int, w - 1))
            x2_int = max(0, min(x2_int, w))

            if y2_int <= y1_int or x2_int <= x1_int:
                continue  # Skip invalid boxes

            mask_crop = binary_mask[y1_int:y2_int, x1_int:x2_int]

            # Resize to 64x64 for encoding
            mask_tf = tf.convert_to_tensor(mask_crop, dtype=tf.uint8)
            mask_resized = tf.image.resize(
                mask_tf[None, :, :, None],
                [64, 64],
                method="bilinear",
                antialias=True,
            )

            # Encode mask to codebook indices
            mask_indices = encode_to_codebook_indices(checkpoint, mask_resized)[0]
            mask_string = tf.strings.reduce_join(tf.gather(seg_tokens, mask_indices))

            # Normalize bbox coordinates to 0-1
            bbox_norm = np.array([y1 / h, x1 / w, y2 / h, x2 / w])
            binned_loc = tf.cast(tf.round(bbox_norm * 1023), tf.int32)
            binned_loc = tf.clip_by_value(binned_loc, 0, 1023)
            loc_string = tf.strings.reduce_join(tf.gather(loc_tokens, binned_loc))

            suffix = tf.strings.join([loc_string, mask_string])
            suffix = f"{suffix.numpy().decode('utf-8')} {category_names[category]}"
            suffix_components.append(suffix)

        if suffix_components:  # Only add if there are valid annotations
            suffix = " ; ".join(suffix_components)
            labels.append({"image_pil": image, "prefix": prefix, "suffix": suffix})

    return labels


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
    def __init__(self, jsonl_file: str = None, image_dir: str = None, hf_dataset=None, entries: List[dict] = None):
        """Dataset that works with JSONL files, HuggingFace datasets, or pre-converted entries.

        Args:
            jsonl_file: Path to JSONL file (for COCO datasets)
            image_dir: Directory containing images (for COCO datasets)
            hf_dataset: HuggingFace dataset split (lazy-loaded)
            entries: Pre-converted list of entries with 'image', 'prefix', 'suffix'
        """
        self.image_dir = image_dir
        self.hf_dataset = hf_dataset

        if entries is not None:
            self.entries = entries
        elif jsonl_file is not None:
            self.jsonl_file = jsonl_file
            self.entries = self._load_entries()
        elif hf_dataset is not None:
            self.entries = None  # Lazy loading from HF dataset
        else:
            raise ValueError("Must provide either jsonl_file, hf_dataset, or entries")

    def _load_entries(self) -> List[dict]:
        entries = []
        with open(self.jsonl_file, "r") as f:
            for line in f:
                entries.append(json.loads(line))
        return entries

    def __len__(self) -> int:
        if self.entries is not None:
            return len(self.entries)
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, dict]:
        if self.hf_dataset is not None:
            # Lazy load from HuggingFace dataset
            example = self.hf_dataset[idx]
            image = example["image"]
            entry = example  # The conversion happens in collator
            return image, entry
        else:
            # Load from pre-converted entries
            entry = self.entries[idx]
            if "image_pil" in entry:
                # HF dataset with pre-converted PIL images
                image = entry["image_pil"]
            else:
                # JSONL with file paths
                image_path = Path(self.image_dir) / entry["image"]
                image = Image.open(image_path)
            return image, entry


def augment_suffix(suffix: str) -> str:
    parts = suffix.split(" ; ")
    import random

    random.shuffle(parts)
    return " ; ".join(parts)


def parse_paligemma_segmentation(text: str) -> List[Dict[str, Any]]:
    """Parse PaliGemma segmentation output into list of detections.

    Format: '<loc{y1}><loc{x1}><loc{y2}><loc{x2}><seg000><seg001>... class_label ; ...'
    Returns: List of dicts with 'bbox' [x1, y1, x2, y2], 'seg_indices', and 'label'
    """
    detections = []
    # Match pattern: <loc###><loc###><loc###><loc###><seg###>... label
    pattern = r'<loc(\d+)><loc(\d+)><loc(\d+)><loc(\d+)>((?:<seg\d{3}>)+)\s+(\S+)'

    for match in re.finditer(pattern, text):
        y1, x1, y2, x2, seg_tokens_str, label = match.groups()

        # Extract segmentation token indices
        seg_indices = [int(m.group(1)) for m in re.finditer(r'<seg(\d{3})>', seg_tokens_str)]

        # Convert from PaliGemma's 0-1024 range to 0-1 normalized coordinates
        bbox = [
            int(x1) / PALIGEMMA_LOCATION_RANGE,
            int(y1) / PALIGEMMA_LOCATION_RANGE,
            int(x2) / PALIGEMMA_LOCATION_RANGE,
            int(y2) / PALIGEMMA_LOCATION_RANGE,
        ]
        detections.append({
            "bbox": bbox,
            "seg_indices": seg_indices,
            "label": label
        })

    return detections


def evaluate_on_dataset(
    model,
    processor,
    dataset: JSONLDataset,
    checkpoint,
    decode_fn,
    device: torch.device,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """Evaluate model on a dataset and compute mAP for masks."""
    from tqdm import tqdm

    model.eval()

    # Collect predictions and targets
    preds = []
    targets = []

    # Build label mapping
    all_labels = set()
    for _, entry in dataset:
        gt_dets = parse_paligemma_segmentation(entry["suffix"])
        for det in gt_dets:
            all_labels.add(det["label"])

    label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}

    with torch.no_grad():
        for image, entry in tqdm(dataset, desc="Evaluating"):
            # Prepare input
            prefix = "<image>" + entry["prefix"]
            inputs = processor(
                text=prefix,
                images=image,
                return_tensors="pt",
            ).to(device)

            # Generate predictions
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

            generated_text = processor.decode(outputs[0], skip_special_tokens=True)
            pred_dets = parse_paligemma_segmentation(generated_text)
            gt_dets = parse_paligemma_segmentation(entry["suffix"])

            # Convert to torchmetrics format
            img_w, img_h = image.size

            if pred_dets:
                pred_boxes = []
                pred_masks = []
                pred_labels = []
                pred_scores = []

                for det in pred_dets:
                    # Convert bbox to absolute coordinates
                    bbox = det["bbox"]
                    x1, y1, x2, y2 = bbox
                    x1_abs, x2_abs = int(x1 * img_w), int(x2 * img_w)
                    y1_abs, y2_abs = int(y1 * img_h), int(y2 * img_h)

                    # Decode segmentation indices to mask (64x64)
                    seg_indices = tf.constant(det["seg_indices"], dtype=tf.int32)
                    mask_64 = decode_fn(checkpoint, seg_indices[None, :])[0, :, :, 0]

                    # Resize mask to bbox size
                    bbox_h = max(1, y2_abs - y1_abs)
                    bbox_w = max(1, x2_abs - x1_abs)
                    mask_resized = tf.image.resize(
                        mask_64[None, :, :, None],
                        [bbox_h, bbox_w],
                        method="bilinear",
                    )[0, :, :, 0]

                    # Create full-size mask
                    full_mask = np.zeros((img_h, img_w), dtype=np.float32)
                    y1_clip = max(0, y1_abs)
                    y2_clip = min(img_h, y2_abs)
                    x1_clip = max(0, x1_abs)
                    x2_clip = min(img_w, x2_abs)

                    mask_np = mask_resized.numpy()
                    y_offset = y1_clip - y1_abs
                    x_offset = x1_clip - x1_abs
                    mask_slice = mask_np[
                        y_offset:y_offset + (y2_clip - y1_clip),
                        x_offset:x_offset + (x2_clip - x1_clip)
                    ]
                    full_mask[y1_clip:y2_clip, x1_clip:x2_clip] = mask_slice

                    pred_boxes.append([x1_abs, y1_abs, x2_abs, y2_abs])
                    pred_masks.append(full_mask > 0.5)
                    pred_labels.append(label_to_idx.get(det["label"], 0))
                    pred_scores.append(1.0)

                preds.append({
                    "boxes": torch.tensor(pred_boxes, dtype=torch.float32),
                    "masks": torch.tensor(np.stack(pred_masks), dtype=torch.bool),
                    "scores": torch.tensor(pred_scores, dtype=torch.float32),
                    "labels": torch.tensor(pred_labels, dtype=torch.int64),
                })
            else:
                preds.append({
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "masks": torch.zeros((0, img_h, img_w), dtype=torch.bool),
                    "scores": torch.zeros(0, dtype=torch.float32),
                    "labels": torch.zeros(0, dtype=torch.int64),
                })

            if gt_dets:
                gt_boxes = []
                gt_masks = []
                gt_labels = []

                for det in gt_dets:
                    # Same process for ground truth
                    bbox = det["bbox"]
                    x1, y1, x2, y2 = bbox
                    x1_abs, x2_abs = int(x1 * img_w), int(x2 * img_w)
                    y1_abs, y2_abs = int(y1 * img_h), int(y2 * img_h)

                    seg_indices = tf.constant(det["seg_indices"], dtype=tf.int32)
                    mask_64 = decode_fn(checkpoint, seg_indices[None, :])[0, :, :, 0]

                    bbox_h = max(1, y2_abs - y1_abs)
                    bbox_w = max(1, x2_abs - x1_abs)
                    mask_resized = tf.image.resize(
                        mask_64[None, :, :, None],
                        [bbox_h, bbox_w],
                        method="bilinear",
                    )[0, :, :, 0]

                    full_mask = np.zeros((img_h, img_w), dtype=np.float32)
                    y1_clip = max(0, y1_abs)
                    y2_clip = min(img_h, y2_abs)
                    x1_clip = max(0, x1_abs)
                    x2_clip = min(img_w, x2_abs)

                    mask_np = mask_resized.numpy()
                    y_offset = y1_clip - y1_abs
                    x_offset = x1_clip - x1_abs
                    mask_slice = mask_np[
                        y_offset:y_offset + (y2_clip - y1_clip),
                        x_offset:x_offset + (x2_clip - x1_clip)
                    ]
                    full_mask[y1_clip:y2_clip, x1_clip:x2_clip] = mask_slice

                    gt_boxes.append([x1_abs, y1_abs, x2_abs, y2_abs])
                    gt_masks.append(full_mask > 0.5)
                    gt_labels.append(label_to_idx.get(det["label"], 0))

                targets.append({
                    "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
                    "masks": torch.tensor(np.stack(gt_masks), dtype=torch.bool),
                    "labels": torch.tensor(gt_labels, dtype=torch.int64),
                })
            else:
                targets.append({
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "masks": torch.zeros((0, img_h, img_w), dtype=torch.bool),
                    "labels": torch.zeros(0, dtype=torch.int64),
                })

    # Compute mAP for masks
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="segm")
    metric.update(preds, targets)
    results = metric.compute()

    return results


class mAPEvaluationCallback(TrainerCallback):
    """Custom callback to evaluate mAP for segmentation after each epoch."""

    def __init__(self, model, processor, val_dataset, checkpoint, decode_fn, device, max_new_tokens=512):
        self.model = model
        self.processor = processor
        self.val_dataset = val_dataset
        self.checkpoint = checkpoint
        self.decode_fn = decode_fn
        self.device = device
        self.max_new_tokens = max_new_tokens

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        print("\n" + "=" * 80)
        print(f"EVALUATING ON VALIDATION SET - Epoch {int(state.epoch)}")
        print("=" * 80)

        results = evaluate_on_dataset(
            self.model,
            self.processor,
            self.val_dataset,
            self.checkpoint,
            self.decode_fn,
            self.device,
            max_new_tokens=self.max_new_tokens,
        )

        print("\n" + "=" * 80)
        print(f"VALIDATION RESULTS - Epoch {int(state.epoch)}")
        print("=" * 80)
        print(f"mAP@50:95: {results['map']:.4f}")
        print(f"mAP@50:    {results['map_50']:.4f}")
        print(f"mAP@75:    {results['map_75']:.4f}")
        print(f"mAP (small):  {results['map_small']:.4f}")
        print(f"mAP (medium): {results['map_medium']:.4f}")
        print(f"mAP (large):  {results['map_large']:.4f}")
        print("=" * 80 + "\n")

        # Log to tensorboard if available
        if state.is_world_process_zero:
            try:
                import tensorboard
                state.log_history.append({
                    "epoch": state.epoch,
                    "eval_map": results['map'].item(),
                    "eval_map_50": results['map_50'].item(),
                    "eval_map_75": results['map_75'].item(),
                })
            except:
                pass

        return control


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
    parser.add_argument("--dataset-dir", help="COCO dataset root containing train/valid/test with _annotations.coco.json")
    parser.add_argument("--hf-dataset", help="HuggingFace dataset name (e.g., keremberke/pcb-defect-segmentation)")
    parser.add_argument("--hf-dataset-config", default="full", help="HuggingFace dataset config name")
    parser.add_argument("--output-dir", default="paligemma2_instance_segmentation", help="Trainer output dir")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation after training")
    parser.add_argument("--eval-max-tokens", type=int, default=512, help="Max tokens for evaluation generation")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.dataset_dir and not args.hf_dataset:
        raise ValueError("Must provide either --dataset-dir or --hf-dataset")

    # Load VQ-VAE checkpoint for mask encoding
    checkpoint = get_checkpoint(model="oi")

    if args.hf_dataset:
        # Load from HuggingFace
        print(f"Loading HuggingFace dataset: {args.hf_dataset} (config: {args.hf_dataset_config})")
        from datasets import load_dataset

        dataset = load_dataset(args.hf_dataset, name=args.hf_dataset_config)

        # Get category names from dataset features
        category_feature = dataset["train"].features["objects"].feature["category"]
        category_names = category_feature.names
        print(f"Found {len(category_names)} categories: {category_names}")

        # Convert HuggingFace dataset to PaliGemma format
        print("Converting train split...")
        train_labels = convert_hf_dataset(dataset["train"], checkpoint, category_names)
        print(f"Converted {len(train_labels)} train samples")

        print("Converting validation split...")
        val_labels = convert_hf_dataset(dataset["validation"], checkpoint, category_names)
        print(f"Converted {len(val_labels)} validation samples")

        # Create datasets with pre-converted entries (don't pass hf_dataset to avoid lazy loading)
        train_ds = JSONLDataset(entries=train_labels)
        val_ds = JSONLDataset(entries=val_labels)

    else:
        # Load from COCO format
        print(f"Loading COCO dataset from: {args.dataset_dir}")
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

    # NOTE: Per-epoch evaluation disabled for now - use evaluate.py after training
    # TODO: Implement evaluation using JAX-based decode from original notebook
    # map_callback = mAPEvaluationCallback(...)

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
        args=training_args,
        # callbacks=[map_callback],
    )

    trainer.train()

    # NOTE: Post-training evaluation disabled for now - use evaluate.py script
    # TODO: Implement evaluation using JAX-based decode from original notebook
    if False and not args.skip_eval:
        print("\n" + "=" * 80)
        print("EVALUATING ON VALIDATION SET")
        print("=" * 80)

        # results = evaluate_on_dataset(...)

        print("\n" + "=" * 80)
        print("VALIDATION RESULTS")
        print("=" * 80)
        print(f"mAP@50:95: {results['map']:.4f}")
        print(f"mAP@50:    {results['map_50']:.4f}")
        print(f"mAP@75:    {results['map_75']:.4f}")
        print(f"mAP (small):  {results['map_small']:.4f}")
        print(f"mAP (medium): {results['map_medium']:.4f}")
        print(f"mAP (large):  {results['map_large']:.4f}")
        print("=" * 80)


if __name__ == "__main__":
    main()
