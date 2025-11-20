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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    TrainerCallback,
)
from torchmetrics.detection import MeanAveragePrecision


# PaliGemma uses 0-1024 coordinate system for location tokens
PALIGEMMA_LOCATION_RANGE = 1024


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


def parse_paligemma_detections(text: str) -> List[Dict[str, Any]]:
    """Parse PaliGemma detection output into list of boxes.

    Format: '<loc{y1}><loc{x1}><loc{y2}><loc{x2}> class_label ; ...'
    Returns: List of dicts with 'bbox' [x1, y1, x2, y2] and 'label'
    """
    detections = []
    # Match pattern: <loc###><loc###><loc###><loc###> label
    pattern = r'<loc(\d+)><loc(\d+)><loc(\d+)><loc(\d+)>\s+(\S+)'

    for match in re.finditer(pattern, text):
        y1, x1, y2, x2, label = match.groups()
        # Convert from PaliGemma's 0-1024 range to 0-1 normalized coordinates
        bbox = [
            int(x1) / PALIGEMMA_LOCATION_RANGE,
            int(y1) / PALIGEMMA_LOCATION_RANGE,
            int(x2) / PALIGEMMA_LOCATION_RANGE,
            int(y2) / PALIGEMMA_LOCATION_RANGE,
        ]
        detections.append({"bbox": bbox, "label": label})

    return detections


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


def evaluate_on_dataset(
    model,
    processor,
    dataset: JSONLDataset,
    device: torch.device,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """Evaluate model on a dataset and compute mAP."""
    from tqdm import tqdm

    model.eval()

    # Collect predictions and targets
    preds = []
    targets = []

    # Build label mapping
    all_labels = set()
    for _, entry in dataset:
        gt_dets = parse_paligemma_detections(entry["suffix"])
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
            pred_dets = parse_paligemma_detections(generated_text)
            gt_dets = parse_paligemma_detections(entry["suffix"])

            # Convert to torchmetrics format
            img_w, img_h = image.size

            if pred_dets:
                pred_boxes = torch.tensor([d["bbox"] for d in pred_dets], dtype=torch.float32)
                pred_labels = torch.tensor(
                    [label_to_idx.get(d["label"], 0) for d in pred_dets],
                    dtype=torch.int64
                )
                pred_scores = torch.ones(len(pred_dets), dtype=torch.float32)

                # Convert to absolute coordinates
                pred_boxes_abs = pred_boxes.clone()
                pred_boxes_abs[:, [0, 2]] *= img_w
                pred_boxes_abs[:, [1, 3]] *= img_h

                preds.append({
                    "boxes": pred_boxes_abs,
                    "scores": pred_scores,
                    "labels": pred_labels,
                })
            else:
                preds.append({
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.zeros(0, dtype=torch.float32),
                    "labels": torch.zeros(0, dtype=torch.int64),
                })

            if gt_dets:
                gt_boxes = torch.tensor([d["bbox"] for d in gt_dets], dtype=torch.float32)
                gt_labels = torch.tensor(
                    [label_to_idx.get(d["label"], 0) for d in gt_dets],
                    dtype=torch.int64
                )

                gt_boxes_abs = gt_boxes.clone()
                gt_boxes_abs[:, [0, 2]] *= img_w
                gt_boxes_abs[:, [1, 3]] *= img_h

                targets.append({
                    "boxes": gt_boxes_abs,
                    "labels": gt_labels,
                })
            else:
                targets.append({
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros(0, dtype=torch.int64),
                })

    # Compute mAP
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.update(preds, targets)
    results = metric.compute()

    return results


class mAPEvaluationCallback(TrainerCallback):
    """Custom callback to evaluate mAP after each epoch."""

    def __init__(self, model, processor, val_dataset, device, max_new_tokens=256):
        self.model = model
        self.processor = processor
        self.val_dataset = val_dataset
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
                # Write to the trainer's log history
                state.log_history.append({
                    "epoch": state.epoch,
                    "eval_map": results['map'].item(),
                    "eval_map_50": results['map_50'].item(),
                    "eval_map_75": results['map_75'].item(),
                })
            except:
                pass

        return control


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
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation after training")
    parser.add_argument("--eval-max-tokens", type=int, default=256, help="Max tokens for evaluation generation")
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

    # Create mAP evaluation callback
    map_callback = mAPEvaluationCallback(
        model=model,
        processor=processor,
        val_dataset=val_ds,
        device=device,
        max_new_tokens=args.eval_max_tokens,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
        args=training_args,
        callbacks=[map_callback],
    )

    trainer.train()

    # Evaluate on validation set after training
    if not args.skip_eval:
        print("\n" + "=" * 80)
        print("EVALUATING ON VALIDATION SET")
        print("=" * 80)

        results = evaluate_on_dataset(
            model,
            processor,
            val_ds,
            device,
            max_new_tokens=args.eval_max_tokens,
        )

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
