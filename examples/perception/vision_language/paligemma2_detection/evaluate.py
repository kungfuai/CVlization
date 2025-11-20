#!/usr/bin/env python3
"""
Evaluate PaliGemma2 object detection model using mAP metrics.

Usage:
    python evaluate.py --model-path ./paligemma2_object_detection/checkpoint-100 \
                       --test-jsonl dataset/_annotations.test.jsonl \
                       --image-dir dataset/images
"""
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from torchmetrics.detection import MeanAveragePrecision


# PaliGemma uses 0-1024 coordinate system for location tokens
PALIGEMMA_LOCATION_RANGE = 1024


def parse_paligemma_detections(text: str) -> List[Dict[str, Any]]:
    """Parse PaliGemma detection output into list of boxes.

    Format: '<loc{y1}><loc{x1}><loc{y2}><loc{x2}> class_label ; ...'
    Returns: List of dicts with 'bbox' [x1, y1, x2, y2] and 'label'
    """
    detections = []
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


def load_annotations(jsonl_file: str) -> List[dict]:
    """Load annotations from JSONL file."""
    annotations = []
    with open(jsonl_file, "r") as f:
        for line in f:
            annotations.append(json.loads(line))
    return annotations


def evaluate_model(
    model,
    processor,
    annotations: List[dict],
    image_dir: str,
    device: torch.device,
    max_new_tokens: int = 256,
) -> Dict[str, float]:
    """Evaluate model on dataset and compute mAP.

    Args:
        model: PaliGemma model
        processor: PaliGemma processor
        annotations: List of annotation dicts with 'image', 'prefix', 'suffix'
        image_dir: Directory containing images
        device: Device to run on
        max_new_tokens: Max tokens to generate

    Returns:
        Dict with mAP metrics
    """
    model.eval()

    # Collect all predictions and targets
    preds = []
    targets = []

    # Build label to index mapping
    all_labels = set()
    for ann in annotations:
        gt_dets = parse_paligemma_detections(ann["suffix"])
        for det in gt_dets:
            all_labels.add(det["label"])

    label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
    print(f"Found {len(label_to_idx)} unique classes: {list(label_to_idx.keys())}")

    with torch.no_grad():
        for ann in tqdm(annotations, desc="Evaluating"):
            # Load image
            image_path = Path(image_dir) / ann["image"]
            image = Image.open(image_path)

            # Prepare input
            prefix = "<image>" + ann["prefix"]
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

            # Decode prediction
            generated_text = processor.decode(outputs[0], skip_special_tokens=True)

            # Parse predictions
            pred_dets = parse_paligemma_detections(generated_text)

            # Parse ground truth
            gt_dets = parse_paligemma_detections(ann["suffix"])

            # Convert to torchmetrics format
            if pred_dets:
                pred_boxes = torch.tensor([d["bbox"] for d in pred_dets], dtype=torch.float32)
                pred_labels = torch.tensor(
                    [label_to_idx.get(d["label"], 0) for d in pred_dets],
                    dtype=torch.int64
                )
                pred_scores = torch.ones(len(pred_dets), dtype=torch.float32)  # Assume all confident

                # Convert from [x1, y1, x2, y2] normalized to absolute coords (assuming 448x448)
                img_w, img_h = image.size
                pred_boxes_abs = pred_boxes.clone()
                pred_boxes_abs[:, [0, 2]] *= img_w
                pred_boxes_abs[:, [1, 3]] *= img_h

                preds.append({
                    "boxes": pred_boxes_abs,
                    "scores": pred_scores,
                    "labels": pred_labels,
                })
            else:
                # No predictions - add empty
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

                # Convert to absolute coords
                img_w, img_h = image.size
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate PaliGemma2 object detection")
    parser.add_argument("--model-path", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--test-jsonl", required=True, help="Test annotations JSONL")
    parser.add_argument("--image-dir", required=True, help="Directory containing images")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Loading model from {args.model_path}...")
    processor = PaliGemmaProcessor.from_pretrained(args.model_path)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype
    ).to(device)

    print(f"Loading annotations from {args.test_jsonl}...")
    annotations = load_annotations(args.test_jsonl)
    print(f"Found {len(annotations)} test samples")

    print("\nRunning evaluation...")
    results = evaluate_model(
        model,
        processor,
        annotations,
        args.image_dir,
        device,
        args.max_new_tokens,
    )

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
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
