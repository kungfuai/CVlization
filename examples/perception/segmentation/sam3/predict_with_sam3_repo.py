#!/usr/bin/env python3
"""
Minimal SAM3 inference runner.

Loads a SAM3 checkpoint (Hugging Face gated weights or a local finetuned ckpt),
runs text-prompted segmentation on a single image, and saves a mask overlay.
"""
import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run text-prompted SAM3 segmentation on an image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        default="examples/sample.jpg",
        help="Path to the input image (default: bundled sample.jpg)",
    )
    parser.add_argument(
        "--text",
        default="cat",
        help="Text prompt describing the concept(s) to segment (default: 'cat')",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Local checkpoint path. If omitted, downloads gated HF weights (requires HF_TOKEN).",
    )
    parser.add_argument(
        "--output",
        default="outputs/sam3/prediction.png",
        help="Path to save the RGBA overlay with masks",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Mask binarization threshold",
    )
    return parser.parse_args()


def _ensure_3d_mask(masks: torch.Tensor) -> torch.Tensor:
    if masks.ndim == 2:
        return masks.unsqueeze(0)
    if masks.ndim == 4:
        # (B, N, H, W) -> (N, H, W) assuming single image batch
        return masks.squeeze(0)
    return masks


def overlay_masks(image: Image.Image, masks: torch.Tensor) -> Image.Image:
    image = image.convert("RGBA")
    masks = masks.cpu().numpy().astype(np.uint8)
    if masks.ndim == 2:
        masks = masks[None, ...]

    rng = np.random.default_rng(42)
    colors: Sequence[np.ndarray] = rng.integers(0, 255, size=(masks.shape[0], 3), dtype=np.uint8)

    for mask, color in zip(masks, colors):
        mask_img = Image.fromarray((mask * 180).astype(np.uint8)).resize(image.size)
        overlay = Image.new("RGBA", image.size, (*color.tolist(), 0))
        overlay.putalpha(mask_img)
        image = Image.alpha_composite(image, overlay)

    return image


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    output_path = Path(args.output)
    if args.image == "examples/sample.jpg":
        print(f"No --image provided, using default sample: {image_path}")
    if args.text == "cat":
        print("No --text provided, using default prompt: 'cat'")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on device {device}...")
    model = build_sam3_image_model(
        device=str(device),
        checkpoint_path=args.checkpoint,
        load_from_HF=args.checkpoint is None,
    )
    processor = Sam3Processor(model)

    print(f"Reading image: {image_path}")
    image = Image.open(image_path).convert("RGB")

    state = processor.set_image(image)
    result = processor.set_text_prompt(state=state, prompt=args.text)

    masks = result.get("masks")
    if masks is None or masks.numel() == 0:
        print("No masks returned.")
        return

    masks = _ensure_3d_mask(masks)
    masks = (masks > args.mask_threshold).to(torch.uint8)

    overlay = overlay_masks(image, masks)
    overlay.save(output_path)

    boxes = result.get("boxes")
    scores = result.get("scores")
    num_masks = masks.shape[0]
    print(f"Done. Found {num_masks} masks. Saved overlay to: {output_path}")

    if boxes is not None and scores is not None:
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        top_scores = scores[: min(5, len(scores))]
        print(f"Top scores: {top_scores}")
        if len(boxes):
            print(f"First box: {boxes[0].tolist()}")


if __name__ == "__main__":
    main()
