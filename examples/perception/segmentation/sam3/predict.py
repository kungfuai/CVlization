#!/usr/bin/env python3
"""
SAM3 inference using Hugging Face transformers.

Loads facebook/sam3 via transformers, runs a text prompt on an image, and saves
an overlay. Defaults to the bundled invoice sample and prompt "text".
"""
import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from cvlization.paths import resolve_input_path, resolve_output_path

DEFAULT_IMAGE = "examples/ref1.png"
DEFAULT_OUTPUT = "outputs/sam3/prediction.png"

TRANSFORMERS_AVAILABLE = True
try:
    from transformers import Sam3Model, Sam3Processor  # type: ignore
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run text-prompted SAM3 segmentation using transformers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to the input image (default: bundled ref1.png camera sample)",
    )
    parser.add_argument(
        "--text",
        default="camera",
        help="Text prompt to segment (default: 'camera' for the sample image)",
    )
    parser.add_argument(
        "--checkpoint",
        default="facebook/sam3",
        help="Hugging Face model id or local checkpoint path",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to save the RGBA overlay with masks",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Mask binarization threshold",
    )
    parser.add_argument(
        "--model_loader",
        choices=["transformers", "repo"],
        default="transformers",
        help="Use HF transformers Sam3Model (default) or the cloned sam3 repo backend",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Instance score threshold for mask filtering (transformers backend)",
    )
    return parser.parse_args()


def overlay_masks(image: Image.Image, masks: torch.Tensor) -> Image.Image:
    image = image.convert("RGBA")
    masks_np = masks.cpu().numpy().astype(np.uint8)
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]

    rng = np.random.default_rng(42)
    colors: Sequence[np.ndarray] = rng.integers(0, 255, size=(masks_np.shape[0], 3), dtype=np.uint8)

    for mask, color in zip(masks_np, colors):
        mask_img = Image.fromarray((mask * 180).astype(np.uint8)).resize(image.size)
        overlay = Image.new("RGBA", image.size, (*color.tolist(), 0))
        overlay.putalpha(mask_img)
        image = Image.alpha_composite(image, overlay)

    return image


def main() -> None:
    args = parse_args()

    # Defaults are local to example dir; user-provided paths resolve to cwd
    if args.image is None:
        image_path = Path(DEFAULT_IMAGE)
        print(f"No --image provided, using bundled sample: {image_path}")
    else:
        image_path = Path(resolve_input_path(args.image))
    # Output always resolves to user's cwd
    output_path = Path(resolve_output_path(args.output))
    if args.text == "text":
        print("No --text provided, using default prompt: 'text'")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Reading image: {image_path}")
    image = Image.open(image_path).convert("RGB")

    use_transformers = args.model_loader == "transformers" and TRANSFORMERS_AVAILABLE

    if use_transformers:
        print(f"Loading transformers model '{args.checkpoint}' on device {device}...")
        model = Sam3Model.from_pretrained(args.checkpoint).to(device).eval()
        processor = Sam3Processor.from_pretrained(args.checkpoint)

        inputs = processor(images=image, text=args.text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = results.get("masks")
        boxes = results.get("boxes")
        scores = results.get("scores")
    else:
        if args.model_loader == "transformers":
            print("Transformers Sam3Model not available; falling back to native SAM3 repo.")
        else:
            print("Using native SAM3 repo backend.")
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor as NativeProcessor

        model = build_sam3_image_model(device=str(device))
        processor = NativeProcessor(model)
        state = processor.set_image(image)
        result = processor.set_text_prompt(state=state, prompt=args.text)
        masks = result.get("masks")
        boxes = result.get("boxes")
        scores = result.get("scores")

    if masks is None or len(masks) == 0:
        print("No masks returned.")
        return

    overlay = overlay_masks(image, masks)
    overlay.save(output_path)
    print(f"Done. Found {len(masks)} masks. Saved overlay to: {output_path}")

    if boxes is not None and len(boxes):
        print(f"First box: {boxes[0].tolist()}")
    if scores is not None and len(scores):
        print(f"Top scores: {scores[:5].tolist()}")


if __name__ == "__main__":
    main()
