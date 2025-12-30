#!/usr/bin/env python3
"""
OWL-ViT: Open-Vocabulary Object Detection with Vision Transformers

This script demonstrates open-vocabulary object detection using OWL-ViT/OWLv2,
which can detect objects based on free-text queries without training on specific classes.

Features:
- Text-conditioned object detection (open-vocabulary)
- Image-conditioned object detection (one-shot)
- Multiple model variants (v1 and v2, different sizes)
- Zero-shot detection on arbitrary object categories
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    Owlv2Processor,
    Owlv2ForObjectDetection,
    OwlViTProcessor,
    OwlViTForObjectDetection,
)

# CVL dual-mode execution support
from cvlization.paths import (
    resolve_input_path,
    resolve_output_path,
)

# Model configurations
MODELS = {
    # OWL-ViT v1 models
    "owlvit-base-patch32": "google/owlvit-base-patch32",
    "owlvit-base-patch16": "google/owlvit-base-patch16",
    "owlvit-large-patch14": "google/owlvit-large-patch14",
    # OWL-ViT v2 models (newer, better performance)
    "owlv2-base-patch16": "google/owlv2-base-patch16",
    "owlv2-base-patch16-ensemble": "google/owlv2-base-patch16-ensemble",
    "owlv2-large-patch14": "google/owlv2-large-patch14",
}

DEFAULT_MODEL = "owlv2-base-patch16"
DEFAULT_QUERIES = ["a photo of a person", "a photo of a necklace", "a photo of a shirt"]


def detect_device():
    """
    Detect the best available device for inference.

    Returns:
        torch.device: The device to use
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU detected)")

    return device


def load_model(model_id: str, device: torch.device):
    """
    Load OWL-ViT or OWLv2 model and processor.

    Args:
        model_id: Model identifier or HuggingFace model path
        device: Device to load model on

    Returns:
        tuple: (processor, model)
    """
    # Resolve model ID from short name if needed
    if model_id in MODELS:
        full_model_id = MODELS[model_id]
    else:
        full_model_id = model_id

    print(f"Loading model: {full_model_id}...")

    # Determine if v1 or v2
    is_v2 = "owlv2" in full_model_id.lower()

    if is_v2:
        processor = Owlv2Processor.from_pretrained(full_model_id)
        model = Owlv2ForObjectDetection.from_pretrained(full_model_id)
    else:
        processor = OwlViTProcessor.from_pretrained(full_model_id)
        model = OwlViTForObjectDetection.from_pretrained(full_model_id)

    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully (v{'2' if is_v2 else '1'})")
    return processor, model


def detect_objects_text(
    image: Image.Image,
    text_queries: List[str],
    processor,
    model,
    device: torch.device,
    score_threshold: float = 0.1,
) -> Tuple[List[dict], torch.Tensor]:
    """
    Detect objects in an image using text queries.

    Args:
        image: PIL Image
        text_queries: List of text descriptions to search for
        processor: Model processor
        model: Detection model
        device: Device to run inference on
        score_threshold: Minimum confidence score for detections

    Returns:
        tuple: (list of detections, scores tensor)
    """
    # Preprocess inputs
    inputs = processor(text=text_queries, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        threshold=score_threshold,
        target_sizes=target_sizes
    )[0]

    # Format detections
    detections = []
    for box, score, label_id in zip(
        results["boxes"].cpu(),
        results["scores"].cpu(),
        results["labels"].cpu()
    ):
        detections.append({
            "box": box.tolist(),  # [x_min, y_min, x_max, y_max]
            "score": float(score),
            "label": text_queries[label_id],
            "label_id": int(label_id),
        })

    return detections, results["scores"]


def visualize_detections(
    image: Image.Image,
    detections: List[dict],
    output_path: str,
    show_labels: bool = True,
) -> None:
    """
    Visualize detections on the image and save to file.

    Args:
        image: Original PIL Image
        detections: List of detection dictionaries
        output_path: Path to save annotated image
        show_labels: Whether to show labels on the image
    """
    # Create a copy to draw on
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Define colors for different labels
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
        "#F7DC6F", "#BB8FCE", "#85C1E2", "#F8B739", "#52B788"
    ]

    # Group detections by label for color consistency
    label_colors = {}

    for det in detections:
        label = det["label"]
        box = det["box"]
        score = det["score"]

        # Assign color based on label
        if label not in label_colors:
            label_colors[label] = colors[len(label_colors) % len(colors)]
        color = label_colors[label]

        # Draw bounding box
        x_min, y_min, x_max, y_max = box
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)

        # Draw label and score
        if show_labels:
            text = f"{label}: {score:.2f}"
            # Get text bounding box for background
            bbox = draw.textbbox((x_min, y_min - 25), text, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x_min, y_min - 25), text, fill="white", font=font)

    # Save annotated image
    img_draw.save(output_path)
    print(f"Annotated image saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="OWL-ViT Open-Vocabulary Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect cats and dogs in an image
  python predict.py --image photo.jpg --queries "a photo of a cat" "a photo of a dog"

  # Use a specific model variant
  python predict.py --image photo.jpg --model owlv2-large-patch14

  # Save results as JSON
  python predict.py --image photo.jpg --output results.json --format json

  # Adjust detection threshold
  python predict.py --image photo.jpg --threshold 0.2
        """
    )

    # Input/output arguments
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image (default: uses bundled sample)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="detections.jpg",
        help="Path to output annotated image (default: outputs/detections.jpg)"
    )
    parser.add_argument(
        "--format",
        choices=["image", "json", "both"],
        default="both",
        help="Output format: image (annotated), json (detections), or both (default: both)"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=list(MODELS.keys()),
        help=f"Model variant to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Custom HuggingFace model path (overrides --model)"
    )

    # Detection arguments
    parser.add_argument(
        "--queries",
        nargs="+",
        default=DEFAULT_QUERIES,
        help='Text queries for object detection (default: ["a photo of a person", "a photo of a necklace", "a photo of a shirt"])'
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Detection confidence threshold (default: 0.1)"
    )

    # Visualization arguments
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Don't show labels on visualizations"
    )

    # Device argument
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="Device to use (default: auto)"
    )

    args = parser.parse_args()

    # Handle bundled sample vs user-provided input
    DEFAULT_SAMPLE = "examples/human.png"
    if args.image is None:
        # Use bundled sample directly (don't resolve through CVL_INPUTS)
        image_path = DEFAULT_SAMPLE
        print(f"No --image provided, using bundled sample: {DEFAULT_SAMPLE}")
    else:
        # User-provided file - resolve through CVL_INPUTS
        image_path = resolve_input_path(args.image)

    # Resolve output path
    output_path = resolve_output_path(args.output)

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Setup device
    if args.device == "auto":
        device = detect_device()
    else:
        device = torch.device(args.device)
        print(f"Using device: {device}")

    # Load model
    model_id = args.model_path if args.model_path else args.model
    processor, model = load_model(model_id, device)

    # Load image
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}")

    # Run detection
    print(f"\nRunning detection with queries: {args.queries}")
    print(f"Confidence threshold: {args.threshold}")

    detections, scores = detect_objects_text(
        image=image,
        text_queries=args.queries,
        processor=processor,
        model=model,
        device=device,
        score_threshold=args.threshold,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"Found {len(detections)} detections:")
    print(f"{'='*60}")

    for i, det in enumerate(detections, 1):
        print(f"{i}. {det['label']}")
        print(f"   Score: {det['score']:.3f}")
        print(f"   Box: [{det['box'][0]:.1f}, {det['box'][1]:.1f}, "
              f"{det['box'][2]:.1f}, {det['box'][3]:.1f}]")

    if len(detections) == 0:
        print("No objects detected. Try:")
        print("  - Lowering the threshold (--threshold)")
        print("  - Using different text queries (--queries)")
        print("  - Trying a different model (--model)")

    # Save outputs
    if args.format in ["image", "both"]:
        visualize_detections(
            image=image,
            detections=detections,
            output_path=output_path,
            show_labels=not args.no_labels,
        )

    if args.format in ["json", "both"]:
        # Save JSON results
        json_path = output_path.rsplit(".", 1)[0] + ".json"
        results = {
            "image": str(image_path),
            "image_size": list(image.size),
            "model": model_id,
            "queries": args.queries,
            "threshold": args.threshold,
            "num_detections": len(detections),
            "detections": detections,
        }

        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"JSON results saved to: {json_path}")

    print(f"\n{'='*60}")
    print("Detection complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
