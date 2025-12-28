#!/usr/bin/env python3
"""
Run DocLayout-YOLO for document layout detection on a single image.

Detects 11 layout element classes:
- Caption, Footnote, Formula, List-item, Page-footer, Page-header,
  Picture, Section-header, Table, Text, Title
"""
import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from cvlization.paths import (
    get_input_dir,
    get_output_dir,
    resolve_input_path,
    resolve_output_path,
)

MODEL_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
MODEL_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"
DEFAULT_IMAGE = "/cvlization_repo/examples/perception/doc_ai/leaderboard/test_data/sample.jpg"


def load_model(device: str):
    """Download and load the DocLayout-YOLO model."""
    from doclayout_yolo import YOLOv10

    print(f"Downloading model from {MODEL_REPO}...")
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    print(f"Loading model on {device}...")
    model = YOLOv10(model_path)
    return model


def run_inference(model, image_path: Path, conf: float, imgsz: int, device: str):
    """Run layout detection on an image."""
    results = model.predict(
        str(image_path),
        imgsz=imgsz,
        conf=conf,
        device=device,
    )
    return results[0]  # Single image, return first result


def format_detections(result) -> list:
    """Convert detection results to a list of dictionaries."""
    detections = []
    boxes = result.boxes
    names = result.names

    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]

        detections.append({
            "class": cls_name,
            "class_id": cls_id,
            "confidence": round(conf, 4),
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            },
        })

    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DocLayout-YOLO document layout detection"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an input image (default: bundled sample).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="detections.json",
        help="Where to write the detection results (JSON).",
    )
    parser.add_argument(
        "--output-image",
        type=str,
        default="annotated.jpg",
        help="Where to save the annotated image.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Input image size for the model.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to run on (default: cuda if available).",
    )
    parser.add_argument(
        "--no-annotated-image",
        action="store_true",
        help="Skip saving the annotated image.",
    )
    args = parser.parse_args()

    input_dir = get_input_dir()
    output_dir = get_output_dir()

    # Resolve paths: None means use bundled sample, otherwise resolve to user's cwd
    if args.image is None:
        image_path = Path(DEFAULT_IMAGE)
        print(f"No --image provided, using bundled sample: {image_path}")
    else:
        image_path = Path(resolve_input_path(args.image, input_dir))
    if not image_path.exists() and args.image is None:
        fallbacks = [
            Path("/cvlization_repo/examples/perception/doc_ai/leaderboard/test_data/sample.jpg"),
            Path(__file__).resolve().parent.parent / "leaderboard" / "test_data" / "sample.jpg",
        ]
        for cand in fallbacks:
            if cand.exists():
                print(f"Input not found at {image_path}, using sample: {cand}")
                image_path = cand
                break
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Resolve output paths
    output_path = Path(resolve_output_path(args.output, output_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_image_path = Path(resolve_output_path(args.output_image, output_dir))
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    # Device selection
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")

    # Load model and run inference
    model = load_model(device)
    print(f"Running inference on {image_path}...")
    result = run_inference(model, image_path, args.conf, args.imgsz, device)

    # Format and save detections
    detections = format_detections(result)
    output_data = {
        "image": str(image_path),
        "model": MODEL_REPO,
        "confidence_threshold": args.conf,
        "image_size": args.imgsz,
        "num_detections": len(detections),
        "detections": detections,
    }
    output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print(f"Saved {len(detections)} detections to {output_path}")

    # Save annotated image
    if not args.no_annotated_image:
        annotated = result.plot(pil=True)
        if isinstance(annotated, Image.Image):
            annotated.save(output_image_path)
        else:
            # If numpy array, convert to PIL
            Image.fromarray(annotated).save(output_image_path)
        print(f"Saved annotated image to {output_image_path}")

    # Print summary
    print(f"\nDetected {len(detections)} layout elements:")
    class_counts = {}
    for det in detections:
        cls = det["class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
