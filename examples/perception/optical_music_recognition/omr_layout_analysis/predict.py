#!/usr/bin/env python3
"""
OMR Layout Analysis — Sheet Music Page Segmentation

Detects systems, grand staves, staves, and measures on full-page piano scores
using a YOLOv8 model (OLA v2.0) trained on ~7k images with ~500k annotations.

Model:  https://github.com/v-dvorak/omr-layout-analysis (OLA v2.0)
Weights: ~40MB, downloaded automatically from GitHub Releases.

Output: JSON with bounding boxes per class + optional cropped system images.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    from cvlization.paths import resolve_input_path, resolve_output_path
    CVL_AVAILABLE = True
except ImportError:
    CVL_AVAILABLE = False
    def resolve_input_path(p):  return Path(p)
    def resolve_output_path(p): return Path(p)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_DIR   = Path("/root/.cache/huggingface/cvl_data/omr_layout_analysis")
MODEL_URL   = ("https://github.com/v-dvorak/omr-layout-analysis/releases/"
               "download/ola-v2.0/ola-layout-analysis-2.0-2025-03-09.pt")
MODEL_CACHE = CACHE_DIR / "ola-layout-analysis-2.0.pt"

SAMPLE_IMAGE_URL = ("https://huggingface.co/datasets/zzsi/cvl/resolve/main/"
                    "omr_layout_analysis/sample_page.jpg")
SAMPLE_IMAGE_CACHE = Path("/root/.cache/huggingface/cvl_data/omr_layout_analysis/sample_page.jpg")

CLASSES = ["system_measures", "stave_measures", "staves", "systems", "grand_staff"]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def ensure_model() -> Path:
    if MODEL_CACHE.exists():
        return MODEL_CACHE
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model weights from GitHub Releases...")
    import urllib.request
    urllib.request.urlretrieve(MODEL_URL, MODEL_CACHE)
    print(f"Saved to {MODEL_CACHE}")
    return MODEL_CACHE


def ensure_sample_image() -> Path:
    if SAMPLE_IMAGE_CACHE.exists():
        return SAMPLE_IMAGE_CACHE
    SAMPLE_IMAGE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading sample image from HuggingFace...")
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="zzsi/cvl",
        repo_type="dataset",
        filename="omr_layout_analysis/sample_page.jpg",
        local_dir=CACHE_DIR.parent,
    )
    return SAMPLE_IMAGE_CACHE


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_detection(image_path: Path, model_path: Path, conf_threshold: float = 0.3) -> dict:
    model = YOLO(str(model_path))
    results = model.predict(str(image_path), verbose=False, conf=conf_threshold)
    result = results[0]

    img_h, img_w = result.boxes.orig_shape
    output = {"width": img_w, "height": img_h}

    for cls_id, cls_name in enumerate(CLASSES):
        output[cls_name] = []

    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf   = float(box.conf.item())
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"class_{cls_id}"
        output[cls_name].append({
            "left": x1, "top": y1,
            "width": x2 - x1, "height": y2 - y1,
            "conf": round(conf, 3),
        })

    # Sort each class by top-to-bottom reading order
    for cls_name in CLASSES:
        output[cls_name].sort(key=lambda b: b["top"])

    return output


def save_crops(image_path: Path, detections: dict, output_dir: Path, cls_name: str = "grand_staff"):
    """Save cropped images for each detected instance of cls_name."""
    boxes = detections.get(cls_name, [])
    if not boxes:
        return []
    img = cv2.imread(str(image_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_paths = []
    for i, box in enumerate(boxes):
        x, y, w, h = box["left"], box["top"], box["width"], box["height"]
        crop = img[y:y+h, x:x+w]
        crop_path = output_dir / f"{cls_name}_{i:02d}.jpg"
        cv2.imwrite(str(crop_path), crop)
        crop_paths.append(str(crop_path))
    return crop_paths


def print_summary(detections: dict, crop_paths: list):
    print()
    print("=" * 50)
    print("DETECTION SUMMARY:")
    print("=" * 50)
    for cls_name in CLASSES:
        boxes = detections.get(cls_name, [])
        print(f"  {cls_name:20s}: {len(boxes):3d} detected")
    if crop_paths:
        print()
        print(f"Saved {len(crop_paths)} grand staff crop(s):")
        for p in crop_paths:
            print(f"  {p}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OMR Layout Analysis — detect systems/staves on sheet music pages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py
  python predict.py --image score.jpg
  python predict.py --image score.jpg --output detections.json --crops crops/
  python predict.py --image score.jpg --conf 0.25
""",
    )
    parser.add_argument("--image",  default=None,
                        help="Input score image (default: auto-download vintage sample)")
    parser.add_argument("--output", default=None,
                        help="Path to save detection JSON (default: print to stdout)")
    parser.add_argument("--crops",  default=None,
                        help="Directory to save cropped grand staff images")
    parser.add_argument("--conf",   type=float, default=0.3,
                        help="Confidence threshold (default: 0.3)")
    args = parser.parse_args()

    # Resolve image
    if args.image:
        image_path = resolve_input_path(args.image)
    else:
        image_path = ensure_sample_image()

    model_path = ensure_model()

    print("=" * 50)
    print("OMR Layout Analysis")
    print("=" * 50)
    print(f"  Image : {image_path}")
    print(f"  Model : {model_path.name}")
    print(f"  Conf  : {args.conf}")
    print("=" * 50)

    detections = run_detection(image_path, model_path, conf_threshold=args.conf)

    # Save crops of grand staff systems
    crop_paths = []
    if args.crops:
        crops_dir = resolve_output_path(args.crops)
        crop_paths = save_crops(image_path, detections, Path(crops_dir), cls_name="grand_staff")

    print_summary(detections, crop_paths)

    # Add crop paths into output JSON
    if crop_paths:
        detections["grand_staff_crops"] = crop_paths

    # Output JSON
    if args.output:
        out_path = resolve_output_path(args.output)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(detections, indent=2))
        print(f"\nDetections saved to: {out_path}")
    else:
        print("\nDetections (JSON):")
        print(json.dumps(detections, indent=2))


if __name__ == "__main__":
    main()
