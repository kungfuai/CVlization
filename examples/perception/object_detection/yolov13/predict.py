import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from ultralytics import YOLO

from cvlization.paths import resolve_input_path, resolve_output_path
from ultralytics.utils import ASSETS

_DEFAULT_BUS = ASSETS / "bus.jpg"
_CAMERA_FALLBACK = Path("/mnt/cvl/workspace/examples/generative/video_generation/phantom/examples/ref1.png")
DEFAULT_IMAGE = str(_DEFAULT_BUS if _DEFAULT_BUS.exists() else _CAMERA_FALLBACK)


def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(args.weights)
    results = model.predict(
        source=resolve_input_path(args.image),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        verbose=False,
    )
    r = results[0]
    boxes = r.boxes
    xyxy = boxes.xyxy.cpu().tolist()
    scores = boxes.conf.cpu().tolist()
    labels = boxes.cls.int().cpu().tolist()
    names = r.names

    output_dir = Path(resolve_output_path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save annotated image (convert BGR to RGB)
    annotated = Image.fromarray(r.plot()[:, :, ::-1])
    image_path = output_dir / "prediction.png"
    annotated.save(image_path)

    records: List[Dict] = []
    for box, score, label in zip(xyxy, scores, labels):
        records.append(
            {
                "label": names.get(label, str(label)),
                "score": float(score),
                "box_xyxy": [float(x) for x in box],
            }
        )

    with open(output_dir / "predictions.json", "w") as f:
        json.dump(records, f, indent=2)

    print(f"Running on device: {device}")
    print(f"Saved annotated image to {image_path}")
    print(f"Saved detections to {output_dir / 'predictions.json'}")
    print(f"Detections kept: {len(records)} (conf >= {args.conf}, iou <= {args.iou})")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv13 object detection inference.")
    parser.add_argument(
        "--image",
        type=str,
        default=str(DEFAULT_IMAGE),
        help="Path to the input image.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov13n.pt",
        help="Weights file or model name (will download if not cached).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (square).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/yolov13",
        help="Directory to save outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
