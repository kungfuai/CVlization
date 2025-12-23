import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from cvlization.paths import resolve_input_path, resolve_output_path

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def load_torchhub_model(device: torch.device, model_name: str = "rtdetrv2_r18vd"):
    model = torch.hub.load("lyuwenyu/RT-DETR", model_name, pretrained=True)
    model.to(device).eval()
    return model


def load_transformers_model(model_id: str, device: torch.device):
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForObjectDetection.from_pretrained(model_id)
    model.to(device).eval()
    return processor, model


def prepare_image(image_path: Path, device: torch.device, resize_to: int = None):
    image = Image.open(image_path).convert("RGB")
    orig_size = (image.height, image.width)
    processed = image.resize((resize_to, resize_to)) if resize_to else image
    array = np.array(processed).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(device)
    return image, tensor, torch.tensor([orig_size], device=device)


def draw_boxes(
    image: Image.Image,
    boxes: List[List[float]],
    labels: List[int],
    scores: List[float],
    id2label: Dict[int, str],
    score_threshold: float,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue
        x0, y0, x1, y1 = box
        label_name = id2label.get(label, str(label))
        caption = f"{label_name} {score:.2f}"
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        text_bbox = draw.textbbox((x0, y0), caption, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        draw.rectangle([x0, y0 - text_h, x0 + text_w, y0], fill="red")
        draw.text((x0, y0 - text_h), caption, fill="white", font=font)
    return image


def run_torchhub(args, device: torch.device, output_dir: Path):
    image, tensor, orig_sizes = prepare_image(
        Path(resolve_input_path(args.image)), device, resize_to=args.input_size
    )
    model = load_torchhub_model(device, model_name=args.model_name)
    with torch.no_grad():
        outputs = model(tensor.unsqueeze(0), orig_sizes)
    # TorchHub deploy() returns tuple (labels, boxes, scores)
    if isinstance(outputs, tuple) and len(outputs) == 3:
        labels, boxes, scores = outputs
        boxes = boxes[0].cpu().tolist()
        scores = scores[0].cpu().tolist()
        labels = labels[0].cpu().tolist()
    else:
        pred = outputs[0]
        boxes = pred["boxes"].cpu().tolist()
        scores = pred["scores"].cpu().tolist()
        labels = pred["labels"].cpu().tolist()

    id2label = {i: name for i, name in enumerate(COCO_CLASSES)}
    annotated = draw_boxes(
        image.copy(),
        boxes,
        labels,
        scores,
        id2label=id2label,
        score_threshold=args.score_threshold,
    )
    save_outputs(output_dir, annotated, boxes, scores, labels, id2label, args)


def run_transformers(args, device: torch.device, output_dir: Path):
    from transformers import AutoImageProcessor

    processor, model = load_transformers_model(args.model_id, device)
    image = Image.open(resolve_input_path(args.image)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    try:
        with torch.no_grad():
            outputs = model(**inputs)
    except Exception as exc:  # pragma: no cover - guidance for missing CUDA dev libs
        raise RuntimeError(
            "Transformers backend failed (likely missing CUDA dev toolkit for "
            "multi-scale deformable attention). Try the torchhub backend or "
            "install CUDA headers and set CUDA_HOME."
        ) from exc

    target_sizes = torch.tensor([image.size[::-1]], device=device)
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=args.score_threshold,
    )[0]

    boxes = results["boxes"].cpu().tolist()
    scores = results["scores"].cpu().tolist()
    labels = results["labels"].cpu().tolist()
    id2label = model.config.id2label

    annotated = draw_boxes(
        image.copy(),
        boxes,
        labels,
        scores,
        id2label=id2label,
        score_threshold=args.score_threshold,
    )
    save_outputs(output_dir, annotated, boxes, scores, labels, id2label, args)


def save_outputs(
    output_dir: Path,
    annotated: Image.Image,
    boxes: List[List[float]],
    scores: List[float],
    labels: List[int],
    id2label: Dict[int, str],
    args,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "prediction.png"
    annotated.save(image_path)

    records = []
    for box, score, label in zip(boxes, scores, labels):
        if score < args.score_threshold:
            continue
        records.append(
            {
                "label": id2label.get(label, str(label)),
                "score": float(score),
                "box_xyxy": [float(x) for x in box],
            }
        )
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved annotated image to {image_path}")
    print(f"Saved detections to {output_dir / 'predictions.json'}")
    print(f"Detections kept (score >= {args.score_threshold}): {len(records)}")


def parse_args():
    parser = argparse.ArgumentParser(description="RT-DETRv2 object detection inference.")
    parser.add_argument(
        "--image",
        type=str,
        default="examples/ref1.png",
        help="Path to the input image.",
    )
    parser.add_argument(
        "--backend",
        choices=["torchhub", "transformers"],
        default="torchhub",
        help="Choose torchhub (default) or transformers backend.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="rtdetrv2_r18vd",
        help="TorchHub model name (for --backend torchhub).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="PekingU/rtdetr_v2_r34vd",
        help="Hugging Face model id (for --backend transformers).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.35,
        help="Confidence threshold for showing detections.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=640,
        help="Square resize for TorchHub backend (models are trained at 640).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/rtdetr",
        help="Where to write annotated image and JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    output_dir = Path(args.output_dir)

    if args.backend == "torchhub":
        print(f"Backend: torchhub (model: {args.model_name})")
        run_torchhub(args, device, output_dir)
    else:
        print(f"Backend: transformers (model: {args.model_id})")
        run_transformers(args, device, output_dir)


if __name__ == "__main__":
    main()
