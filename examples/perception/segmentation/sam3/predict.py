#!/usr/bin/env python3
"""
SAM3 inference: text-prompted segmentation for images and videos.

Image mode: Loads facebook/sam3 via transformers (or native repo fallback),
runs a text prompt on an image, and saves an overlay PNG.

Video mode: Uses the native SAM3 repo video predictor to detect, segment,
and track objects across video frames. Outputs an overlay MP4 and optional
per-frame binary mask PNGs (compatible with ProPainter --mask folder format).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image

from cvlization.paths import resolve_input_path, resolve_output_path

DEFAULT_OUTPUT_IMAGE = "prediction.png"
DEFAULT_OUTPUT_VIDEO = "prediction_video.mp4"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

ASSETS_REPO = "zzsi/cvl"
SAMPLE_VIDEO_PATH = "sam3/soccer_360p.mp4"


def get_sample_video() -> str:
    """Return local path to sample video, downloading from HuggingFace if needed."""
    filename = os.path.basename(SAMPLE_VIDEO_PATH)
    local_path = os.path.join(os.path.dirname(__file__), "examples", filename)
    if os.path.exists(local_path):
        return local_path

    print(f"Downloading sample video: {filename}")
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=ASSETS_REPO,
        filename=SAMPLE_VIDEO_PATH,
        repo_type="dataset",
    )
    return downloaded

TRANSFORMERS_AVAILABLE = True
try:
    from transformers import Sam3Model, Sam3Processor  # type: ignore
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run text-prompted SAM3 segmentation on images or videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Path to input image or video (auto-detected by extension)",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="(Deprecated, use --input) Path to input image",
    )
    parser.add_argument(
        "--text",
        default="text",
        help="Text prompt to segment",
    )
    parser.add_argument(
        "--checkpoint",
        default="facebook/sam3",
        help="Hugging Face model id or local checkpoint path",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: prediction.png for images, prediction_video.mp4 for videos)",
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
        help="Use HF transformers Sam3Model (default) or the cloned sam3 repo backend (image only)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.1,
        help="Instance score threshold for mask filtering (transformers backend)",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="(Video only) Save per-frame binary mask PNGs for ProPainter compatibility",
    )
    parser.add_argument(
        "--propagation-direction",
        choices=["forward", "backward", "both"],
        default="forward",
        help="(Video only) Direction to propagate tracking from the prompted frame",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help="(Video only) Cap the number of tracked objects to limit GPU memory usage. "
        "Keeps the top-scoring detections. Recommended: 5-8 for 24GB GPUs at 1080p.",
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


def overlay_masks_cv2(frame_bgr: np.ndarray, masks: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay boolean masks on a BGR frame using consistent colors."""
    out = frame_bgr.copy()
    rng = np.random.default_rng(42)
    n_objects = masks.shape[0] if masks.ndim == 3 else 1
    colors = rng.integers(0, 255, size=(n_objects, 3), dtype=np.uint8)

    if masks.ndim == 2:
        masks = masks[None, ...]

    for mask, color in zip(masks, colors):
        h, w = frame_bgr.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        color_overlay = np.zeros_like(frame_bgr)
        color_overlay[mask] = color.tolist()
        out = np.where(mask[..., None], cv2.addWeighted(out, 1 - alpha, color_overlay, alpha, 0), out)

    return out


def process_image(args: argparse.Namespace, image_path: Path, output_path: Path) -> None:
    """Run SAM3 image segmentation (existing logic)."""
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


def process_video(args: argparse.Namespace, input_path: Path, output_path: Path) -> None:
    """Run SAM3 video segmentation using the native repo video predictor."""
    print("Video mode: using native SAM3 repo video predictor.")
    from sam3.model_builder import build_sam3_video_predictor

    # Build predictor and start session
    print("Building SAM3 video predictor...")
    predictor = build_sam3_video_predictor(gpus_to_use=[0])
    print(f"Starting session for: {input_path}")
    resp = predictor.start_session(resource_path=str(input_path))
    session_id = resp["session_id"]

    # Add text prompt on frame 0
    print(f"Adding text prompt on frame 0: '{args.text}'")
    prompt_result = predictor.add_prompt(session_id, frame_idx=0, text=args.text)

    # Cap tracked objects if --max-objects is set
    if args.max_objects is not None:
        outputs = prompt_result["outputs"]
        obj_ids = np.asarray(outputs["out_obj_ids"])
        scores = np.asarray(outputs["out_probs"])
        n_detected = len(obj_ids)
        if n_detected > args.max_objects:
            # Keep top-scoring objects, remove the rest
            top_indices = np.argsort(scores)[::-1][: args.max_objects]
            keep_ids = set(obj_ids[top_indices].tolist())
            remove_ids = [int(oid) for oid in obj_ids if int(oid) not in keep_ids]
            for oid in remove_ids:
                predictor.remove_object(session_id, obj_id=oid)
            print(
                f"Capped objects: kept {args.max_objects} of {n_detected} "
                f"(removed {len(remove_ids)} lowest-scoring)"
            )

    # Propagate through video
    print(f"Propagating masks ({args.propagation_direction})...")
    frame_masks = {}  # frame_idx -> (N, H, W) bool array
    for result in predictor.propagate_in_video(
        session_id,
        propagation_direction=args.propagation_direction,
        start_frame_idx=0,
        max_frame_num_to_track=None,
    ):
        fidx = result["frame_index"]
        masks = result["outputs"]["out_binary_masks"]  # (N_objects, H, W) bool
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        frame_masks[fidx] = masks

    predictor.close_session(session_id)
    print(f"Tracking complete: {len(frame_masks)} frames, "
          f"{frame_masks[0].shape[0] if 0 in frame_masks else '?'} object(s)")

    if not frame_masks:
        print("No masks returned from video predictor.")
        return

    # Save per-frame binary masks if requested
    if args.save_masks:
        masks_dir = output_path.parent / (output_path.stem + "_masks")
        masks_dir.mkdir(parents=True, exist_ok=True)
        for fidx in sorted(frame_masks.keys()):
            masks = frame_masks[fidx]
            # Union all object masks into a single binary mask per frame
            union_mask = np.any(masks, axis=0).astype(np.uint8) * 255
            mask_path = masks_dir / f"{fidx:04d}.png"
            cv2.imwrite(str(mask_path), union_mask)
        print(f"Saved {len(frame_masks)} binary mask PNGs to: {masks_dir}/")

    # Re-read video and write overlay MP4
    _write_overlay_video(input_path, output_path, frame_masks)


def _write_overlay_video(
    input_path: Path, output_path: Path, frame_masks: dict
) -> None:
    """Read source video, overlay colored masks, write MP4 via ffmpeg."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: cannot open video {input_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    print(f"Writing overlay video to: {output_path}")
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fidx in frame_masks:
            frame = overlay_masks_cv2(frame, frame_masks[fidx])
        proc.stdin.write(frame.tobytes())
        fidx += 1

    cap.release()
    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode()
        print(f"ffmpeg error (exit {proc.returncode}): {stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"Done. Wrote {fidx} frames to: {output_path}")


def main() -> None:
    args = parse_args()

    # Resolve input: --input > --image (deprecated) > sample video
    raw_input = args.input or args.image
    if args.image and not args.input:
        print("Warning: --image is deprecated, use --input instead.")

    if raw_input is None:
        input_path = Path(get_sample_video())
        print(f"No --input provided, using sample video: {input_path}")
        if args.text == "text":
            args.text = "person"
            print("Using default prompt 'person' for sample video")
    else:
        input_path = Path(resolve_input_path(raw_input))

    ext = input_path.suffix.lower()
    is_video = ext in VIDEO_EXTENSIONS

    # Resolve output path
    if args.output:
        output_path = Path(resolve_output_path(args.output))
    else:
        default = DEFAULT_OUTPUT_VIDEO if is_video else DEFAULT_OUTPUT_IMAGE
        output_path = Path(resolve_output_path(default))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if is_video:
        process_video(args, input_path, output_path)
    else:
        process_image(args, input_path, output_path)


if __name__ == "__main__":
    main()
