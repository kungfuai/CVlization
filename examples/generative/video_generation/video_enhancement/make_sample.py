#!/usr/bin/env python
"""
Generate a sample watermarked video for inference testing.

Picks a Pexels Animals video, applies floating text watermark using the same
artifact pipeline as training, and saves the result.

Usage:
    python make_sample.py                          # default output
    python make_sample.py -o my_watermarked.mp4    # custom output
    python make_sample.py --artifact diagonal_text  # specific artifact type
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from visual_artifacts import ArtifactGenerator, apply_overlay_artifact


def pick_video(data_dir: str = None) -> str:
    """Pick a landscape Pexels video (prefer 640x360 / 24-30fps)."""
    import os

    if data_dir is None:
        cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        data_dir = os.path.join(cache_home, "cvlization", "data", "pexels_videos")

    if not os.path.isdir(data_dir):
        # Download via the dataset builder
        from pexels_animals import PexelsAnimalsBuilder
        builder = PexelsAnimalsBuilder()
        builder.prepare()
        data_dir = str(builder.dataset_dir)

    videos = sorted(Path(data_dir).glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No videos found in {data_dir}")

    # Prefer landscape SD videos
    landscape = [v for v in videos if "640_360" in v.name or "sd_640" in v.name]
    if landscape:
        videos = landscape

    random.seed(123)
    return str(random.choice(videos))


def make_watermarked_video(
    input_path: str,
    output_path: str,
    artifact_type: str = "tiled_pattern",
    max_frames: int = 120,
    layers: int = 1,
    min_opacity: float = 0.5,
    max_opacity: float = 0.85,
) -> None:
    """Read video, apply text watermark(s), write output."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(total, max_frames)

    print(f"Source: {input_path}")
    print(f"  {width}x{height} @ {fps:.0f}fps, using {n_frames}/{total} frames")

    # Read frames
    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # [C, H, W]
        frames.append(tensor)
    cap.release()

    if not frames:
        raise RuntimeError("No frames read")

    clean = torch.stack(frames)  # [T, C, H, W]
    print(f"  Loaded {len(frames)} frames: {clean.shape}")

    # Apply artifact layers (stacking multiple overlays for harder examples)
    watermarked = clean
    for layer_idx in range(layers):
        gen = ArtifactGenerator(
            frame_size=(height, width),
            min_opacity=min_opacity,
            max_opacity=max_opacity,
            mode="val",
            size_scale=1.5,
        )
        # Advance the text generator so each layer gets different text
        gen._text_call_count = layer_idx * 100
        mask, _, meta = gen.generate(len(frames), artifact_type=artifact_type)
        print(f"  Layer {layer_idx + 1}/{layers}: {artifact_type}, meta: {meta}")
        watermarked = apply_overlay_artifact(watermarked, mask)

    # Write output video
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(len(frames)):
        frame = watermarked[i].permute(1, 2, 0).numpy()
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"  Saved watermarked video: {output_path} ({len(frames)} frames)")


def main():
    parser = argparse.ArgumentParser(description="Generate sample watermarked video")
    parser.add_argument("-o", "--output", default="examples/sample_watermarked.mp4",
                        help="Output path")
    parser.add_argument("--video", default=None, help="Specific source video path")
    parser.add_argument("--artifact", default="tiled_pattern",
                        choices=["text_overlay", "tiled_pattern", "diagonal_text",
                                 "corner_logo", "channel_logo", "moving_logo"],
                        help="Artifact type to apply")
    parser.add_argument("--max-frames", type=int, default=120,
                        help="Max frames to process")
    parser.add_argument("--layers", type=int, default=1,
                        help="Number of artifact layers to stack (more = harder)")
    parser.add_argument("--min-opacity", type=float, default=0.5,
                        help="Minimum artifact opacity")
    parser.add_argument("--max-opacity", type=float, default=0.85,
                        help="Maximum artifact opacity")
    args = parser.parse_args()

    video_path = args.video or pick_video()
    make_watermarked_video(video_path, args.output, args.artifact, args.max_frames,
                           layers=args.layers, min_opacity=args.min_opacity,
                           max_opacity=args.max_opacity)


if __name__ == "__main__":
    main()
