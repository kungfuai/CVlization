#!/usr/bin/env python
"""
Video Enhancement Inference with Hugging Face Model Download

Downloads model from HF Hub (cached) and runs inference on videos.

Usage:
    python infer.py -i input.mp4 -o output.mp4
    python infer.py -i input.mp4 -o output.mp4 --model composite
    python infer.py -i input.mp4 -o output.mp4 --checkpoint local_model.pt
"""
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple

# HF download
try:
    from huggingface_hub import hf_hub_download
    HAS_HF = True
except ImportError:
    HAS_HF = False
    print("huggingface_hub not installed. Install with: pip install huggingface_hub")


# Model registry
HF_REPO = "zzsi/cvl_models"
MODELS = {
    "nafunet": "video_enhancement/nafunet.pt",
    "composite": "video_enhancement/composite.pt",
    "nafunet_widescale": "video_enhancement/nafunet_widescale.pt",
}


def get_model_path(model_name: str = "nafunet", checkpoint: Optional[str] = None) -> str:
    """Get model path - download from HF if needed."""
    if checkpoint:
        return checkpoint

    if not HAS_HF:
        raise RuntimeError("huggingface_hub required for downloading models")

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    print(f"Downloading {model_name} from HuggingFace...")
    path = hf_hub_download(HF_REPO, MODELS[model_name])
    print(f"Model cached at: {path}")
    return path


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint."""
    from model import TemporalNAFUNet, ExplicitCompositeNet

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model_type = config.get("model", "temporal_nafunet")

    # Build encoder_channels from base channels
    base_channels = config.get("channels", 64)
    encoder_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]

    num_frames = config.get("num_frames", 4)

    if model_type == "composite":
        model = ExplicitCompositeNet(
            in_channels=3,
            out_channels=3,
            encoder_channels=encoder_channels,
            num_frames=num_frames,
        )
    else:
        model = TemporalNAFUNet(
            in_channels=3,
            out_channels=3,
            encoder_channels=encoder_channels,
            num_frames=num_frames,
            mask_guidance=config.get("mask_guidance", "modulation"),
            predict_mask=True,
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded {model_type} model (step {checkpoint.get('step', 'unknown')})")
    return model


def align_to_multiple(x: int, multiple: int = 16) -> int:
    """Round down to nearest multiple."""
    return (x // multiple) * multiple


def process_video(
    model: torch.nn.Module,
    input_path: str,
    output_path: str,
    device: torch.device,
    clip_length: int = 4,
    overlap: int = 2,
    max_size: int = 512,
) -> None:
    """Process video with overlapping clips."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate processing size (preserve aspect ratio, align to 16)
    scale = min(max_size / max(orig_width, orig_height), 1.0)
    proc_width = align_to_multiple(int(orig_width * scale), 16)
    proc_height = align_to_multiple(int(orig_height * scale), 16)

    print(f"Input: {orig_width}x{orig_height} @ {fps:.1f}fps, {total_frames} frames")
    print(f"Processing at: {proc_width}x{proc_height}")

    # Use processing dimensions
    width, height = proc_width, proc_height

    # Setup output (at original resolution)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))

    # Process frames
    frames_buffer = []
    output_frames = []
    stride = clip_length - overlap

    pbar = tqdm(total=total_frames, desc="Processing")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for processing
        frame_resized = cv2.resize(frame, (proc_width, proc_height))

        # Convert BGR to RGB, normalize
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)  # [C, H, W]
        frames_buffer.append(frame_tensor)

        # Process when we have enough frames
        if len(frames_buffer) >= clip_length:
            clip = torch.stack(frames_buffer[:clip_length])  # [T, C, H, W]
            clip = clip.unsqueeze(0).to(device)  # [1, T, C, H, W]

            with torch.no_grad():
                result = model(clip)
                if isinstance(result, tuple):
                    restored = result[0]
                else:
                    restored = result

            # Get output frames
            restored = restored.squeeze(0).cpu()  # [T, C, H, W]

            # For first clip, keep all frames; for subsequent, keep only non-overlapping
            if len(output_frames) == 0:
                keep_from = 0
            else:
                keep_from = overlap

            for i in range(keep_from, clip_length):
                out_frame = restored[i].permute(1, 2, 0).numpy()  # [H, W, C]
                out_frame = (out_frame * 255).clip(0, 255).astype(np.uint8)
                out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
                # Resize back to original resolution
                out_frame = cv2.resize(out_frame, (orig_width, orig_height))
                out.write(out_frame)
                output_frames.append(1)
                pbar.update(1)

            # Keep overlap frames for next iteration
            frames_buffer = frames_buffer[stride:]

    # Process remaining frames
    if len(frames_buffer) > 0:
        # Pad to clip_length if needed
        while len(frames_buffer) < clip_length:
            frames_buffer.append(frames_buffer[-1])

        clip = torch.stack(frames_buffer[:clip_length])
        clip = clip.unsqueeze(0).to(device)

        with torch.no_grad():
            result = model(clip)
            if isinstance(result, tuple):
                restored = result[0]
            else:
                restored = result

        restored = restored.squeeze(0).cpu()

        # Write remaining (non-padded) frames
        remaining = total_frames - len(output_frames)
        for i in range(overlap, overlap + remaining):
            if i < clip_length:
                out_frame = restored[i].permute(1, 2, 0).numpy()
                out_frame = (out_frame * 255).clip(0, 255).astype(np.uint8)
                out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
                # Resize back to original resolution
                out_frame = cv2.resize(out_frame, (orig_width, orig_height))
                out.write(out_frame)
                pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Video Enhancement Inference")
    parser.add_argument("-i", "--input", required=True, help="Input video path (mp4, mov, etc.)")
    parser.add_argument("-o", "--output", required=True, help="Output video path")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="composite",
                        help="Model to use (downloads from HuggingFace)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Local checkpoint path (overrides --model)")
    parser.add_argument("--clip-length", type=int, default=4, help="Frames per clip")
    parser.add_argument("--overlap", type=int, default=2, help="Overlap between clips")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu/mps)")
    args = parser.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    model_path = get_model_path(args.model, args.checkpoint)
    model = load_model(model_path, device)

    # Process video
    process_video(
        model=model,
        input_path=args.input,
        output_path=args.output,
        device=device,
        clip_length=args.clip_length,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
