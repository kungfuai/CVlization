#!/usr/bin/env python
"""
Video Enhancement Inference with Hugging Face Model Download

Downloads model from HF Hub (cached) and runs inference on videos.

Usage:
    python infer.py -i input.mp4 -o output.mp4
    python infer.py -i input.mp4 -o output.mp4 --model nafunet
    python infer.py -i input.mp4 -o output.mp4 --side-by-side
    python infer.py -i input.mp4 -o output.mp4 --checkpoint local_model.pt
"""
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, List

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


def run_clip(model, clip: torch.Tensor, device: torch.device):
    """
    Run model on a clip and return (restored, mask).

    Args:
        clip: [T, C, H, W] tensor
    Returns:
        restored: [T, C, H, W] on cpu
        mask: [T, 1, H, W] on cpu, or None if model doesn't predict masks
    """
    clip_input = clip.unsqueeze(0).to(device)  # [1, T, C, H, W]

    with torch.no_grad():
        result = model(clip_input)

    if isinstance(result, tuple):
        restored = result[0].squeeze(0).cpu()
        mask = result[1].squeeze(0).cpu()
    else:
        restored = result.squeeze(0).cpu()
        mask = None

    return restored, mask


def tensor_to_bgr(tensor: torch.Tensor, target_size: Tuple[int, int] = None) -> np.ndarray:
    """Convert [C, H, W] float tensor to BGR uint8 numpy array."""
    frame = tensor.permute(1, 2, 0).numpy()
    frame = (frame * 255).clip(0, 255).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if target_size:
        frame = cv2.resize(frame, target_size)
    return frame


def mask_to_bgr(mask: torch.Tensor, target_size: Tuple[int, int] = None) -> np.ndarray:
    """Convert [1, H, W] mask tensor to BGR heatmap."""
    m = mask.squeeze(0).numpy()
    m = (m * 255).clip(0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(m, cv2.COLORMAP_JET)
    if target_size:
        heatmap = cv2.resize(heatmap, target_size)
    return heatmap


def process_video(
    model: torch.nn.Module,
    input_path: str,
    output_path: str,
    device: torch.device,
    clip_length: int = 4,
    overlap: int = 2,
    max_size: int = 512,
    side_by_side: bool = False,
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

    # Output dimensions
    if side_by_side:
        # 3 panels: input | mask | output, each at original resolution
        out_width = orig_width * 3
        out_height = orig_height
        print(f"Side-by-side output: {out_width}x{out_height} (input | mask | output)")
    else:
        out_width = orig_width
        out_height = orig_height

    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    # Read all input frames (for side-by-side we need the originals)
    input_frames_bgr = []  # original resolution BGR frames for side-by-side
    frames_buffer = []     # processing resolution tensors
    all_frames_tensors = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if side_by_side:
            input_frames_bgr.append(frame.copy())

        frame_resized = cv2.resize(frame, (proc_width, proc_height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)
        all_frames_tensors.append(frame_tensor)

    cap.release()
    total_frames = len(all_frames_tensors)

    # Process clips
    output_restored = [None] * total_frames  # [C, H, W] tensors
    output_masks = [None] * total_frames     # [1, H, W] tensors or None
    stride = clip_length - overlap

    pbar = tqdm(total=total_frames, desc="Processing")
    written = 0
    pos = 0

    while pos < total_frames:
        # Build clip
        end = min(pos + clip_length, total_frames)
        clip_frames = all_frames_tensors[pos:end]

        # Pad if needed
        while len(clip_frames) < clip_length:
            clip_frames.append(clip_frames[-1])

        clip = torch.stack(clip_frames)
        restored, mask = run_clip(model, clip, device)

        # Determine which frames to keep
        if pos == 0:
            keep_from = 0
        else:
            keep_from = overlap

        actual_end = min(pos + clip_length, total_frames)
        for i in range(keep_from, actual_end - pos):
            idx = pos + i
            if idx < total_frames and output_restored[idx] is None:
                output_restored[idx] = restored[i]
                if mask is not None:
                    output_masks[idx] = mask[i]
                written += 1
                pbar.update(1)

        pos += stride

    pbar.close()

    # Write output
    target_size = (orig_width, orig_height)
    for idx in range(total_frames):
        if output_restored[idx] is None:
            continue

        restored_bgr = tensor_to_bgr(output_restored[idx], target_size)

        if side_by_side:
            input_bgr = input_frames_bgr[idx]
            if output_masks[idx] is not None:
                mask_bgr = mask_to_bgr(output_masks[idx], target_size)
            else:
                # No mask — show a blank panel
                mask_bgr = np.zeros_like(input_bgr)
            frame_out = np.concatenate([input_bgr, mask_bgr, restored_bgr], axis=1)
        else:
            frame_out = restored_bgr

        out.write(frame_out)

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
    parser.add_argument("--max-size", type=int, default=512, help="Max processing dimension")
    parser.add_argument("--side-by-side", action="store_true",
                        help="Output side-by-side video: input | predicted mask | output")
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
        max_size=args.max_size,
        side_by_side=args.side_by_side,
    )


if __name__ == "__main__":
    main()
