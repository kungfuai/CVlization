#!/usr/bin/env python3
"""
Long Video Inference Script

Two inference modes:
1. Overlapping clips (--mode overlap): Process all frames with temporal overlap
2. Skip + warp (--mode skip): Process keyframes, warp mask/inpainted for skipped frames

Usage:
    # Overlapping clips (safest, highest quality)
    python inference_video.py -i input.mp4 -o output.mp4 -c checkpoint.pt --mode overlap

    # Skip + warp (faster, for static/slow scenes)
    python inference_video.py -i input.mp4 -o output.mp4 -c checkpoint.pt --mode skip --skip-every 4
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import cv2
from tqdm import tqdm

from model import TemporalNAFUNet, ExplicitCompositeNet
from model_lama import LamaWithMask
from model_elir import ElirWithMask


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Detect model type from config
    config = checkpoint.get("config", {})
    model_type = config.get("model_type", "temporal_nafunet")

    # Get model config
    model_config = config.get("model", {})
    encoder_channels = model_config.get("encoder_channels", [32, 64, 128, 256])
    num_frames = model_config.get("num_frames", 2)

    # Create model
    if model_type == "elir":
        model = ElirWithMask(
            in_channels=3,
            out_channels=3,
            latent_channels=16,
            hidden_channels=encoder_channels[0] if encoder_channels else 64,
            flow_hidden_channels=(encoder_channels[0] if encoder_channels else 64) * 2,
            k_steps=3,
            use_mask_unet=config.get("use_mask_unet", False),
        )
    elif model_type == "lama":
        model = LamaWithMask(
            in_channels=3,
            out_channels=3,
            base_channels=encoder_channels[0] if encoder_channels else 64,
        )
    elif model_type == "composite":
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
            predict_mask=True,
            mask_guidance=config.get("mask_guidance", "none"),
        )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    print(f"Loaded {model_type} model from {checkpoint_path}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, num_frames


class VideoReader:
    """Simple video reader using OpenCV."""

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return self.total_frames

    def close(self):
        self.cap.release()


class VideoWriter:
    """Simple video writer using OpenCV."""

    def __init__(self, path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    def write(self, frame: np.ndarray):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(frame_bgr)

    def close(self):
        self.writer.release()


def frame_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert numpy frame [H, W, 3] uint8 to tensor [1, 3, H, W] float."""
    tensor = torch.from_numpy(frame).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def tensor_to_frame(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor [1, 3, H, W] or [3, H, W] float to numpy frame [H, W, 3] uint8."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    frame = tensor.permute(1, 2, 0).cpu().numpy()
    frame = (frame * 255).clip(0, 255).astype(np.uint8)
    return frame


def compute_optical_flow(prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
    """Compute dense optical flow using Farneback method."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    return flow


def warp_with_flow(image: torch.Tensor, flow: np.ndarray) -> torch.Tensor:
    """Warp image tensor using optical flow."""
    device = image.device
    B, C, H, W = image.shape

    # Create sampling grid
    flow_tensor = torch.from_numpy(flow).float().to(device)

    # Normalize flow to [-1, 1]
    flow_tensor[..., 0] = flow_tensor[..., 0] / (W / 2)
    flow_tensor[..., 1] = flow_tensor[..., 1] / (H / 2)

    # Create base grid
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    grid = torch.stack([x, y], dim=-1).unsqueeze(0)

    # Add flow to grid
    grid = grid + flow_tensor.unsqueeze(0)

    # Warp
    warped = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped


def compute_warp_confidence(prev_frame: np.ndarray, curr_frame: np.ndarray, flow: np.ndarray) -> torch.Tensor:
    """Compute confidence mask for warping (based on forward-backward consistency)."""
    # Warp prev to curr
    prev_tensor = frame_to_tensor(prev_frame, torch.device('cpu'))
    warped = warp_with_flow(prev_tensor, flow)

    # Compute error
    curr_tensor = frame_to_tensor(curr_frame, torch.device('cpu'))
    error = (warped - curr_tensor).abs().mean(dim=1, keepdim=True)

    # Convert error to confidence (low error = high confidence)
    confidence = torch.exp(-error * 10)  # Exponential falloff
    return confidence


@torch.inference_mode()
def inference_overlap(
    model: torch.nn.Module,
    video_path: str,
    output_path: str,
    device: torch.device,
    clip_length: int = 8,
    overlap: int = 4,
) -> Dict[str, float]:
    """
    Inference with overlapping clips (Mode 1).

    Process all frames with temporal overlap for consistency.
    Highest quality, processes every frame.
    """
    reader = VideoReader(video_path)
    writer = VideoWriter(output_path, reader.fps, reader.width, reader.height)

    stride = clip_length - overlap
    frame_buffer = []
    output_buffer = []
    frames_processed = 0

    print(f"Mode: Overlapping clips (clip={clip_length}, overlap={overlap}, stride={stride})")
    print(f"Video: {reader.total_frames} frames, {reader.fps:.1f} fps, {reader.width}x{reader.height}")

    pbar = tqdm(total=reader.total_frames, desc="Processing")

    for frame in reader:
        frame_buffer.append(frame)

        # Process when we have enough frames
        if len(frame_buffer) >= clip_length:
            # Stack frames to tensor [1, T, C, H, W]
            clip_tensors = [frame_to_tensor(f, device) for f in frame_buffer[:clip_length]]
            clip = torch.cat(clip_tensors, dim=0).unsqueeze(0)  # [1, T, C, H, W]
            clip = clip.squeeze(2)  # [1, T, C, H, W] - remove extra dim if present

            # Handle different tensor shapes
            if clip.dim() == 4:  # [T, C, H, W]
                clip = clip.unsqueeze(0)  # [1, T, C, H, W]

            # Forward pass
            result = model(clip)

            # Extract output (handle different return types)
            if isinstance(result, dict):
                output_clip = result['output']
            elif isinstance(result, tuple):
                output_clip = result[0]
            else:
                output_clip = result

            # output_clip: [1, T, C, H, W] or [1, C, H, W]
            if output_clip.dim() == 4:
                output_clip = output_clip.unsqueeze(1)

            # Write non-overlapping frames
            start_idx = 0 if not output_buffer else overlap
            for i in range(start_idx, clip_length):
                if i < output_clip.shape[1]:
                    out_frame = tensor_to_frame(output_clip[0, i])
                    writer.write(out_frame)
                    frames_processed += 1
                    pbar.update(1)

            # Slide buffer
            frame_buffer = frame_buffer[stride:]

    # Process remaining frames
    if frame_buffer:
        # Pad to clip_length if needed
        while len(frame_buffer) < clip_length:
            frame_buffer.append(frame_buffer[-1])

        clip_tensors = [frame_to_tensor(f, device) for f in frame_buffer[:clip_length]]
        clip = torch.cat(clip_tensors, dim=0).unsqueeze(0)

        if clip.dim() == 4:
            clip = clip.unsqueeze(0)

        result = model(clip)

        if isinstance(result, dict):
            output_clip = result['output']
        elif isinstance(result, tuple):
            output_clip = result[0]
        else:
            output_clip = result

        if output_clip.dim() == 4:
            output_clip = output_clip.unsqueeze(1)

        # Write remaining frames
        remaining = reader.total_frames - frames_processed
        start_idx = overlap if frames_processed > 0 else 0
        for i in range(start_idx, start_idx + remaining):
            if i < output_clip.shape[1]:
                out_frame = tensor_to_frame(output_clip[0, i])
                writer.write(out_frame)
                pbar.update(1)

    pbar.close()
    reader.close()
    writer.close()

    return {"frames_processed": reader.total_frames, "forward_passes": reader.total_frames // stride + 1}


@torch.inference_mode()
def inference_skip_warp(
    model: torch.nn.Module,
    video_path: str,
    output_path: str,
    device: torch.device,
    clip_length: int = 8,
    skip_every: int = 4,
    similarity_threshold: float = 0.02,
    confidence_threshold: float = 0.7,
) -> Dict[str, float]:
    """
    Inference with frame skipping and optical flow warping (Mode 2).

    Process keyframes with full model, warp mask/inpainted for skipped frames.
    Faster for static/slow scenes, with low-risk warping strategy.
    """
    reader = VideoReader(video_path)
    writer = VideoWriter(output_path, reader.fps, reader.width, reader.height)

    print(f"Mode: Skip + Warp (skip_every={skip_every}, threshold={similarity_threshold})")
    print(f"Video: {reader.total_frames} frames, {reader.fps:.1f} fps, {reader.width}x{reader.height}")

    # State for warping
    anchor_frame = None
    anchor_output = None
    anchor_mask = None
    anchor_inpainted = None
    frames_since_anchor = 0

    forward_passes = 0
    skipped_frames = 0

    pbar = tqdm(total=reader.total_frames, desc="Processing")

    for i, frame in enumerate(reader):
        frame_tensor = frame_to_tensor(frame, device)
        should_process = False

        # Decide whether to process this frame
        if anchor_frame is None:
            should_process = True
        elif frames_since_anchor >= skip_every:
            should_process = True
        else:
            # Check similarity
            anchor_tensor = frame_to_tensor(anchor_frame, device)
            diff = (frame_tensor - anchor_tensor).abs().mean().item()
            if diff > similarity_threshold:
                should_process = True

        if should_process:
            # Full forward pass
            # For single frame models, process as [1, C, H, W]
            # For temporal models, create a mini-clip
            if clip_length > 1 and anchor_frame is not None:
                # Create a 2-frame clip for temporal context
                prev_tensor = frame_to_tensor(anchor_frame, device)
                clip = torch.stack([prev_tensor.squeeze(0), frame_tensor.squeeze(0)], dim=0).unsqueeze(0)
            else:
                clip = frame_tensor

            result = model(clip)

            # Extract outputs
            if isinstance(result, dict):
                output = result['output']
                mask = result.get('pred_mask')
                inpainted = result.get('restored', output)
            elif isinstance(result, tuple):
                output = result[0]
                mask = result[1] if len(result) > 1 else None
                inpainted = output
            else:
                output = result
                mask = None
                inpainted = output

            # Get last frame if temporal
            if output.dim() == 5:  # [B, T, C, H, W]
                output = output[:, -1]
                if mask is not None and mask.dim() == 5:
                    mask = mask[:, -1]
                if inpainted.dim() == 5:
                    inpainted = inpainted[:, -1]

            # Update anchor
            anchor_frame = frame.copy()
            anchor_output = output
            anchor_mask = mask
            anchor_inpainted = inpainted
            frames_since_anchor = 0
            forward_passes += 1

            out_frame = tensor_to_frame(output)

        else:
            # Warp from anchor (low-risk strategy)
            flow = compute_optical_flow(anchor_frame, frame)
            confidence = compute_warp_confidence(anchor_frame, frame, flow)
            confidence = confidence.to(device)

            # Warp mask (always needed)
            if anchor_mask is not None:
                warped_mask = warp_with_flow(anchor_mask, flow)
            else:
                warped_mask = torch.zeros(1, 1, frame.shape[0], frame.shape[1], device=device)

            # Warp inpainted region
            warped_inpainted = warp_with_flow(anchor_inpainted, flow)

            # Low-risk composite: use CURRENT frame for background
            # Only use warped result in mask region where confidence is high
            safe_mask = warped_mask * (confidence > confidence_threshold).float()

            output = frame_tensor * (1 - safe_mask) + warped_inpainted * safe_mask

            frames_since_anchor += 1
            skipped_frames += 1

            out_frame = tensor_to_frame(output)

        writer.write(out_frame)
        pbar.update(1)

    pbar.close()
    reader.close()
    writer.close()

    speedup = reader.total_frames / forward_passes if forward_passes > 0 else 1.0
    print(f"\nStats: {forward_passes} forward passes, {skipped_frames} skipped ({speedup:.1f}x speedup)")

    return {
        "frames_total": reader.total_frames,
        "forward_passes": forward_passes,
        "skipped_frames": skipped_frames,
        "speedup": speedup,
    }


def main():
    parser = argparse.ArgumentParser(description="Long Video Inference")
    parser.add_argument("-i", "--input", required=True, help="Input video path")
    parser.add_argument("-o", "--output", required=True, help="Output video path")
    parser.add_argument("-c", "--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--mode", choices=["overlap", "skip"], default="overlap",
                        help="Inference mode: 'overlap' for quality, 'skip' for speed")
    parser.add_argument("--clip-length", type=int, default=8, help="Frames per clip")
    parser.add_argument("--overlap", type=int, default=4, help="Overlap frames (for overlap mode)")
    parser.add_argument("--skip-every", type=int, default=4, help="Process every N frames (for skip mode)")
    parser.add_argument("--similarity-threshold", type=float, default=0.02,
                        help="Frame difference threshold for skipping")
    parser.add_argument("--confidence-threshold", type=float, default=0.7,
                        help="Warp confidence threshold for using warped result")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")

    args = parser.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load model
    model, trained_num_frames = load_model(args.checkpoint, device)

    # Adjust clip length to match training
    clip_length = args.clip_length
    if clip_length != trained_num_frames:
        print(f"Warning: Model trained with {trained_num_frames} frames, using {clip_length}")

    # Run inference
    if args.mode == "overlap":
        stats = inference_overlap(
            model=model,
            video_path=args.input,
            output_path=args.output,
            device=device,
            clip_length=clip_length,
            overlap=args.overlap,
        )
    else:
        stats = inference_skip_warp(
            model=model,
            video_path=args.input,
            output_path=args.output,
            device=device,
            clip_length=clip_length,
            skip_every=args.skip_every,
            similarity_threshold=args.similarity_threshold,
            confidence_threshold=args.confidence_threshold,
        )

    print(f"\nDone! Output saved to: {args.output}")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
