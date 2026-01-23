"""
Inference Script for Video Artifact Removal

Optimized for Apple M4 Mac Mini deployment.

Usage:
    python predict.py --input video.mp4 --output clean.mp4 --checkpoint best.pt
    python predict.py --input frame.png --output clean.png --checkpoint best.pt
"""
import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
import time
from typing import Optional, Tuple, List
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from PIL import Image
import torchvision.transforms.functional as TF

from model import ArtifactRemovalNet, ArtifactRemovalNetLite


def load_model(
    checkpoint_path: str,
    device: torch.device,
    lite: bool = False,
) -> torch.nn.Module:
    """Load trained model from checkpoint"""

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    model_config = checkpoint.get("config", {}).get("model", {})

    if lite:
        model = ArtifactRemovalNetLite(
            residual_learning=model_config.get("residual_learning", False),
        )
    else:
        model = ArtifactRemovalNet(
            encoder_channels=model_config.get("encoder_channels", [32, 64, 128, 256]),
            use_temporal_attention=model_config.get("use_temporal_attention", True),
            num_frames=model_config.get("num_frames", 5),
            residual_learning=model_config.get("residual_learning", False),
            predict_mask=model_config.get("predict_mask", False),
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def get_device() -> torch.device:
    """Get best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess_frame(
    frame: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Preprocess frame for model input.
    
    Args:
        frame: BGR numpy array from cv2 or RGB from PIL
        target_size: Optional resize (H, W)
    
    Returns:
        tensor: [1, 3, H, W] normalized tensor
        original_size: (H, W) for later restoration
    """
    original_size = frame.shape[:2]
    
    # Convert BGR to RGB if needed (cv2 loads as BGR)
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        # Assume BGR from cv2
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame
    
    # Convert to tensor
    tensor = torch.from_numpy(frame_rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
    
    # Resize if needed
    if target_size:
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=target_size,
            mode="bilinear",
            align_corners=False
        ).squeeze(0)
    
    return tensor.unsqueeze(0), original_size  # [1, 3, H, W]


def postprocess_frame(
    tensor: torch.Tensor,
    original_size: Tuple[int, int],
) -> np.ndarray:
    """
    Convert model output back to image.
    
    Args:
        tensor: [1, 3, H, W] model output
        original_size: (H, W) original frame size
    
    Returns:
        BGR numpy array for cv2
    """
    # Resize back to original
    if tensor.shape[-2:] != original_size:
        tensor = F.interpolate(
            tensor,
            size=original_size,
            mode="bilinear",
            align_corners=False
        )
    
    # Convert to numpy
    frame = tensor.squeeze(0).clamp(0, 1)
    frame = frame.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
    frame = (frame * 255).astype(np.uint8)
    
    # RGB to BGR for cv2
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    return frame_bgr


@torch.no_grad()
def process_image(
    model: torch.nn.Module,
    image_path: str,
    output_path: str,
    device: torch.device,
    target_size: Optional[Tuple[int, int]] = None,
):
    """Process a single image"""
    print(f"Processing image: {image_path}")
    
    # Load image
    if HAS_CV2:
        frame = cv2.imread(image_path)
    else:
        frame = np.array(Image.open(image_path).convert("RGB"))
    
    # Preprocess
    tensor, original_size = preprocess_frame(frame, target_size)
    tensor = tensor.to(device)
    
    # Inference
    start = time.time()
    output = model(tensor)
    inference_time = time.time() - start
    
    print(f"  Inference time: {inference_time * 1000:.1f}ms")
    
    # Postprocess
    result = postprocess_frame(output, original_size)
    
    # Save
    cv2.imwrite(output_path, result)
    print(f"  Saved to: {output_path}")


@torch.no_grad()
def process_video(
    model: torch.nn.Module,
    video_path: str,
    output_path: str,
    device: torch.device,
    target_size: Optional[Tuple[int, int]] = None,
    batch_frames: int = 5,
    use_temporal: bool = True,
):
    """
    Process video file.
    
    Args:
        model: Trained model
        video_path: Input video path
        output_path: Output video path
        device: Compute device
        target_size: Optional resize for processing
        batch_frames: Number of frames to process together
        use_temporal: Use temporal processing (requires model with temporal attention)
    """
    if not HAS_CV2:
        raise RuntimeError("OpenCV required for video processing")
    
    print(f"Processing video: {video_path}")
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    
    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process in batches
    frame_buffer = []
    processed = 0
    total_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_buffer.append(frame)
        
        # Process when buffer is full or at end
        if len(frame_buffer) >= batch_frames:
            # Preprocess batch
            tensors = []
            original_sizes = []
            for f in frame_buffer:
                t, size = preprocess_frame(f, target_size)
                tensors.append(t)
                original_sizes.append(size)
            
            if use_temporal:
                # Stack as video batch [1, T, C, H, W]
                batch = torch.cat(tensors, dim=0).unsqueeze(0).to(device)
                
                start = time.time()
                outputs = model(batch)
                total_time += time.time() - start
                
                outputs = outputs.squeeze(0)  # [T, C, H, W]
            else:
                # Process frame by frame
                batch = torch.cat(tensors, dim=0).to(device)
                
                start = time.time()
                outputs = model(batch)
                total_time += time.time() - start
            
            # Postprocess and write
            for i, output in enumerate(outputs):
                result = postprocess_frame(
                    output.unsqueeze(0),
                    original_sizes[i]
                )
                out.write(result)
            
            processed += len(frame_buffer)
            
            # Progress
            if processed % 50 == 0:
                fps_actual = processed / total_time if total_time > 0 else 0
                print(f"\r  Processed {processed}/{total_frames} frames "
                      f"({fps_actual:.1f} fps)", end="", flush=True)
            
            frame_buffer = []
    
    # Process remaining frames
    if frame_buffer:
        tensors = []
        original_sizes = []
        for f in frame_buffer:
            t, size = preprocess_frame(f, target_size)
            tensors.append(t)
            original_sizes.append(size)
        
        batch = torch.cat(tensors, dim=0)
        if use_temporal:
            batch = batch.unsqueeze(0)
        batch = batch.to(device)
        
        start = time.time()
        outputs = model(batch)
        total_time += time.time() - start
        
        if use_temporal:
            outputs = outputs.squeeze(0)
        
        for i, output in enumerate(outputs):
            result = postprocess_frame(output.unsqueeze(0), original_sizes[i])
            out.write(result)
        
        processed += len(frame_buffer)
    
    cap.release()
    out.release()
    
    print()  # New line
    fps_avg = processed / total_time if total_time > 0 else 0
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Average FPS: {fps_avg:.1f}")
    print(f"  Saved to: {output_path}")


def benchmark(
    model: torch.nn.Module,
    device: torch.device,
    sizes: List[Tuple[int, int]] = [(256, 256), (512, 512), (720, 1280)],
    num_frames: int = 5,
    warmup: int = 3,
    iterations: int = 10,
):
    """Benchmark model performance"""
    print("\nBenchmarking...")
    print("-" * 50)
    
    model.eval()
    
    for H, W in sizes:
        # Single frame benchmark
        x = torch.randn(1, 3, H, W, device=device)
        
        # Warmup
        for _ in range(warmup):
            _ = model(x)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        
        # Time
        start = time.time()
        for _ in range(iterations):
            _ = model(x)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        
        elapsed = time.time() - start
        fps = iterations / elapsed
        
        print(f"Single frame {W}x{H}: {fps:.1f} FPS ({1000/fps:.1f}ms)")
        
        # Video benchmark
        x = torch.randn(1, num_frames, 3, H, W, device=device)
        
        for _ in range(warmup):
            _ = model(x)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        
        start = time.time()
        for _ in range(iterations):
            _ = model(x)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        
        elapsed = time.time() - start
        fps = iterations * num_frames / elapsed
        
        print(f"Video ({num_frames} frames) {W}x{H}: {fps:.1f} FPS ({1000*elapsed/iterations/num_frames:.1f}ms/frame)")
    
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Artifact removal inference")
    
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input image or video path")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output path")
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                        help="Model checkpoint path")
    
    # Processing options
    parser.add_argument("--size", type=int, nargs=2, default=None,
                        help="Processing size (H W), e.g., --size 256 256")
    parser.add_argument("--batch-frames", type=int, default=5,
                        help="Frames to process together for video")
    parser.add_argument("--no-temporal", action="store_true",
                        help="Disable temporal processing")
    parser.add_argument("--lite", action="store_true",
                        help="Use lightweight model")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    
    # Benchmark mode
    parser.add_argument("--benchmark", action="store_true",
                        help="Run performance benchmark")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device, lite=args.lite)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    
    # Benchmark mode
    if args.benchmark:
        benchmark(model, device)
        return
    
    # Process input
    input_path = Path(args.input)
    target_size = tuple(args.size) if args.size else None
    
    if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        # Video processing
        process_video(
            model,
            str(input_path),
            args.output,
            device,
            target_size=target_size,
            batch_frames=args.batch_frames,
            use_temporal=not args.no_temporal,
        )
    else:
        # Image processing
        process_image(
            model,
            str(input_path),
            args.output,
            device,
            target_size=target_size,
        )


if __name__ == "__main__":
    main()
