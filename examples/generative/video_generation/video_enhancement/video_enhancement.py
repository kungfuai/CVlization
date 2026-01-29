"""
Video Enhancement - Simple API

Usage:
    from video_enhancement import VideoEnhancer

    enhancer = VideoEnhancer.from_pretrained()
    enhancer.process_video("input.mp4", "output.mp4")

    # Or process frames directly
    restored, mask = enhancer(frames)  # frames: [B, T, C, H, W]
"""
import torch
from pathlib import Path
from typing import Optional, Tuple, Union

try:
    from huggingface_hub import hf_hub_download
    HAS_HF = True
except ImportError:
    HAS_HF = False


HF_REPO = "zzsi/cvl_models"
MODELS = {
    "nafunet": "video_enhancement/nafunet.pt",
    "composite": "video_enhancement/composite.pt",
}


class VideoEnhancer:
    """High-level API for video enhancement."""

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "nafunet",
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
    ) -> "VideoEnhancer":
        """
        Load model from HuggingFace or local checkpoint.

        Args:
            model_name: "nafunet" or "composite"
            checkpoint: Local checkpoint path (overrides model_name)
            device: "cuda", "cpu", or "mps" (auto-detected if None)
        """
        from infer import load_model, get_model_path

        # Setup device
        if device:
            dev = torch.device(device)
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")

        # Load model
        model_path = get_model_path(model_name, checkpoint)
        model = load_model(model_path, dev)

        return cls(model, dev)

    def __call__(
        self,
        frames: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhance video frames.

        Args:
            frames: [B, T, C, H, W] or [T, C, H, W] tensor, values 0-1

        Returns:
            restored: Enhanced frames (same shape as input)
            mask: Predicted artifact mask
        """
        # Add batch dim if needed
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        frames = frames.to(self.device)

        with torch.no_grad():
            result = self.model(frames)

        if isinstance(result, tuple):
            restored, mask = result
        else:
            restored = result
            mask = None

        if squeeze_batch:
            restored = restored.squeeze(0)
            if mask is not None:
                mask = mask.squeeze(0)

        return restored, mask

    def process_video(
        self,
        input_path: str,
        output_path: str,
        clip_length: int = 4,
        overlap: int = 2,
        max_size: int = 512,
    ) -> None:
        """
        Process a video file.

        Args:
            input_path: Input video path (mp4, mov, etc.)
            output_path: Output video path
            clip_length: Frames per clip
            overlap: Overlap between clips
            max_size: Max processing resolution
        """
        from infer import process_video
        process_video(
            model=self.model,
            input_path=input_path,
            output_path=output_path,
            device=self.device,
            clip_length=clip_length,
            overlap=overlap,
            max_size=max_size,
        )
