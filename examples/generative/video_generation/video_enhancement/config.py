"""
Configuration for Video Artifact Removal Training
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Input
    in_channels: int = 3
    out_channels: int = 3

    # Encoder channels
    encoder_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])

    # Temporal attention
    use_temporal_attention: bool = True
    num_frames: int = 5  # Number of frames to process together
    attention_heads: int = 4
    attention_dim: int = 128

    # Decoder
    use_skip_connections: bool = True

    # Learning mode
    residual_learning: bool = False  # If True, predict residual and add to input
    predict_mask: bool = False  # If True, also output artifact mask
    mask_guidance: str = "none"  # "none" or "modulation"


@dataclass
class DataConfig:
    """Dataset configuration"""
    # Video sources
    video_dir: str = "./data/clean_videos"

    # Frame settings
    frame_size: Tuple[int, int] = (256, 256)  # H, W - start small for prototyping
    num_frames: int = 5

    # Artifact generation - types to enable during training
    # Available overlay types: corner_logo, text_overlay, tiled_pattern,
    #                          moving_logo, channel_logo, diagonal_text
    # Available degradation types: jpeg_compression, video_compression,
    #                              gaussian_noise, salt_pepper_noise, film_grain,
    #                              color_banding, blur
    enabled_artifacts: Optional[List[str]] = None  # None = overlay types only (default)

    # Augmentation (for overlay artifacts)
    min_opacity: float = 0.2
    max_opacity: float = 0.8
    

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Basic
    batch_size: int = 4
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Loss weights
    w_pixel: float = 1.0
    w_perceptual: float = 0.1
    w_temporal: float = 0.5
    w_fft: float = 0.05
    w_mask: float = 0.3  # Mask prediction loss (only for overlay artifacts)
    
    # Scheduler
    use_scheduler: bool = True
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 5
    
    # Device
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    
    # Mixed precision
    use_amp: bool = False  # Automatic mixed precision (can cause inf loss with FFT)
    
    def get_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.device)


@dataclass
class Config:
    """Master configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Experiment
    experiment_name: str = "artifact_removal_v1"
    seed: int = 42


def get_config() -> Config:
    """Get default configuration"""
    return Config()
