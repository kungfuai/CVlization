"""
Dataset for Video Artifact Removal

Creates synthetic training data by adding visual artifacts to clean videos.
Supports overlay artifacts (logos, text) and degradation artifacts
(compression, noise, blur).
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import random
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("OpenCV not available, using PIL for video loading")

from PIL import Image
import os

from visual_artifacts import ArtifactGenerator, apply_overlay_artifact


class VideoFrameDataset(Dataset):
    """
    Dataset that loads video frames and adds synthetic artifacts.

    For prototyping, can also work with image folders.
    """

    def __init__(
        self,
        data_path: str,
        frame_size: Tuple[int, int] = (256, 256),
        num_frames: int = 5,
        min_opacity: float = 0.2,
        max_opacity: float = 0.8,
        mode: str = "train",  # "train", "val", "test"
        max_samples: Optional[int] = None,
        enabled_artifacts: Optional[set] = None,
    ):
        """
        Args:
            data_path: Path to video files or image folder
            frame_size: (H, W) output frame size
            num_frames: Number of consecutive frames to load
            min_opacity: Minimum overlay artifact opacity
            max_opacity: Maximum overlay artifact opacity
            mode: Dataset mode (affects augmentation)
            max_samples: Limit number of samples (for quick testing)
            enabled_artifacts: Set of artifact types to use (see ArtifactGenerator.ARTIFACT_TYPES)
        """
        self.data_path = Path(data_path)
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.mode = mode

        # Initialize artifact generator
        self.artifact_gen = ArtifactGenerator(
            frame_size=frame_size,
            min_opacity=min_opacity,
            max_opacity=max_opacity,
            enabled_artifacts=enabled_artifacts,
        )
        
        # Find all data sources
        self.samples = self._find_samples()
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        # Transforms
        self.transform = T.Compose([
            T.Resize(frame_size),
            T.ToTensor(),
        ])
        
        # Augmentations (only for training)
        self.augment = mode == "train"
    
    def _find_samples(self) -> List[Dict[str, Any]]:
        """Find all video files or image sequences"""
        samples = []
        
        if not self.data_path.exists():
            print(f"Warning: Data path {self.data_path} does not exist")
            return samples
        
        # Check for video files
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        for ext in video_extensions:
            for video_path in self.data_path.glob(f"**/*{ext}"):
                if HAS_CV2:
                    cap = cv2.VideoCapture(str(video_path))
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                    # Create multiple samples from each video
                    for start_frame in range(0, max(1, frame_count - self.num_frames), self.num_frames):
                        samples.append({
                            "type": "video",
                            "path": video_path,
                            "start_frame": start_frame,
                        })
        
        # Check for image folders
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        for folder in self.data_path.iterdir():
            if folder.is_dir():
                images = sorted([
                    f for f in folder.iterdir()
                    if f.suffix.lower() in image_extensions
                ])
                if len(images) >= self.num_frames:
                    for start_idx in range(0, len(images) - self.num_frames + 1, self.num_frames):
                        samples.append({
                            "type": "image_folder",
                            "images": images[start_idx:start_idx + self.num_frames],
                        })
        
        # Also check for loose images (treat as independent frames)
        loose_images = sorted([
            f for f in self.data_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ])
        if loose_images:
            for i in range(0, len(loose_images) - self.num_frames + 1, self.num_frames):
                samples.append({
                    "type": "image_folder",
                    "images": loose_images[i:i + self.num_frames],
                })
        
        print(f"Found {len(samples)} samples in {self.data_path}")
        return samples
    
    def _load_video_frames(self, video_path: Path, start_frame: int) -> List[Image.Image]:
        """Load frames from video file"""
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        
        cap.release()
        
        # Pad if needed
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else Image.new('RGB', self.frame_size))
        
        return frames
    
    def _load_image_frames(self, image_paths: List[Path]) -> List[Image.Image]:
        """Load frames from image files"""
        frames = []
        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB')
                frames.append(img)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(Image.new('RGB', self.frame_size))
        
        return frames
    
    def _augment_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to frames"""
        # frames: [T, C, H, W]
        
        # Random horizontal flip
        if random.random() < 0.5:
            frames = torch.flip(frames, dims=[-1])
        
        # Random vertical flip
        if random.random() < 0.3:
            frames = torch.flip(frames, dims=[-2])
        
        # Random brightness/contrast adjustment
        if random.random() < 0.5:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            frames = torch.clamp(frames * contrast + (brightness - 1), 0, 1)
        
        return frames
    
    def __len__(self) -> int:
        return max(1, len(self.samples))  # At least 1 for synthetic data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with:
                - clean: [T, C, H, W] clean frames
                - degraded: [T, C, H, W] degraded frames (with artifacts)
                - mask: [T, 1, H, W] artifact mask (for overlay types, zeros for degradation types)
        """
        if len(self.samples) == 0:
            # Generate synthetic data if no real data
            return self._generate_synthetic_sample()

        sample = self.samples[idx % len(self.samples)]

        # Load frames
        if sample["type"] == "video" and HAS_CV2:
            frames = self._load_video_frames(sample["path"], sample["start_frame"])
        else:
            frames = self._load_image_frames(sample["images"])

        # Transform to tensors
        frames = torch.stack([self.transform(f) for f in frames])  # [T, C, H, W]

        # Augment
        if self.augment:
            frames = self._augment_frames(frames)

        # Generate artifact
        mask, degraded_direct, meta = self.artifact_gen.generate(
            self.num_frames, clean_frames=frames
        )

        # Apply artifact based on type
        if mask is not None:
            # Overlay artifact - apply mask to frames
            degraded = apply_overlay_artifact(frames, mask)
        else:
            # Degradation artifact - frames already degraded
            degraded = degraded_direct
            # Create zero mask for consistency
            mask = torch.zeros(self.num_frames, 1, *self.frame_size)

        return {
            "clean": frames,
            "degraded": degraded,
            "mask": mask,
        }
    
    def _generate_synthetic_sample(self) -> Dict[str, torch.Tensor]:
        """Generate fully synthetic sample for testing"""
        # Create random gradient/color images
        frames = []
        base_color = torch.rand(3) * 0.5 + 0.25

        for t in range(self.num_frames):
            # Gradient with slight temporal variation
            H, W = self.frame_size
            y_grad = torch.linspace(0, 1, H).view(H, 1, 1).expand(H, W, 3)
            x_grad = torch.linspace(0, 1, W).view(1, W, 1).expand(H, W, 3)

            noise = torch.rand(H, W, 3) * 0.1
            temporal_shift = t / self.num_frames * 0.1

            frame = base_color + y_grad * 0.3 + x_grad * 0.2 + noise + temporal_shift
            frame = frame.clamp(0, 1).permute(2, 0, 1)  # [C, H, W]
            frames.append(frame)

        frames = torch.stack(frames)  # [T, C, H, W]

        # Generate artifact
        mask, degraded_direct, meta = self.artifact_gen.generate(
            self.num_frames, clean_frames=frames
        )

        if mask is not None:
            degraded = apply_overlay_artifact(frames, mask)
        else:
            degraded = degraded_direct
            mask = torch.zeros(self.num_frames, 1, *self.frame_size)

        return {
            "clean": frames,
            "degraded": degraded,
            "mask": mask,
        }


class DummyDataset(Dataset):
    """
    Dummy dataset for quick testing without real data.
    Generates random images with synthetic artifacts.
    """

    def __init__(
        self,
        num_samples: int = 100,
        frame_size: Tuple[int, int] = (256, 256),
        num_frames: int = 5,
        enabled_artifacts: Optional[set] = None,
    ):
        self.num_samples = num_samples
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.artifact_gen = ArtifactGenerator(
            frame_size=frame_size,
            enabled_artifacts=enabled_artifacts,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Random clean frames
        H, W = self.frame_size

        # Create somewhat realistic looking frames with gradients and noise
        torch.manual_seed(idx)  # Reproducible

        base = torch.rand(3).view(3, 1, 1) * 0.5 + 0.25
        frames = []

        for t in range(self.num_frames):
            # Base gradient
            gradient = torch.zeros(3, H, W)
            for c in range(3):
                gradient[c] = torch.linspace(0, 1, H).view(H, 1) * 0.3

            # Add noise and temporal variation
            noise = torch.rand(3, H, W) * 0.15
            frame = base + gradient + noise + t * 0.02
            frame = frame.clamp(0, 1)
            frames.append(frame)

        frames = torch.stack(frames)  # [T, C, H, W]

        # Generate artifact
        mask, degraded_direct, meta = self.artifact_gen.generate(
            self.num_frames, clean_frames=frames
        )

        if mask is not None:
            degraded = apply_overlay_artifact(frames, mask)
        else:
            degraded = degraded_direct
            mask = torch.zeros(self.num_frames, 1, *self.frame_size)

        return {
            "clean": frames,
            "degraded": degraded,
            "mask": mask,
        }


def get_dataloaders(
    data_path: str = "./data",
    batch_size: int = 4,
    frame_size: Tuple[int, int] = (256, 256),
    num_frames: int = 5,
    num_workers: int = 4,
    use_dummy: bool = False,
    enabled_artifacts: Optional[set] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_path: Path to data directory
        batch_size: Batch size
        frame_size: (H, W) frame size
        num_frames: Frames per sample
        num_workers: DataLoader workers
        use_dummy: Use dummy data for testing
        enabled_artifacts: Set of artifact types to use (None = overlay types only)

    Returns:
        train_loader, val_loader
    """
    if use_dummy:
        train_dataset = DummyDataset(
            num_samples=500,
            frame_size=frame_size,
            num_frames=num_frames,
            enabled_artifacts=enabled_artifacts,
        )
        val_dataset = DummyDataset(
            num_samples=50,
            frame_size=frame_size,
            num_frames=num_frames,
            enabled_artifacts=enabled_artifacts,
        )
    else:
        train_dataset = VideoFrameDataset(
            data_path=f"{data_path}/train",
            frame_size=frame_size,
            num_frames=num_frames,
            mode="train",
            enabled_artifacts=enabled_artifacts,
        )
        val_dataset = VideoFrameDataset(
            data_path=f"{data_path}/val",
            frame_size=frame_size,
            num_frames=num_frames,
            mode="val",
            enabled_artifacts=enabled_artifacts,
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


class VimeoArtifactDataset(Dataset):
    """
    Vimeo Septuplet dataset with synthetic artifact generation.

    Wraps VimeoSeptupletDataset and adds artifacts to create paired training data.
    """

    def __init__(
        self,
        vimeo_dataset,
        frame_size: Tuple[int, int] = (256, 256),
        num_frames: int = 5,
        enabled_artifacts: Optional[set] = None,
        min_opacity: float = 0.2,
        max_opacity: float = 0.8,
    ):
        """
        Args:
            vimeo_dataset: VimeoSeptupletDataset instance
            frame_size: Output frame size (H, W)
            num_frames: Number of frames to use (1-7)
            enabled_artifacts: Set of artifact types to use
            min_opacity: Min opacity for overlay artifacts
            max_opacity: Max opacity for overlay artifacts
        """
        self.vimeo_dataset = vimeo_dataset
        self.frame_size = frame_size
        self.num_frames = min(num_frames, 7)

        self.artifact_gen = ArtifactGenerator(
            frame_size=frame_size,
            min_opacity=min_opacity,
            max_opacity=max_opacity,
            enabled_artifacts=enabled_artifacts,
        )

        self.resize = T.Resize(frame_size)

    def __len__(self) -> int:
        return len(self.vimeo_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get clean frames from Vimeo
        sample = self.vimeo_dataset[idx]
        frames = sample["frames"]  # [T, C, H, W]

        # Take only num_frames
        frames = frames[:self.num_frames]

        # Resize to target size
        resized = []
        for i in range(frames.shape[0]):
            resized.append(self.resize(frames[i]))
        frames = torch.stack(resized)

        # Generate artifact
        mask, degraded_direct, meta = self.artifact_gen.generate(
            self.num_frames, clean_frames=frames
        )

        if mask is not None:
            degraded = apply_overlay_artifact(frames, mask)
        else:
            degraded = degraded_direct
            mask = torch.zeros(self.num_frames, 1, *self.frame_size)

        return {
            "clean": frames,
            "degraded": degraded,
            "mask": mask,
        }


def get_vimeo_dataloaders(
    batch_size: int = 4,
    frame_size: Tuple[int, int] = (256, 256),
    num_frames: int = 5,
    num_workers: int = 4,
    enabled_artifacts: Optional[set] = None,
    data_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders using Vimeo Septuplet.

    Args:
        batch_size: Batch size
        frame_size: (H, W) frame size
        num_frames: Frames per sample (1-7)
        num_workers: DataLoader workers
        enabled_artifacts: Set of artifact types to use
        data_dir: Override data directory

    Returns:
        train_loader, val_loader
    """
    from vimeo_septuplet import VimeoSeptupletBuilder

    # Build Vimeo dataset
    kwargs = {}
    if data_dir:
        kwargs["data_dir"] = data_dir

    builder = VimeoSeptupletBuilder(num_frames=num_frames, **kwargs)

    # Ensure dataset is ready
    if not builder.is_extracted():
        raise RuntimeError(
            "Vimeo dataset not ready. Run prepare_data.sh first, or:\n"
            "  python vimeo_septuplet.py --prepare"
        )

    # Create artifact-augmented datasets
    train_dataset = VimeoArtifactDataset(
        vimeo_dataset=builder.training_dataset(),
        frame_size=frame_size,
        num_frames=num_frames,
        enabled_artifacts=enabled_artifacts,
    )

    val_dataset = VimeoArtifactDataset(
        vimeo_dataset=builder.validation_dataset(),
        frame_size=frame_size,
        num_frames=num_frames,
        enabled_artifacts=enabled_artifacts,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# Test
if __name__ == "__main__":
    # Test dummy dataset with overlay artifacts (default)
    print("Testing DummyDataset with overlay artifacts...")
    dataset = DummyDataset(num_samples=10, frame_size=(256, 256), num_frames=5)
    sample = dataset[0]
    print(f"  Clean shape: {sample['clean'].shape}")
    print(f"  Degraded shape: {sample['degraded'].shape}")
    print(f"  Mask shape: {sample['mask'].shape}")

    # Test with degradation artifacts
    print("\nTesting DummyDataset with degradation artifacts...")
    dataset_deg = DummyDataset(
        num_samples=10,
        frame_size=(256, 256),
        num_frames=5,
        enabled_artifacts={"gaussian_noise", "blur", "color_banding"}
    )
    sample = dataset_deg[0]
    print(f"  Clean shape: {sample['clean'].shape}")
    print(f"  Degraded shape: {sample['degraded'].shape}")
    print(f"  Mask shape: {sample['mask'].shape}")

    # Test dataloader
    print("\nTesting DataLoader...")
    train_loader, val_loader = get_dataloaders(
        batch_size=2,
        use_dummy=True,
        num_workers=0,
    )

    for batch in train_loader:
        print(f"  Batch clean shape: {batch['clean'].shape}")
        print(f"  Batch degraded shape: {batch['degraded'].shape}")
        break
