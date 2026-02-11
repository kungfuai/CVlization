"""
Pexels Animals Dataset Builder

Downloads animal videos from HuggingFace (zzsi/animals_pexels) and provides
a PyTorch Dataset for video enhancement training.

All videos are public domain from Pexels.com.

Usage:
    builder = PexelsAnimalsBuilder()
    builder.prepare()  # Downloads videos from HuggingFace if needed

    train_ds = builder.training_dataset(num_frames=5, frame_size=(256, 256))
    val_ds = builder.validation_dataset(num_frames=5, frame_size=(256, 256))
"""
import os
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from PIL import Image

# Dataset constants
HF_REPO_ID = "zzsi/animals_pexels"
DATASET_NAME = "pexels_animals"
VAL_FRACTION = 0.1  # 10% of videos for validation


def get_default_data_dir() -> Path:
    """Get default data directory: ~/.cache/cvlization/data/"""
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_dir = Path(cache_home) / "cvlization" / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class PexelsAnimalsDataset(Dataset):
    """
    PyTorch Dataset for Pexels animal videos.

    Each sample is a sequence of consecutive frames from a video clip.
    """

    def __init__(
        self,
        video_paths: List[Path],
        num_frames: int = 5,
        frame_size: Optional[Tuple[int, int]] = None,
        transform=None,
    ):
        """
        Args:
            video_paths: List of paths to .mp4 video files
            num_frames: Number of consecutive frames per sample
            frame_size: Optional (H, W) to resize frames
            transform: Optional transform applied to each PIL frame
        """
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform

        # Build video index: list of (video_path, total_frames)
        # Start position is sampled randomly in __getitem__
        self.videos = []
        for vp in video_paths:
            n = self._count_frames(vp)
            if n >= num_frames:
                self.videos.append((vp, n))

    @staticmethod
    def _count_frames(video_path: Path) -> int:
        if not HAS_CV2:
            return 0
        cap = cv2.VideoCapture(str(video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with:
                - frames: [T, C, H, W] float tensor in [0, 1]
                - video_id: str filename of the source video
        """
        video_path, total_frames = self.videos[idx]
        start = random.randint(0, total_frames - self.num_frames)

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        frames = []
        for _ in range(self.num_frames):
            ret, bgr = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            if self.transform:
                img = self.transform(img)
            else:
                if self.frame_size:
                    img = img.resize((self.frame_size[1], self.frame_size[0]), Image.LANCZOS)
                img = torch.from_numpy(np.array(img)).float() / 255.0
                img = img.permute(2, 0, 1)  # HWC -> CHW

            frames.append(img)
        cap.release()

        # Pad if video ended early
        while len(frames) < self.num_frames:
            frames.append(frames[-1].clone())

        return {
            "frames": torch.stack(frames),
            "video_id": video_path.stem,
        }


@dataclass
class PexelsAnimalsBuilder:
    """
    Dataset builder for Pexels Animals videos.

    Downloads from HuggingFace (zzsi/animals_pexels) on first use,
    then provides train/val splits.

    Usage:
        builder = PexelsAnimalsBuilder()
        builder.prepare()

        train_ds = builder.training_dataset(num_frames=5, frame_size=(256, 256))
        val_ds = builder.validation_dataset(num_frames=5, frame_size=(256, 256))
    """

    data_dir: str = field(default_factory=lambda: str(get_default_data_dir()))
    val_fraction: float = VAL_FRACTION
    seed: int = 42
    quiet: bool = False

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.dataset_dir = self.data_dir / DATASET_NAME

    @property
    def dataset_provider(self) -> str:
        return "cvlization"

    def is_ready(self) -> bool:
        """Check if videos are downloaded."""
        if not self.dataset_dir.exists():
            return False
        return len(list(self.dataset_dir.glob("*.mp4"))) > 0

    def prepare(self) -> None:
        """Download videos from HuggingFace if needed."""
        if self.is_ready():
            if not self.quiet:
                print(f"Dataset ready at {self.dataset_dir}")
            return

        if not self.quiet:
            print(f"Downloading {HF_REPO_ID} from HuggingFace...")

        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=str(self.dataset_dir),
            allow_patterns="*.mp4",
        )

        if not self.quiet:
            n = len(list(self.dataset_dir.glob("*.mp4")))
            print(f"Downloaded {n} videos to {self.dataset_dir}")

    def _split_videos(self) -> Tuple[List[Path], List[Path]]:
        """Deterministic train/val split of video files."""
        videos = sorted(self.dataset_dir.glob("*.mp4"))
        rng = random.Random(self.seed)
        rng.shuffle(videos)

        n_val = max(1, int(len(videos) * self.val_fraction))
        return videos[n_val:], videos[:n_val]

    def get_stats(self) -> dict:
        if not self.is_ready():
            return {"status": "not_downloaded"}

        train_vids, val_vids = self._split_videos()
        return {
            "status": "ready",
            "train_videos": len(train_vids),
            "val_videos": len(val_vids),
            "total_videos": len(train_vids) + len(val_vids),
            "dataset_dir": str(self.dataset_dir),
        }

    def training_dataset(
        self,
        num_frames: int = 5,
        frame_size: Optional[Tuple[int, int]] = None,
        transform=None,
    ) -> PexelsAnimalsDataset:
        if not self.is_ready():
            raise RuntimeError("Dataset not ready. Run prepare() first.")
        train_vids, _ = self._split_videos()
        return PexelsAnimalsDataset(
            video_paths=train_vids,
            num_frames=num_frames,
            frame_size=frame_size,
            transform=transform,
        )

    def validation_dataset(
        self,
        num_frames: int = 5,
        frame_size: Optional[Tuple[int, int]] = None,
        transform=None,
    ) -> PexelsAnimalsDataset:
        if not self.is_ready():
            raise RuntimeError("Dataset not ready. Run prepare() first.")
        _, val_vids = self._split_videos()
        return PexelsAnimalsDataset(
            video_paths=val_vids,
            num_frames=num_frames,
            frame_size=frame_size,
            transform=transform,
        )


def prepare_dataset(data_dir: Optional[str] = None, quiet: bool = False) -> dict:
    """Convenience function to download and prepare the dataset."""
    kwargs = {"quiet": quiet}
    if data_dir:
        kwargs["data_dir"] = data_dir
    builder = PexelsAnimalsBuilder(**kwargs)
    builder.prepare()
    return builder.get_stats()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pexels Animals Dataset Manager")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--prepare", action="store_true", help="Download if needed")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    kwargs = {"quiet": args.quiet}
    if args.data_dir:
        kwargs["data_dir"] = args.data_dir

    builder = PexelsAnimalsBuilder(**kwargs)

    if args.prepare:
        builder.prepare()

    stats = builder.get_stats()
    print("\nPexels Animals Dataset Statistics:")
    print("-" * 40)
    for key, value in stats.items():
        print(f"  {key}: {value}")
