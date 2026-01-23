"""
Vimeo Septuplet Dataset Builder

Downloads, extracts, and provides train/val splits for the Vimeo-90K Septuplet dataset.
Each sample contains 7 consecutive video frames.

Dataset URL: http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip
Project page: http://toflow.csail.mit.edu/index.html#septuplet

If you use this dataset, please cite:

    @article{xue2019video,
        title={Video Enhancement with Task-Oriented Flow},
        author={Xue, Tianfan and Chen, Baian and Wu, Jiajun and Wei, Donglai and Freeman, William T},
        journal={International Journal of Computer Vision (IJCV)},
        volume={127},
        number={8},
        pages={1106--1125},
        year={2019},
        publisher={Springer}
    }

Structure after extraction:
    vimeo_septuplet/
    ├── sequences/
    │   ├── 00001/
    │   │   ├── 0001/
    │   │   │   ├── im1.png
    │   │   │   ├── im2.png
    │   │   │   ├── ...
    │   │   │   └── im7.png
    │   │   └── .../
    │   └── .../
    ├── sep_trainlist.txt
    ├── sep_testlist.txt
    └── readme.txt
"""
import os
import zipfile
import requests
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


# Dataset constants
VIMEO_SEPTUPLET_URL = "http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip"
DATASET_NAME = "vimeo_septuplet"
ZIP_FILENAME = "vimeo_septuplet.zip"


def get_cache_dir() -> Path:
    """Get CVlization cache directory."""
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return Path(cache_home) / "cvlization"


def get_default_data_dir() -> Path:
    """Get default data directory: ~/.cache/cvlization/data/"""
    # In Docker, this will be /cvl-cache/data/ if XDG_CACHE_HOME is set
    cache_dir = get_cache_dir() / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_file(url: str, dest: Path, quiet: bool = False) -> bool:
    """
    Download file from URL with progress bar.

    Returns True if downloaded, False if already exists.
    """
    if dest.exists() and dest.stat().st_size > 0:
        if not quiet:
            print(f"File already exists: {dest}")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest.with_suffix(dest.suffix + ".tmp")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(temp_path, 'wb') as f:
            if quiet or total_size == 0:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            else:
                with tqdm(total=total_size, unit='B', unit_scale=True,
                          desc=f"Downloading {dest.name}") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

        # Atomic move
        temp_path.rename(dest)
        return True

    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}")


def extract_zip(zip_path: Path, extract_dir: Path, quiet: bool = False) -> None:
    """Extract ZIP file with progress bar."""
    if not quiet:
        print(f"Extracting {zip_path.name}...")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        if quiet:
            zf.extractall(extract_dir)
        else:
            for member in tqdm(members, desc="Extracting"):
                zf.extract(member, extract_dir)


class VimeoSeptupletDataset(Dataset):
    """
    PyTorch Dataset for Vimeo Septuplet.

    Each sample is a sequence of 7 consecutive frames from a video clip.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        num_frames: int = 7,
        frame_indices: Optional[List[int]] = None,
        transform=None,
    ):
        """
        Args:
            data_dir: Path to vimeo_septuplet directory
            split: "train" or "test"
            num_frames: Number of frames to load per sample (1-7)
            frame_indices: Specific frame indices to load (e.g., [0, 3, 6] for first, middle, last)
            transform: Optional transform to apply to frames
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_frames = num_frames
        self.frame_indices = frame_indices
        self.transform = transform

        # Validate
        if num_frames < 1 or num_frames > 7:
            raise ValueError(f"num_frames must be 1-7, got {num_frames}")

        if frame_indices is not None:
            if any(i < 0 or i > 6 for i in frame_indices):
                raise ValueError(f"frame_indices must be in range [0, 6], got {frame_indices}")

        # Load sequence list
        self.sequences = self._load_sequence_list()

    def _load_sequence_list(self) -> List[Tuple[str, str]]:
        """Load list of (folder, subfolder) tuples from split file."""
        if self.split == "train":
            list_file = self.data_dir / "sep_trainlist.txt"
        else:
            list_file = self.data_dir / "sep_testlist.txt"

        if not list_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {list_file}\n"
                f"Run VimeoSeptupletBuilder.prepare() first to download and extract the dataset."
            )

        sequences = []
        with open(list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('/')
                    if len(parts) == 2:
                        sequences.append((parts[0], parts[1]))

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with:
                - frames: [T, C, H, W] tensor of frames
                - sequence_id: str identifier for the sequence
        """
        folder, subfolder = self.sequences[idx]
        seq_dir = self.data_dir / "sequences" / folder / subfolder

        # Determine which frames to load
        if self.frame_indices is not None:
            indices = self.frame_indices
        else:
            # Load consecutive frames starting from frame 1
            indices = list(range(self.num_frames))

        # Load frames
        frames = []
        for i in indices:
            frame_path = seq_dir / f"im{i + 1}.png"  # im1.png through im7.png

            if not frame_path.exists():
                raise FileNotFoundError(f"Frame not found: {frame_path}")

            # Load image
            from PIL import Image
            img = Image.open(frame_path).convert('RGB')

            if self.transform:
                img = self.transform(img)
            else:
                # Default: convert to tensor [C, H, W] in range [0, 1]
                import numpy as np
                img = torch.from_numpy(np.array(img)).float() / 255.0
                img = img.permute(2, 0, 1)  # HWC -> CHW

            frames.append(img)

        frames = torch.stack(frames)  # [T, C, H, W]

        return {
            "frames": frames,
            "sequence_id": f"{folder}/{subfolder}",
        }


@dataclass
class VimeoSeptupletBuilder:
    """
    Dataset builder for Vimeo Septuplet.

    Handles downloading, extraction, and provides train/test splits.

    Usage:
        builder = VimeoSeptupletBuilder()
        builder.prepare()  # Downloads and extracts if needed

        train_ds = builder.training_dataset()
        val_ds = builder.validation_dataset()
    """

    data_dir: str = field(default_factory=lambda: str(get_default_data_dir()))
    num_frames: int = 7
    frame_indices: Optional[List[int]] = None
    transform: Optional[object] = None
    quiet: bool = False

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.dataset_dir = self.data_dir / DATASET_NAME
        self.zip_path = self.data_dir / ZIP_FILENAME

    @property
    def dataset_provider(self) -> str:
        return "cvlization"

    def is_downloaded(self) -> bool:
        """Check if ZIP file exists."""
        return self.zip_path.exists() and self.zip_path.stat().st_size > 0

    def is_extracted(self) -> bool:
        """Check if dataset is extracted."""
        return (
            (self.dataset_dir / "sequences").exists() and
            (self.dataset_dir / "sep_trainlist.txt").exists() and
            (self.dataset_dir / "sep_testlist.txt").exists()
        )

    def download(self) -> bool:
        """
        Download the dataset ZIP file.

        Returns True if downloaded, False if already exists.
        """
        if not self.quiet:
            print(f"Downloading Vimeo Septuplet to {self.zip_path}")
            print(f"URL: {VIMEO_SEPTUPLET_URL}")
            print("Note: This is a large file (~82GB), download may take a while...")

        return download_file(VIMEO_SEPTUPLET_URL, self.zip_path, quiet=self.quiet)

    def extract(self) -> None:
        """Extract the dataset from ZIP file."""
        if not self.is_downloaded():
            raise FileNotFoundError(
                f"ZIP file not found: {self.zip_path}\n"
                f"Run download() first."
            )

        if self.is_extracted():
            if not self.quiet:
                print(f"Dataset already extracted at {self.dataset_dir}")
            return

        extract_zip(self.zip_path, self.data_dir, quiet=self.quiet)

        if not self.quiet:
            print(f"Dataset extracted to {self.dataset_dir}")

    def prepare(self) -> None:
        """Download and extract if needed."""
        if not self.is_extracted():
            if not self.is_downloaded():
                self.download()
            self.extract()
        elif not self.quiet:
            print(f"Dataset ready at {self.dataset_dir}")

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        if not self.is_extracted():
            return {"status": "not_extracted"}

        train_list = self.dataset_dir / "sep_trainlist.txt"
        test_list = self.dataset_dir / "sep_testlist.txt"

        train_count = sum(1 for line in open(train_list) if line.strip())
        test_count = sum(1 for line in open(test_list) if line.strip())

        return {
            "status": "ready",
            "train_sequences": train_count,
            "test_sequences": test_count,
            "total_sequences": train_count + test_count,
            "frames_per_sequence": 7,
            "total_frames": (train_count + test_count) * 7,
            "dataset_dir": str(self.dataset_dir),
        }

    def training_dataset(self) -> VimeoSeptupletDataset:
        """Get training dataset."""
        if not self.is_extracted():
            raise RuntimeError("Dataset not ready. Run prepare() first.")

        return VimeoSeptupletDataset(
            data_dir=self.dataset_dir,
            split="train",
            num_frames=self.num_frames,
            frame_indices=self.frame_indices,
            transform=self.transform,
        )

    def validation_dataset(self) -> VimeoSeptupletDataset:
        """Get validation/test dataset."""
        if not self.is_extracted():
            raise RuntimeError("Dataset not ready. Run prepare() first.")

        return VimeoSeptupletDataset(
            data_dir=self.dataset_dir,
            split="test",
            num_frames=self.num_frames,
            frame_indices=self.frame_indices,
            transform=self.transform,
        )

    def test_dataset(self) -> VimeoSeptupletDataset:
        """Alias for validation_dataset (Vimeo uses test set for validation)."""
        return self.validation_dataset()


def prepare_dataset(data_dir: Optional[str] = None, quiet: bool = False) -> dict:
    """
    Convenience function to download and prepare the dataset.

    Args:
        data_dir: Override data directory (default: ~/.cache/cvlization/data/)
        quiet: Suppress progress output

    Returns:
        dict with dataset statistics
    """
    kwargs = {"quiet": quiet}
    if data_dir:
        kwargs["data_dir"] = data_dir

    builder = VimeoSeptupletBuilder(**kwargs)
    builder.prepare()
    return builder.get_stats()


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vimeo Septuplet Dataset Manager")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (default: ~/.cache/cvlization/data/)")
    parser.add_argument("--download", action="store_true",
                        help="Download the dataset")
    parser.add_argument("--extract", action="store_true",
                        help="Extract the dataset")
    parser.add_argument("--prepare", action="store_true",
                        help="Download and extract if needed")
    parser.add_argument("--stats", action="store_true",
                        help="Print dataset statistics")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")

    args = parser.parse_args()

    kwargs = {"quiet": args.quiet}
    if args.data_dir:
        kwargs["data_dir"] = args.data_dir

    builder = VimeoSeptupletBuilder(**kwargs)

    if args.download:
        builder.download()

    if args.extract:
        builder.extract()

    if args.prepare:
        builder.prepare()

    if args.stats or not (args.download or args.extract or args.prepare):
        stats = builder.get_stats()
        print("\nVimeo Septuplet Dataset Statistics:")
        print("-" * 40)
        for key, value in stats.items():
            print(f"  {key}: {value}")
