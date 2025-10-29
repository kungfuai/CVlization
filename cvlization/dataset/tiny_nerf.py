from dataclasses import dataclass, field
import numpy as np
import os
from pathlib import Path
import random
from subprocess import check_output
from typing import Union, List
from ..data.dataset_builder import Dataset, DatasetProvider
from ..data.dataset_builder import TransformedMapStyleDataset
from cvl.core.downloads import get_cache_dir


def _default_data_dir() -> Path:
    """Return the centralized cache directory for Tiny Nerf data."""
    cache_dir = get_cache_dir() / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@dataclass
class TinyNerfDatasetBuilder:

    channels_first: bool = False
    to_torch_tensor: bool = False
    data_dir: str = field(default_factory=lambda: str(_default_data_dir()))
    download_url: str = (
        # "https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz"
        "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
    )
    preload: bool = False
    label_offset: int = 0

    def __post_init__(self):
        if self.preload:
            self.training_dataset()
            self.validation_dataset()

    @property
    def dataset_provider(self):
        return DatasetProvider.CVLIZATION

    def get_totensor_transform(self):
        import torch

        def to_tensor(example):
            inputs, targets = example
            return [torch.tensor(x) for x in inputs], [torch.tensor(x) for x in targets]

        return to_tensor

    def training_dataset(self) -> Dataset:
        ds = TinyNerfDataset(
            channels_first=self.channels_first,
            data_dir=self.data_dir,
            download_url=self.download_url,
            is_train=True,
        )
        if self.preload:
            ds.load_annotations()

        if self.to_torch_tensor:
            ds = TransformedMapStyleDataset(
                base_dataset=ds, transform=self.get_totensor_transform()
            )
        return ds

    def validation_dataset(self) -> Union[Dataset, List[Dataset]]:
        ds = TinyNerfDataset(
            channels_first=self.channels_first,
            data_dir=self.data_dir,
            download_url=self.download_url,
            is_train=False,
        )
        if self.preload:
            ds.load_annotations()

        if self.to_torch_tensor:
            ds = TransformedMapStyleDataset(
                base_dataset=ds, transform=self.get_totensor_transform()
            )
        return ds


class TinyNerfDataset:
    def __init__(
        self,
        data_dir: str = "./data",
        channels_first: bool = False,
        download_url: str = "https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz",
        is_train: bool = True,
    ):
        self.annotations = None
        self.data_dir = data_dir
        self.channels_first = channels_first
        self.download_url = download_url
        self.is_train = is_train

    def __getitem__(self, index: int) -> dict:
        if self.annotations is None:
            self.annotations = self.load_annotations()

        pose, image = self.annotations[index]
        inputs = [
            pose,
            self.focal.reshape((1,)),
            np.array(image.shape[0]).reshape((1,)),
            np.array(image.shape[1]).reshape((1,)),
        ]
        targets = [image]
        return inputs, targets

    def __len__(self):
        if self.annotations is None:
            print("Loading annotations...")
            self.annotations = self.load_annotations()

        if self.annotations is None:
            raise ValueError("Annotations not loaded correctly.")
        return len(self.annotations)

    def download(self):
        assert self.download_url is not None, f"download_url is None"
        check_output("mkdir -p ./data".split())
        outfile = self.download_file_local_path
        check_output(f"wget --no-check-certificate {self.download_url} -O {outfile}".split())

    @property
    def download_file_local_path(self):
        filename = self.download_url.split("/")[-1]
        return os.path.join(self.data_dir, filename)

    def _is_downloaded(self):
        return os.path.isfile(self.download_file_local_path)

    def load_annotations(self):
        if not self._is_downloaded():
            self.download()
        print("Loading from", self.download_file_local_path)
        data = np.load(self.download_file_local_path)
        images = data["images"]
        if self.channels_first:
            images = images.transpose((0, 3, 1, 2))
        poses = data["poses"]
        self.focal = data["focal"]
        poses_and_images = list(zip(poses, images))
        random.seed(0)
        random.shuffle(poses_and_images)
        split_idx = int(len(poses_and_images) * 0.8)
        if self.is_train:
            poses_and_images = poses_and_images[:split_idx]
        else:
            poses_and_images = poses_and_images[split_idx:]
        # H, W = images.shape[1:3]
        self.annotations = poses_and_images
        return poses_and_images


if __name__ == "__main__":
    """
    python -m cvlization.dataset.tiny_nerf
    """

    ds = TinyNerfDataset()
    print(len(ds), "examples in the dataset")
    example = ds[10]
    assert isinstance(example, tuple)
    inputs, targets = example
    img = targets[0]
    print("image:", img.shape, img.dtype, type(img), "max:", img.max())
    for j in range(len(inputs)):
        print(
            f"inputs[{j}]:",
            inputs[j].shape,
            inputs[j].dtype,
            type(inputs[j]),
            "max:",
            np.max(inputs[j]),
        )
