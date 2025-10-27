"""
Source data: https://www.elderlab.yorku.ca/resources/york-urban-line-segment-database-information/s
Prepared using script: https://github.com/zzsi/LETR/blob/dev/helper/york_split.py.
Much of the code here is thanks to the authors of the LETR model: https://github.com/mlpc-ucsd/LETR.
"""

from dataclasses import dataclass, field
import numpy as np
import os
from pathlib import Path
from subprocess import check_output
import torch
from torchvision.datasets import CocoDetection
from typing import Union, List
from . import york_transforms as T
from ..data.dataset_builder import Dataset, DatasetProvider
from ..torch.net.line_detection.letr.util import collate_fn
from cvl.core.downloads import get_cache_dir


def _default_data_dir() -> Path:
    """Return the centralized cache directory for York Lines data."""
    cache_dir = get_cache_dir() / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@dataclass
class YorkLinesDatasetBuilder:
    """
    :param label_offset
        By default, label_offset is 0 which means class labels start from 0. Set it to 1 if you want to
        reserve 0 for the background class.
    :param flavor
        If None, `__getitem__()` will return a list of input arrays and a list of target arrays. If
        "torchvision", it will return an image (numpy array), and a dictionary.
    """

    channels_first: bool = True
    data_dir: str = field(default_factory=lambda: str(_default_data_dir()))
    dataset_folder: str = "york_lines"
    train_ann_file: str = "annotations/lines_train.json"
    val_ann_file: str = "annotations/lines_val.json"
    flavor: str = None  # one of None, "torchvision"
    download_url: str = (
        "https://storage.googleapis.com/research-datasets-public/york_lines.zip"
    )
    preload: bool = True
    label_offset: int = 0

    def __post_init__(self):
        if self.preload:
            self.training_dataset()
            self.validation_dataset()

    @property
    def dataset_provider(self):
        return DatasetProvider.CVLIZATION

    @property
    def img_folder(self):
        return self.training_dataset().img_folder

    @property
    def classes(self):
        return ["line_segment"]

    @property
    def num_classes(self):
        return 1

    def training_dataset(self) -> Dataset:
        ds = YorkLinesDataset(
            flavor=self.flavor,
            ann_file=self.train_ann_file,
            dataset_folder=self.dataset_folder,
            channels_first=self.channels_first,
            data_dir=self.data_dir,
            download_url=self.download_url,
            label_offset=self.label_offset,
        )
        if self.preload:
            ds.load_annotations()
        return ds

    def validation_dataset(self) -> Union[Dataset, List[Dataset]]:
        # For some use cases, more than one validation datasets are returned.
        ds = YorkLinesDataset(
            flavor=self.flavor,
            ann_file=self.val_ann_file,
            dataset_folder=self.dataset_folder,
            channels_first=self.channels_first,
            data_dir=self.data_dir,
            download_url=self.download_url,
            label_offset=self.label_offset,
        )
        if self.preload:
            ds.load_annotations()
        return ds


class YorkLinesDataset:
    def __init__(
        self,
        download_url: str,
        flavor: str = None,  # one of None, "torchvision"
        data_dir: str = "./data",
        dataset_folder: str = "york_lines",
        img_folder: str = "images",
        ann_file: str = "annotations/lines_train.json",
        channels_first: bool = True,
        label_offset: int = 0,
    ):
        """
        Data flow: download -> extract -> load_annotations
        """
        self.annotations = None
        self.flavor = flavor
        self.data_dir = data_dir
        self.dataset_folder = dataset_folder
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.channels_first = channels_first
        self.label_offset = label_offset
        self.download_url = download_url

        self.prepare = ConvertCocoPolysToMask()
        self.transforms = make_coco_transforms(is_train="train" in self.ann_file)

    def _get_ann_info(self, idx):
        """Get COCO annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """
        ann_info = [self.anns[idx]]
        return self._parse_ann_info(ann_info)

    def __getitem__(self, index: int):
        if self.annotations is None:
            self.annotations = self.load_annotations()

        img, target = self.coco_ds[index]
        image_id = self.coco_ds.ids[index]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        img, target = self.transforms(img, target)
        assert isinstance(target, dict)
        assert "lines" in target, f"keys in target: {target.keys()}"
        if self.flavor == "torchvision":
            return img, target
        elif self.flavor is None:
            np_img = np.array(img, dtype=np.float32)
            np_img = (np_img - np_img.min()) / max(1e-3, np_img.max() - np_img.min())
            inputs = [np_img]
            targets = [
                # TODO: use relative coordinates for lines so that orig_size is not needed
                target["lines"].numpy(),
                target["labels"].numpy(),
                target["orig_size"].numpy(),
            ]
            return inputs, targets
        else:
            raise ValueError("flavor must be torchvision or None")

    def __len__(self):
        if self.annotations is None:
            self.annotations = self.load_annotations()
        return len(self.coco_ds)

    def download(self):
        assert self.download_url is not None, f"download_url is None"
        os.makedirs(self.data_dir, exist_ok=True)
        outfile = os.path.join(self.data_dir, f"{self.dataset_folder}.zip")
        check_output(f"wget {self.download_url} -O {outfile}".split())

    @property
    def download_file_local_path(self):
        return os.path.join(self.data_dir, f"{self.dataset_folder}.zip")

    def extract(self):
        if not self._is_downloaded():
            self.download()
        outfile = self.download_file_local_path
        dst_dir = self.data_dir
        os.makedirs(dst_dir, exist_ok=True)
        check_output(f"unzip {outfile} -d {dst_dir}".split())

    def _is_downloaded(self):
        return os.path.isfile(self.download_file_local_path)

    def _is_extracted(self):
        ann_path = os.path.join(
            self.data_dir,
            self.dataset_folder,
            self.ann_file,
        )
        return os.path.isfile(ann_path)

    def load_annotations(self):
        if not self._is_extracted():
            self.extract()

        data_root_for_coco = os.path.join(self.data_dir, self.dataset_folder)
        self.coco_ds = CocoDetection(
            root=os.path.join(data_root_for_coco, "images"),
            annFile=os.path.join(data_root_for_coco, self.ann_file),
        )
        self.annotations = self.coco_ds
        return self.coco_ds


class ConvertCocoPolysToMask(object):
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno]

        lines = [obj["line"] for obj in anno]
        lines = torch.as_tensor(lines, dtype=torch.float32).reshape(-1, 4)

        lines[:, 2:] += lines[:, :2]  # xyxy

        lines[:, 0::2].clamp_(min=0, max=w)
        lines[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target["lines"] = lines

        target["labels"] = classes

        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno]
        )
        target["area"] = area
        target["iscrowd"] = iscrowd

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(is_train: bool = False):
    """Torchvision transforms."""

    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.538, 0.494, 0.453], [0.257, 0.263, 0.273])]
    )

    scales = [480, 512, 544, 576, 608, 640, 672, 680, 690, 704, 736, 768, 788, 800]
    test_size = 1100
    max = 1333

    if is_train:
        return T.Compose(
            [
                T.RandomSelect(
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                ),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize(scales, max_size=max),
                        ]
                    ),
                ),
                T.ColorJitter(),
                normalize,
            ]
        )

    else:
        return T.Compose(
            [
                T.RandomResize([test_size], max_size=max),
                normalize,
            ]
        )


if __name__ == "__main__":
    """
    python -m cvlization.lab.york_lines
    """

    ds = YorkLinesDataset(
        download_url="https://storage.googleapis.com/research-datasets-public/york_lines.zip",
    )
    ds[0]
    assert ds._is_extracted()
    print(len(ds), "examples in the dataset")
    example = ds[10]
    assert isinstance(example, tuple)
    inputs, targets = example
    img = inputs[0]
    print("image:", img.shape, img.dtype, type(img))
    print(f"{len(targets)} targets")
    for j in range(2):
        print(
            f"target[{j}]:",
            targets[j].shape,
            targets[j].dtype,
            type(targets[j]),
            "max:",
            np.max(targets[j]),
        )

    from torch.utils.data import DataLoader

    ds = YorkLinesDataset(
        download_url="https://storage.googleapis.com/research-datasets-public/york_lines.zip",
        flavor="torchvision",
    )

    print("Now inspecting the dataloader..")
    dl = DataLoader(
        ds,
        batch_size=3,
        shuffle=False,
        # collate_fn=lambda batch: tuple(zip(*batch))
        collate_fn=collate_fn,
    )
    for i, (inputs, targets) in enumerate(dl):
        # print("batch:", i, inputs[0].shape, targets[0].shape, targets[1].shape)
        print("image batch:", i, inputs.tensors.shape)
        print("len(targets) =", len(targets))
        break
