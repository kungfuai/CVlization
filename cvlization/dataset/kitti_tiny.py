from dataclasses import dataclass, field
import numpy as np
import os
from pathlib import Path
from subprocess import check_output
from PIL import Image
from typing import Union, List
from ..data.dataset_builder import Dataset, DatasetProvider
from ..data.dataset_builder import TransformedMapStyleDataset
from cvl.core.downloads import get_cache_dir


def _default_data_dir() -> Path:
    """Return the centralized cache directory for Kitti data."""
    cache_dir = get_cache_dir() / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@dataclass
class KittiTinyDatasetBuilder:
    """
    :param label_offset
        By default, label_offset is 0 which means class labels start from 0. Set it to 1 if you want to
        reserve 0 for the background class.
    :param flavor
        If None, `__getitem__()` will return a list of input arrays and a list of target arrays. If
        "torchvision", it will return an image (numpy array), and a dictionary.
    """

    channels_first: bool = True
    to_torch_tensor: bool = True
    flavor: str = None  # one of None, "torchvision"
    data_dir: str = field(default_factory=lambda: str(_default_data_dir()))
    preload: bool = False
    label_offset: int = 1

    @property
    def dataset_provider(self):
        return DatasetProvider.CVLIZATION

    @property
    def image_dir(self):
        return os.path.join(self.data_dir, "kitti_tiny")

    @property
    def num_classes(self):
        return len(KittiTinyDataset.CLASSES)

    def get_totensor_transform(self):
        import torch

        def to_tensor(example):
            inputs, targets = example
            return [torch.tensor(x) for x in inputs], [torch.tensor(x) for x in targets]

        return to_tensor

    def to_torchvision(self, example):
        import torch

        inputs, targets = example
        image = torch.tensor(inputs[0])
        boxes = torch.tensor(targets[0])
        labels = torch.tensor(targets[1]).type(torch.long)
        labels = torch.squeeze(labels, -1)
        return image, dict(boxes=boxes, labels=labels)

    def training_dataset(self) -> Dataset:
        ds = KittiTinyDataset(
            ann_file="kitti_tiny/train.txt",
            channels_first=self.channels_first,
            data_dir=self.data_dir,
        )
        if self.preload:
            ds.load_annotations()
        if self.flavor == "torchvision":
            ds = TransformedMapStyleDataset(
                base_dataset=ds, transform=self.to_torchvision
            )
            return ds

        if self.to_torch_tensor:
            ds = TransformedMapStyleDataset(
                base_dataset=ds, transform=self.get_totensor_transform()
            )
        return ds

    def validation_dataset(self) -> Union[Dataset, List[Dataset]]:
        # For some use cases, more than one validation datasets are returned.
        ds = KittiTinyDataset(
            ann_file="kitti_tiny/val.txt",
            channels_first=self.channels_first,
            data_dir=self.data_dir,
        )
        if self.preload:
            ds.load_annotations()
        if self.flavor == "torchvision":
            ds = TransformedMapStyleDataset(
                base_dataset=ds, transform=self.to_torchvision
            )
            return ds

        if self.to_torch_tensor:
            ds = TransformedMapStyleDataset(
                base_dataset=ds, transform=self.get_totensor_transform()
            )
        return ds


class KittiTinyDataset:

    CLASSES = ("Car", "Pedestrian", "Cyclist")

    def __init__(
        self,
        data_dir: str = "./data",
        ann_file: str = "kitti_tiny/train.txt",
        channels_first: bool = True,
        label_offset: int = 0,
    ):
        """
        Data flow: download -> extract -> load_annotations
        """
        self.annotations = None
        self.data_dir = data_dir
        self.ann_file = os.path.join(data_dir, ann_file)
        self.channels_first = channels_first
        self.label_offset = label_offset

    def __getitem__(self, index: int):
        if self.annotations is None:
            self.annotations = self.load_annotations()

        row = self.annotations[index]
        np_img = np.array(Image.open(row["image_path"]), dtype=np.float32)
        np_img = (np_img - np_img.min()) / max(1e-3, np_img.max() - np_img.min())
        if self.channels_first:
            np_img = np_img.transpose((2, 0, 1))
        return [np_img], [
            row["bboxes"],
            self.label_offset + np.expand_dims(row["labels"], -1),
        ]

    def __len__(self):
        if self.annotations is None:
            print("Loading annotations...")
            self.annotations = self.load_annotations()

        if self.annotations is None:
            raise ValueError("Annotations not loaded correctly.")
        return len(self.annotations)

    def download(self):
        check_output("mkdir -p ./data".split())
        outfile = os.path.join(self.data_dir, "kitti_tiny.zip")
        check_output(
            f"wget https://download.openmmlab.com/mmdetection/data/kitti_tiny.zip -O {outfile}".split()
        )

    def extract(self):
        if not self._is_downloaded():
            self.download()
        outfile = os.path.join(self.data_dir, "kitti_tiny.zip")
        check_output(f"unzip {outfile} -d {self.data_dir}".split())

    def _is_downloaded(self):
        return os.path.isfile(os.path.join(self.data_dir, "kitti_tiny.zip"))

    def _is_extracted(self):
        return os.path.isfile(os.path.join(self.data_dir, "kitti_tiny/train.txt"))

    def load_annotations(self):
        if not self._is_extracted():
            self.extract()
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        annotations = []
        with open(self.ann_file) as f:
            image_ids = [line.strip("\n") for line in f if len(line) > 1]
            for image_id in image_ids:
                image_path = os.path.join(
                    self.data_dir, "kitti_tiny/training/image_2", image_id + ".jpeg"
                )
                label_filepath = os.path.join(
                    self.data_dir, "kitti_tiny/training/label_2", image_id + ".txt"
                )
                with open(label_filepath) as f2:
                    content = [line.strip().split(" ") for line in f2 if len(line) > 1]
                    bbox_names = [x[0] for x in content]
                    bboxes = [[float(info) for info in x[4:8]] for x in content]
                    gt_bboxes = []
                    gt_labels = []
                    gt_bboxes_ignore = []
                    gt_labels_ignore = []

                    # filter 'DontCare'
                    for bbox_name, bbox in zip(bbox_names, bboxes):
                        if bbox_name in cat2label:
                            gt_labels.append(cat2label[bbox_name])
                            gt_bboxes.append(bbox)
                        else:
                            gt_labels_ignore.append(-1)
                            gt_bboxes_ignore.append(bbox)

                row = dict(
                    image_path=image_path,
                    bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                    labels=np.array(gt_labels, dtype=np.int32),
                    bboxes_ignore=(
                        np.array(gt_bboxes_ignore, dtype=np.float32).reshape(-1, 4),
                    ),
                    labels_ignore=np.array(gt_labels_ignore, dtype=np.int32),
                )
                annotations.append(row)

        self.annotations = annotations
        return annotations


if __name__ == "__main__":
    """
    python -m cvlization.dataset.kitti_tiny
    """
    kitti = KittiTinyDataset()
    print(len(kitti), "examples in the dataset")
    example = kitti[10]
    assert isinstance(example, tuple)
    inputs, targets = example
    img = inputs[0]
    print("image:", img.shape, img.dtype, type(img))
    print("target 0:", targets[0].shape, targets[0].dtype, type(targets[0]))
    print("target 1:", targets[1].shape, targets[1].dtype, type(targets[1]))

    from torch.utils.data import DataLoader

    print("Now inspecting the dataloader..")
    dl = DataLoader(
        kitti, batch_size=3, shuffle=False, collate_fn=lambda batch: tuple(zip(*batch))
    )
    for i, (inputs, targets) in enumerate(dl):
        # print("batch:", i, inputs[0].shape, targets[0].shape, targets[1].shape)
        print("batch:", i, len(inputs), "images")
        print("image 0:", inputs[0][0].shape)
        print("len(targets) =", len(targets))
        print("targets [0]:", targets[0])
        print("targets 0[0]:", targets[0][0].shape)
        print("targets 0[1]:", targets[0][1].shape)
        break
