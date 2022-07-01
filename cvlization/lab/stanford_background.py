"""Adapted from MMSegmentation tutorial: 
https://colab.research.google.com/github/open-mmlab/mmsegmentation/blob/master/demo/MMSegmentation_Tutorial.ipynb

!wget http://dags.stanford.edu/data/iccv09Data.tar.gz -O stanford_background.tar.gz
!tar xf stanford_background.tar.gz
"""
from dataclasses import dataclass
import numpy as np
import os
import random
from subprocess import check_output
from glob import glob
from PIL import Image
from typing import Union, List
from ..data.dataset_builder import Dataset, DatasetProvider
from ..data.dataset_builder import TransformedMapStyleDataset


CLASSES = ("sky", "tree", "road", "grass", "water", "bldg", "mntn", "fg obj")
PALETTE = [
    [128, 128, 128],
    [129, 127, 38],
    [120, 69, 125],
    [53, 125, 34],
    [0, 11, 123],
    [118, 20, 12],
    [122, 81, 25],
    [241, 134, 51],
]


@dataclass
class StanfordBackgroundDatasetBuilder:
    """
    :param label_offset
        By default, label_offset is 0 which means class labels start from 0. Set it to 1 if you want to
        reserve 0 for the background class.
    :param flavor
        If None, `__getitem__()` will return a list of input arrays and a list of target arrays. If
        "torchvision", it will return an image (numpy array), and a dictionary.
    """

    channels_first: bool = True
    to_torch_tensor: bool = False
    flavor: str = None  # one of None, "torchvision"
    data_dir: str = "./data"
    dataset_folder: str = "stanford_background"
    dataset_original_folder: str = "iccv09Data"
    archive_ext: str = "tar.gz"
    train_ann_file: str = "train.txt"
    val_ann_file: str = "val.txt"
    download_url: str = "http://dags.stanford.edu/data/iccv09Data.tar.gz"
    preload: bool = False
    label_offset: int = 0

    def __post_init__(self):
        self.CLASSES = CLASSES
        self.PALETTE = PALETTE
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
    def seg_folder(self):
        return self.training_dataset().seg_folder

    @property
    def classes(self):
        return [c["name"] for c in self.training_dataset().coco.dataset["categories"]]

    @property
    def things_classes(self) -> List[dict]:
        return self.training_dataset().things_classes

    @property
    def stuff_classes(self) -> List[dict]:
        return self.training_dataset().stuff_classes

    @property
    def num_classes(self):
        return len(self.CLASSES)

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
        seg_map = torch.tensor(targets[0])
        return image, seg_map

    def training_dataset(self) -> Dataset:
        ds = StanfordBackgroundDataset(
            ann_file=self.train_ann_file,
            dataset_folder=self.dataset_folder,
            dataset_original_folder=self.dataset_original_folder,
            archive_ext=self.archive_ext,
            channels_first=self.channels_first,
            data_dir=self.data_dir,
            download_url=self.download_url,
            label_offset=self.label_offset,
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
        ds = StanfordBackgroundDataset(
            ann_file=self.val_ann_file,
            dataset_folder=self.dataset_folder,
            archive_ext=self.archive_ext,
            dataset_original_folder=self.dataset_original_folder,
            channels_first=self.channels_first,
            data_dir=self.data_dir,
            download_url=self.download_url,
            label_offset=self.label_offset,
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


class StanfordBackgroundDataset:
    CLASSES = CLASSES
    PALETTE = PALETTE

    def __init__(
        self,
        data_dir: str = "./data",
        dataset_folder: str = "stanford_background",
        dataset_original_folder: str = "iccv09Data",
        archive_ext: str = "tar.gz",
        img_folder: str = "images",
        seg_folder: str = "labels",
        ann_file: str = "train.txt",
        channels_first: bool = True,
        label_offset: int = 0,
        download_url: str = "http://dags.stanford.edu/data/iccv09Data.tar.gz",
    ):
        """
        Data flow: download -> extract -> load_annotations
        """
        self.annotations = None
        self.data_dir = data_dir
        self.dataset_folder = dataset_folder
        self.dataset_original_folder = dataset_original_folder
        self.archive_ext = archive_ext
        self.img_folder = img_folder
        self.seg_folder = seg_folder
        self.ann_file = ann_file
        self.channels_first = channels_first
        self.label_offset = label_offset
        self.download_url = download_url

    def __getitem__(self, index: int):
        if self.annotations is None:
            self.annotations = self.load_annotations()

        data_info = self.annotations[index]
        img_path = data_info["img_path"]
        label_path = data_info["label_path"]
        img = Image.open(img_path)
        np_img = np.array(img, dtype=np.float32)
        if self.channels_first:
            np_img = np_img.transpose((2, 0, 1))

        seg_map = np.loadtxt(label_path).astype(np.uint8)
        seg_map = seg_map[np.newaxis, :, :]
        return [np_img], [
            seg_map,  # shape: (n, H, W)
        ]

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
        check_output(f"wget {self.download_url} -O {outfile}".split())

    @property
    def download_file_local_path(self):
        return os.path.join(self.data_dir, f"{self.dataset_folder}.{self.archive_ext}")

    def extract(self):
        if not self._is_downloaded():
            self.download()
        outfile = self.download_file_local_path
        dst_dir = self.data_dir
        os.makedirs(dst_dir, exist_ok=True)
        if self.archive_ext.lower() == "tar.gz":
            check_output(f"tar -xzf {outfile} -C {dst_dir}".split())
        elif self.archive_ext.lower() == "zip":
            check_output(f"unzip {outfile} -d {dst_dir}".split())
        if (
            self.dataset_folder != self.dataset_original_folder
        ) and self.dataset_original_folder:
            src_folder = os.path.join(self.data_dir, self.dataset_original_folder)
            dst_folder = os.path.join(self.data_dir, self.dataset_folder)
            os.rename(src_folder, dst_folder)

    def _is_downloaded(self):
        return os.path.isfile(self.download_file_local_path)

    def _is_extracted(self):
        return os.path.isfile(
            os.path.join(
                self.data_dir,
                self.dataset_folder,
                "horizons.txt",
            )
        )

    def _load_annotations_from_ann_file(self, ann_path: str):
        data_infos = []
        with open(ann_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_id: str = line
                img_path = os.path.join(
                    self.data_dir,
                    self.dataset_folder,
                    self.img_folder,
                    img_id + ".jpg",
                )
                label_path = os.path.join(
                    self.data_dir,
                    self.dataset_folder,
                    self.seg_folder,
                    img_id + ".regions.txt",
                )
                data_info = {
                    "img_path": img_path,
                    "label_path": label_path,
                }
                data_infos.append(data_info)

        self.annotations = data_infos
        return data_infos

    def _scan_and_split_images(self):
        label_regions_paths = glob(
            os.path.join(
                self.data_dir, self.dataset_folder, self.seg_folder, "*.regions.txt"
            )
        )
        image_ids = []
        for label_regions_path in label_regions_paths:
            self._convert_regions_txt_to_png(label_regions_path)
            image_id = os.path.basename(label_regions_path).replace(".regions.txt", "")
            image_ids.append(image_id)
        random.seed(0)
        random.shuffle(image_ids)
        train_portion = 0.8
        split_idx = int(len(image_ids) * train_portion)
        train_image_ids = image_ids[:split_idx]
        val_image_ids = image_ids[split_idx:]
        train_txt_path = os.path.join(self.data_dir, self.dataset_folder, "train.txt")
        val_txt_path = os.path.join(self.data_dir, self.dataset_folder, "val.txt")
        with open(train_txt_path, "w") as f:
            for image_id in train_image_ids:
                f.write(image_id + "\n")
        with open(val_txt_path, "w") as f:
            for image_id in val_image_ids:
                f.write(image_id + "\n")

    def load_annotations(self):
        if not self._is_extracted():
            self.extract()

        assert self.ann_file in [
            "train.txt",
            "val.txt",
        ], f"ann_file is not valid: {self.ann_file}, need to be one of train.txt or val.txt"
        ann_path = os.path.join(self.data_dir, self.dataset_folder, self.ann_file)
        if not os.path.isfile(ann_path):
            self._scan_and_split_images()
            assert os.path.isfile(
                ann_path
            ), f"ann_file is not generated successfully: {ann_path}"

        return self._load_annotations_from_ann_file(ann_path)

    def _convert_regions_txt_to_png(self, txt_path):
        dst_path = txt_path.replace(".regions.txt", ".png")
        if os.path.isfile(dst_path):
            return
        seg_map = np.loadtxt(txt_path).astype(np.uint8)
        seg_img = Image.fromarray(seg_map).convert("P")
        seg_img.putpalette(np.array(self.PALETTE, dtype=np.uint8))
        seg_img.save(dst_path)


if __name__ == "__main__":
    """
    python -m cvlization.lab.stanford_background
    """

    ds = StanfordBackgroundDataset()
    print(len(ds), "examples in the dataset")
    example = ds[10]
    assert isinstance(example, tuple)
    inputs, targets = example
    img = inputs[0]
    print("image:", img.shape, img.dtype, type(img), "max:", img.max())
    for j in range(1):
        print(
            f"target[{j}]:",
            targets[j].shape,
            targets[j].dtype,
            type(targets[j]),
            "max:",
            np.max(targets[j]),
        )

    from torch.utils.data import DataLoader

    print("Now inspecting the dataloader..")
    dl = DataLoader(
        ds, batch_size=3, shuffle=False, collate_fn=lambda batch: tuple(zip(*batch))
    )
    for i, (inputs, targets) in enumerate(dl):
        # print("batch:", i, inputs[0].shape, targets[0].shape, targets[1].shape)
        print("batch:", i, len(inputs), "images")
        print("image 0:", inputs[0][0].shape)
        print("len(targets) =", len(targets))
        print("targets [0]:", targets[0])
        print("targets 0[0]:", targets[0][0].shape)
        break
