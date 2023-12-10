"""
Dataset adapted from: https://github.com/open-mmlab/mmpose/blob/master/demo/MMPose_Tutorial.ipynb
"""

from dataclasses import dataclass
import json
import numpy as np
import os
from subprocess import check_output
from PIL import Image
from typing import Union, List, Tuple
from panopticapi.utils import rgb2id
from pycocotools.coco import COCO
from ..data.dataset_builder import Dataset, DatasetProvider
from ..data.dataset_builder import TransformedMapStyleDataset


@dataclass
class CocoPoseTinyDatasetBuilder:
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
    dataset_folder: str = "coco_pose_tiny"
    train_ann_file: str = "person_keypoints_val2017_first80.json"
    val_ann_file: str = "person_keypoints_val2017_last20.json"
    download_url: str = (
        "https://storage.googleapis.com/research-datasets-public/coco_pose_tiny.zip"
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
        return [c["name"] for c in self.training_dataset().coco.dataset["categories"]]

    @property
    def num_classes(self):
        return len(CocoPanopticTinyDataset.CLASSES)

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
        masks = torch.tensor(targets[2])
        seg_map = torch.tensor(targets[3])
        return image, dict(boxes=boxes, labels=labels, masks=masks, seg_map=seg_map)

    def training_dataset(self) -> Dataset:
        ds = CocoPoseTinyDataset(
            ann_file=self.train_ann_file,
            dataset_folder=self.dataset_folder,
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
        ds = CocoPoseTinyDataset(
            ann_file=self.val_ann_file,
            dataset_folder=self.dataset_folder,
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


class CocoPoseTinyDataset:
    def __init__(
        self,
        data_dir: str = "./data",
        dataset_folder: str = "coco_pose_tiny",
        img_folder: str = "images",
        ann_file: str = "person_keypoints_val2017_first80.json",
        channels_first: bool = True,
        label_offset: int = 0,
        download_url: str = "https://storage.googleapis.com/research-datasets-public/coco_pose_tiny.zip",
    ):
        """
        Data flow: download -> extract -> load_annotations
        """
        self.annotations = None
        self.data_dir = data_dir
        self.dataset_folder = dataset_folder
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.channels_first = channels_first
        self.label_offset = label_offset
        self.download_url = download_url

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

        id = self.ids[index]
        image_filename = self.coco.loadImgs(id)[0]["file_name"]
        ann_info = [x for x in self.annotations if x["image_id"] == id]
        boxes = np.stack([ann["bbox"] for ann in ann_info], axis=0)
        joints_3d = np.stack([ann["joints_3d"] for ann in ann_info], axis=0)
        pil_img = Image.open(
            os.path.join(
                self.data_dir, self.dataset_folder, self.img_folder, image_filename
            )
        ).convert("RGB")
        np_img = np.array(pil_img, dtype=np.float32)
        np_img = (np_img - np_img.min()) / max(1e-3, np_img.max() - np_img.min())

        return [np_img], [
            boxes,
            joints_3d,
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

    def _get_db(self):
        ann_path = os.path.join(self.data_dir, self.dataset_folder, self.ann_file)
        coco = self.coco = COCO(ann_path)

        ann_ids = coco.getAnnIds()
        self.anns = anns = coco.loadAnns(ann_ids)
        catIds = coco.getCatIds(catNms=["person"])
        imgIds = coco.getImgIds(catIds=catIds)
        self.ids = imgIds

        db = []
        for idx, ann in enumerate(anns):
            # get image path
            img_id = ann["image_id"]
            img = coco.loadImgs([img_id])[0]
            image_file = os.path.join(
                self.data_dir,
                self.dataset_folder,
                self.img_folder,
                img["file_name"],
            )
            # get bbox
            bbox = ann["bbox"]
            # get keypoints
            keypoints = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
            num_joints = keypoints.shape[0]
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d[:, :2] = keypoints[:, :2]
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

            sample = {
                "image_file": image_file,
                "bbox": bbox,
                "rotation": 0,
                "joints_3d": joints_3d,
                "joints_3d_visible": joints_3d_visible,
                "bbox_score": 1,
                "bbox_id": idx,
                "image_id": img_id,
            }
            db.append(sample)

        # flip_pairs, upper_body_ids and lower_body_ids will be used
        # in some data augmentations like random flip
        self.ann_info = {}
        self.ann_info["flip_pairs"] = [
            # These numbers are specific to the keypoint definitions of this dataset.
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
            [13, 14],
            [15, 16],
        ]
        self.ann_info["upper_body_ids"] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.ann_info["lower_body_ids"] = (11, 12, 13, 14, 15, 16)
        self.ann_info["joint_weights"] = None
        self.ann_info["use_different_joint_weights"] = False

        return db

    def load_annotations(self):
        if not self._is_extracted():
            self.extract()

        self.annotations = self._get_db()
        return self.annotations


if __name__ == "__main__":
    """
    python -m cvlization.lab.coco_pose_tiny
    """

    ds = CocoPoseTinyDataset()
    assert ds._is_extracted()
    print(len(ds), "examples in the dataset")
    example = ds[10]
    assert isinstance(example, tuple)
    inputs, targets = example
    img = inputs[0]
    print("image:", img.shape, img.dtype, type(img))
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
        print("targets 0[1]:", targets[0][1].shape)
        break
