from dataclasses import dataclass
import numpy as np
import os
from subprocess import check_output
from PIL import Image
from typing import Union, List
from ..data.dataset_builder import Dataset, DatasetProvider
from ..data.dataset_builder import TransformedMapStyleDataset


@dataclass
class PennFudanPedestrianDatasetBuilder:
    """
    Data source: https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
    """

    channels_first: bool = True
    to_torch_tensor: bool = True
    flavor: str = None  # one of None, "torchvision"
    data_dir: str = "./data"
    preload: bool = False
    include_masks: bool = True

    @property
    def dataset_provider(self):
        return DatasetProvider.CVLIZATION

    @property
    def image_dir(self):
        return os.path.join(self.data_dir, "PennFudanPed")

    @property
    def num_classes(self):
        return len(PennFudanPedestrianDataset.CLASSES)

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
        if self.include_masks:
            masks = torch.tensor(targets[2])
            return image, dict(boxes=boxes, labels=labels, masks=masks)
        else:
            return image, dict(boxes=boxes, labels=labels)

    def training_dataset(self) -> Dataset:
        ds = PennFudanPedestrianDataset(
            channels_first=self.channels_first,
            data_dir=self.data_dir,
            start_idx=0,
            end_idx=-50,
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
        ds = PennFudanPedestrianDataset(
            channels_first=self.channels_first,
            data_dir=self.data_dir,
            start_idx=-50,
            end_idx=None,
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


class PennFudanPedestrianDataset:

    CLASSES = (
        "Background",
        "Pedestrian",
    )

    def __init__(
        self,
        data_dir: str = "./data",
        channels_first: bool = True,
        start_idx: int = 0,
        end_idx: int = -50,
        include_masks: bool = True,
    ):
        """
        Data flow: download -> extract -> load_annotations
        """
        self.annotations = None
        self.data_dir = data_dir
        self.channels_first = channels_first
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.include_masks = include_masks
        self.parent_dir = "PennFudanPed"

    def __getitem__(self, index: int):
        # load images and masks
        img_path = os.path.join(
            self.data_dir, self.parent_dir, "PNGImages", self.imgs[index]
        )
        mask_path = os.path.join(
            self.data_dir, self.parent_dir, "PedMasks", self.masks[index]
        )
        # img = Image.open(img_path).convert("RGB")
        np_img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        np_img = (np_img - np_img.min()) / max(1e-3, np_img.max() - np_img.min())
        if self.channels_first:
            np_img = np_img.transpose((2, 0, 1))
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        if self.channels_first:
            pass
        else:
            masks = masks.transpose((1, 2, 0))

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = np.array(boxes, dtype=np.float32)
        # there is only one foreground class
        labels = np.ones((num_objs, 1), dtype=np.int64)
        labels = labels.astype(np.float32)
        masks = np.array(masks, dtype=np.uint8)

        image_id = np.array([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = np.zeros((num_objs,), dtype=np.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.include_masks:
            return [np_img], [target["boxes"], target["labels"], target["masks"]]
        else:
            return [np_img], [target["boxes"], target["labels"]]

    def __len__(self):
        if self.annotations is None:
            print("Loading annotations...")
            self.annotations = self.load_annotations()

        if self.annotations is None:
            raise ValueError("Annotations not loaded correctly.")
        return len(self.annotations)

    def download(self):
        check_output("mkdir -p ./data".split())
        outfile = os.path.join(self.data_dir, "PennFudanPed.zip")
        check_output(
            f"wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip -O {outfile}".split()
        )

    def extract(self):
        if not self._is_downloaded():
            self.download()
        outfile = os.path.join(self.data_dir, "PennFudanPed.zip")
        check_output(f"unzip {outfile} -d {self.data_dir}".split())

    def _is_downloaded(self):
        return os.path.isfile(os.path.join(self.data_dir, "PennFudanPed.zip"))

    def _is_extracted(self):
        has_one_input_image = os.path.isfile(
            os.path.join(
                self.data_dir, f"{self.parent_dir}/PNGImages/FudanPed00001.png"
            )
        )
        has_one_mask = os.path.isfile(
            os.path.join(
                self.data_dir, f"{self.parent_dir}/PedMasks/FudanPed00002_mask.png"
            )
        )
        return has_one_input_image and has_one_mask

    def load_annotations(self):
        if not self._is_extracted():
            self.extract()
        img_dir = os.path.join(self.data_dir, self.parent_dir, "PNGImages")
        assert os.path.isdir(img_dir)
        self.imgs = list(sorted(os.listdir(img_dir)))
        self.masks = list(
            sorted(os.listdir(os.path.join(self.data_dir, self.parent_dir, "PedMasks")))
        )
        self.imgs = self.imgs[self.start_idx : self.end_idx]
        self.masks = self.masks[self.start_idx : self.end_idx]
        return self.masks


if __name__ == "__main__":
    """
    python -m cvlization.lab.penn_fudan_pedestrian
    """
    dsb = PennFudanPedestrianDatasetBuilder(flavor=None)
    ds = PennFudanPedestrianDataset(start_idx=0, end_idx=12)
    # ds = dsb.training_dataset()
    print(len(ds), "examples in the dataset")
    example = ds[10]
    assert isinstance(example, tuple)
    inputs, targets = example
    img = inputs[0]
    print("image:", img.shape, img.dtype, type(img))
    for j in range(3):
        print(f"target {j}:", targets[j].shape, targets[j].dtype, type(targets[j]))

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
        print("target 0:", targets[0][:2])
        print("targets 0[0]:", targets[0][0].shape)
        print("targets 0[1]:", targets[0][1].shape)
        print("targets 0[2]:", targets[0][2].shape)
        break
