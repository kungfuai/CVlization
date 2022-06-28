"""
Dataset adapted from: https://github.com/open-mmlab/mmpose/blob/master/demo/MMPose_Tutorial.ipynb
"""

from dataclasses import dataclass
import numpy as np
import os
from subprocess import check_output
from PIL import Image
from typing import Union, List, Tuple
from panopticapi.utils import rgb2id
from mmdet.datasets.coco_panoptic import COCOPanoptic
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
    train_ann_file: str = ""
    val_ann_file: str = ""
    download_url: str = (
        "https://storage.googleapis.com/research-datasets-public/coco_pose_tiny.zip"
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
        ds = CocoPanopticTinyDataset(
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
        ds = CocoPanopticTinyDataset(
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
        ann_file: str = "train.json",
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
        img_id = self.ids[idx]
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        # filter out unmatched images
        ann_info = [i for i in ann_info if i["image_id"] == img_id]
        return self._parse_ann_info(ann_info)

    def _parse_ann_info(self, ann_info):
        """Parse annotations and load panoptic ground truths.
        Args:
            img_info (int): Image info of an image.
            ann_info (list[dict]): Annotation info of an image.
        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_mask_infos = []

        for i, ann in enumerate(ann_info):
            x1, y1, w, h = ann["bbox"]
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            category_id = ann["category_id"]
            contiguous_cat_id = self.cat2label[category_id]

            is_thing = self.coco.load_cats(ids=category_id)[0]["isthing"]
            if is_thing:
                is_crowd = ann.get("iscrowd", False)
                if not is_crowd:
                    gt_bboxes.append(bbox)
                    gt_labels.append(contiguous_cat_id)
                else:
                    gt_bboxes_ignore.append(bbox)
                    is_thing = False

            mask_info = {
                "id": ann["id"],
                "category": contiguous_cat_id,
                "is_thing": is_thing,
            }
            gt_mask_infos.append(mask_info)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_mask_infos,
        )

        return ann

    def _parse_pan_label_image(
        self, pan_label_image: np.ndarray, mask_infos: List[dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Parse panoptic label image and load panoptic ground truths."""
        pan_png = pan_label_image
        pan_png = rgb2id(pan_png)

        gt_masks = []
        gt_seg = np.zeros_like(pan_png) + 255  # 255 as ignore

        for mask_info in mask_infos:
            mask = pan_png == mask_info["id"]
            gt_seg = np.where(mask, mask_info["category"], gt_seg)

            # The legal thing masks
            if mask_info.get("is_thing"):
                gt_masks.append(mask.astype(np.uint8))

        gt_masks = np.stack(gt_masks, axis=0)
        assert gt_masks.ndim == 3, f"gt_masks.ndim = {gt_masks.ndim}, expected: 3"

        return gt_seg, gt_masks

    def __getitem__(self, index: int):
        if self.annotations is None:
            self.annotations = self.load_annotations()

        id = self.ids[index]
        ann = self._get_ann_info(index)
        image_filename = self.coco.loadImgs(id)[0]["file_name"]
        pil_img = Image.open(
            os.path.join(
                self.data_dir, self.dataset_folder, self.img_folder, image_filename
            )
        ).convert("RGB")
        np_img = np.array(pil_img, dtype=np.float32)
        np_img = (np_img - np_img.min()) / max(1e-3, np_img.max() - np_img.min())

        pan_png_filename = image_filename.replace("jpg", "png")
        pan_label_image = Image.open(
            os.path.join(
                self.data_dir,
                self.dataset_folder,
                self.seg_folder,
                pan_png_filename,
            )
        )
        pan_label_image = np.array(pan_label_image)
        gt_seg, gt_masks = self._parse_pan_label_image(pan_label_image, ann["masks"])

        """
        Code sample from mmdet.datasets.pipelines.loading:
        (for parsing panoptic segmentations)

        pan_png = mmcv.imfrombytes(
            img_bytes, flag='color', channel_order='rgb').squeeze()
        pan_png = rgb2id(pan_png)

        gt_masks = []
        gt_seg = np.zeros_like(pan_png) + 255  # 255 as ignore

        for mask_info in results['ann_info']['masks']:
            mask = (pan_png == mask_info['id'])
            gt_seg = np.where(mask, mask_info['category'], gt_seg)

            # The legal thing masks
            if mask_info.get('is_thing'):
                gt_masks.append(mask.astype(np.uint8))
        
        if self.with_mask:
            h, w = results['img_info']['height'], results['img_info']['width']
            gt_masks = BitmapMasks(gt_masks, h, w)
            results['gt_masks'] = gt_masks
            results['mask_fields'].append('gt_masks')

        if self.with_seg:
            results['gt_semantic_seg'] = gt_seg
            results['seg_fields'].append('gt_semantic_seg')
        """

        if self.channels_first:
            np_img = np_img.transpose((2, 0, 1))

        box_labels = self.label_offset + np.expand_dims(ann["labels"], -1)
        return [np_img], [
            ann["bboxes"],  # shape: num_boxes, 4
            box_labels,  # shape: num_boxes, 1
            gt_masks,  # shape: num_boxes, h, w
            gt_seg,  # shape: h, w (int32)
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
        return os.path.isfile(
            os.path.join(
                self.data_dir,
                self.dataset_folder,
                self.ann_file,
            )
        )

    def _get_db(self):
        self.load_annotations()

    def load_annotations(self):
        if not self._is_extracted():
            self.extract()

        print("data_dir:", self.data_dir)
        print("ann_file:", self.ann_file)
        print("dataset_folder:", self.dataset_folder)
        # self.coco = COCOPanoptic(
        #     os.path.join(self.data_dir, self.dataset_folder, self.ann_file)
        # )
        # self.ids = list(sorted(self.coco.imgs.keys()))
        # self.CLASSES = self.coco.cats.keys()
        # print(f"{len(self.CLASSES)} classes")
        # self.cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        # return self.coco.anns


if __name__ == "__main__":
    """
    python -m cvlization.lab.coco_pose_tiny
    """

    ds = CocoPoseTinyDataset()
    print(len(ds), "examples in the dataset")
    example = ds[10]
    assert isinstance(example, tuple)
    inputs, targets = example
    img = inputs[0]
    print("image:", img.shape, img.dtype, type(img))
    for j in range(4):
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
