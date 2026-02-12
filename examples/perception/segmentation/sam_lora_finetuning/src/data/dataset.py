"""Datasets for SAM LoRA fine-tuning.

Supports two formats:
- COCO JSON annotations (CocoSamDataset)
- Simple image + mask directory pairs (ImageMaskDataset)

Each sample is a single annotation: (preprocessed image, bbox prompt, gt mask).
"""

import io
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import Dataset

# HuggingFace repo and path for the default ring dataset
_HF_REPO_ID = "zzsi/cvl"
_HF_RING_PATH = "datasets/sam_lora_ring"


class CocoSamDataset(Dataset):
    """Yields per-annotation samples for SAM training.

    Each item is a dict with keys expected by SAM's batched forward:
        image, original_size, boxes, ground_truth_mask
    """

    def __init__(self, coco_json: str, images_dir: str, sam_model):
        self.images_dir = Path(images_dir)
        with open(coco_json) as f:
            coco = json.load(f)

        self.images_by_id = {img["id"]: img for img in coco["images"]}
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.pixel_mean = sam_model.pixel_mean
        self.pixel_std = sam_model.pixel_std
        self.img_size = sam_model.image_encoder.img_size

        # Build flat list of (image_info, annotation) pairs
        self.samples = []
        for ann in coco["annotations"]:
            img_info = self.images_by_id.get(ann["image_id"])
            if img_info is None:
                continue
            bbox = ann.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            self.samples.append((img_info, ann))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_info, ann = self.samples[idx]
        image = Image.open(self.images_dir / img_info["file_name"]).convert("RGB")
        w, h = image.size
        original_size = (h, w)

        # Decode mask
        gt_mask = self._decode_mask(ann, h, w)

        # COCO bbox is [x, y, w, h] -> convert to [x1, y1, x2, y2]
        bx, by, bw, bh = ann["bbox"]
        box = [bx, by, bx + bw, by + bh]

        # Preprocess image
        nd_image = np.array(image)
        input_image = self.transform.apply_image(nd_image)
        input_image_torch = torch.as_tensor(input_image).permute(2, 0, 1).contiguous().float()

        # Preprocess box
        box_np = np.array(box).reshape(1, 4)
        box_transformed = self.transform.apply_boxes(box_np, original_size)
        box_torch = torch.as_tensor(box_transformed, dtype=torch.float)[None, :]  # (1, 1, 4)

        return {
            "image": input_image_torch,
            "original_size": original_size,
            "boxes": box_torch,
            "ground_truth_mask": torch.from_numpy(gt_mask).float(),
        }

    @staticmethod
    def _decode_mask(ann, h, w):
        """Decode COCO segmentation (RLE or polygon) to binary mask."""
        seg = ann.get("segmentation")
        if seg is None:
            return np.zeros((h, w), dtype=np.uint8)

        if isinstance(seg, dict):
            # RLE format
            if isinstance(seg.get("counts"), str):
                rle = seg
            else:
                rle = mask_utils.frPyObjects(seg, h, w)
            return mask_utils.decode(rle).astype(np.uint8)

        if isinstance(seg, list):
            # Polygon format
            rle = mask_utils.frPyObjects(seg, h, w)
            return mask_utils.merge(rle).astype(np.uint8) if isinstance(rle, list) else mask_utils.decode(rle).astype(np.uint8)

        return np.zeros((h, w), dtype=np.uint8)


class ImageMaskDataset(Dataset):
    """Dataset that pairs images with binary masks from matching filenames.

    Directory layout expected::

        images_dir/foo.jpg
        masks_dir/foo.jpg      (or foo.png — matched by stem)

    Bounding boxes are derived from the mask's nonzero extent with a small
    random perturbation (same approach as Sam_LoRA's ``get_bounding_box``).
    """

    def __init__(self, images_dir: str, masks_dir: str, sam_model, bbox_noise: int = 20):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.bbox_noise = bbox_noise

        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.pixel_mean = sam_model.pixel_mean
        self.pixel_std = sam_model.pixel_std
        self.img_size = sam_model.image_encoder.img_size

        # Match images to masks by stem
        mask_stems = {p.stem: p for p in sorted(self.masks_dir.iterdir()) if p.is_file()}
        self.samples = []
        for img_path in sorted(self.images_dir.iterdir()):
            if not img_path.is_file():
                continue
            mask_path = mask_stems.get(img_path.stem)
            if mask_path is not None:
                self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        original_size = (h, w)

        # Load mask as single-channel binary
        mask = np.array(Image.open(mask_path).convert("L"))
        gt_mask = (mask > 127).astype(np.uint8)

        # Derive bounding box from mask with random perturbation
        box = self._bbox_from_mask(gt_mask)

        # Preprocess image
        nd_image = np.array(image)
        input_image = self.transform.apply_image(nd_image)
        input_image_torch = torch.as_tensor(input_image).permute(2, 0, 1).contiguous().float()

        # Preprocess box
        box_np = np.array(box).reshape(1, 4)
        box_transformed = self.transform.apply_boxes(box_np, original_size)
        box_torch = torch.as_tensor(box_transformed, dtype=torch.float)[None, :]  # (1, 1, 4)

        return {
            "image": input_image_torch,
            "original_size": original_size,
            "boxes": box_torch,
            "ground_truth_mask": torch.from_numpy(gt_mask).float(),
        }

    def _bbox_from_mask(self, mask: np.ndarray) -> list:
        """Compute [x1, y1, x2, y2] bbox from a binary mask with random noise."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return [0, 0, mask.shape[1], mask.shape[0]]
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        h, w = mask.shape
        noise = self.bbox_noise
        x1 = max(0, x1 - np.random.randint(0, noise))
        x2 = min(w, x2 + np.random.randint(0, noise))
        y1 = max(0, y1 - np.random.randint(0, noise))
        y2 = min(h, y2 + np.random.randint(0, noise))
        return [int(x1), int(y1), int(x2), int(y2)]


def collate_fn(batch):
    """Return list of dicts (SAM batched_input format)."""
    return list(batch)


def download_hf_dataset(hf_dataset: str, output_dir: str = "data") -> str:
    """Download HuggingFace dataset and convert to COCO format.

    Returns path to the converted dataset directory.
    """
    from datasets import load_dataset

    dataset_name = hf_dataset.split("/")[-1]
    dataset_path = Path(output_dir) / dataset_name
    dataset_path.mkdir(parents=True, exist_ok=True)

    if (dataset_path / "train" / "_annotations.coco.json").exists():
        print(f"Dataset already exists at: {dataset_path}")
        return str(dataset_path)

    print(f"Downloading HuggingFace dataset: {hf_dataset}")
    dataset = load_dataset(hf_dataset)

    for split_name in ["train", "validation"]:
        if split_name not in dataset:
            continue

        split_dir_name = "train" if split_name == "train" else "valid"
        split_path = dataset_path / split_dir_name
        images_path = split_path / "images"
        images_path.mkdir(parents=True, exist_ok=True)

        split_data = dataset[split_name]
        coco_data = {"images": [], "annotations": [], "categories": []}

        # Build categories
        if len(split_data) > 0:
            first_item = split_data[0]
            if "objects" in first_item and "category" in first_item["objects"]:
                categories = set()
                for item in split_data:
                    if "objects" in item:
                        for cat in item["objects"]["category"]:
                            categories.add(cat)
                for cat_idx, cat_name in enumerate(sorted(categories)):
                    coco_data["categories"].append(
                        {"id": cat_idx, "name": str(cat_name), "supercategory": "object"}
                    )

        ann_id = 1
        for img_id, item in enumerate(split_data, 1):
            image = item["image"]
            if isinstance(image, dict):
                image = Image.open(io.BytesIO(image["bytes"]))

            image_filename = f"image_{img_id:06d}.jpg"
            image.save(images_path / image_filename)
            img_w, img_h = image.size

            coco_data["images"].append(
                {"id": img_id, "file_name": image_filename, "width": img_w, "height": img_h}
            )

            if "objects" in item:
                objects = item["objects"]
                bboxes = objects.get("bbox", [])
                segmentations = objects.get("segmentation", [])
                categories_list = objects.get("category", [])

                for bbox, seg, cat in zip(bboxes, segmentations, categories_list):
                    if isinstance(seg, list) and len(seg) > 0:
                        poly = seg[0] if isinstance(seg[0], list) else seg
                        rle = mask_utils.frPyObjects([poly], img_h, img_w)
                        binary_mask = mask_utils.decode(rle)[..., 0]
                        rle_encoded = mask_utils.encode(
                            np.asfortranarray(binary_mask.astype(np.uint8))
                        )
                        rle_encoded["counts"] = rle_encoded["counts"].decode("utf-8")

                        cat_id = next(
                            (c["id"] for c in coco_data["categories"] if c["name"] == str(cat)), 0
                        )
                        area = float(mask_utils.area(mask_utils.frPyObjects([poly], img_h, img_w)[0]))

                        coco_data["annotations"].append(
                            {
                                "id": ann_id,
                                "image_id": img_id,
                                "category_id": cat_id,
                                "bbox": bbox,
                                "area": area,
                                "segmentation": rle_encoded,
                                "iscrowd": 0,
                            }
                        )
                        ann_id += 1

        coco_json_path = split_path / "_annotations.coco.json"
        with open(coco_json_path, "w") as f:
            json.dump(coco_data, f)

        print(
            f"  {split_dir_name}: {len(coco_data['images'])} images, "
            f"{len(coco_data['annotations'])} annotations"
        )

    print(f"Dataset saved to: {dataset_path}")
    return str(dataset_path)


def download_ring_dataset() -> str:
    """Download the default ring dataset from the zzsi/cvl HuggingFace repo.

    Returns:
        Path to the downloaded dataset root (contains train/ and test/ dirs).
    """
    from huggingface_hub import snapshot_download

    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_dir = Path(cache_home) / "cvlization" / "data" / "sam_lora_ring"

    # Check if already downloaded
    if (cache_dir / "train" / "images").is_dir() and (cache_dir / "train" / "masks").is_dir():
        print(f"Ring dataset already cached at: {cache_dir}")
        return str(cache_dir)

    print(f"Downloading ring dataset from {_HF_REPO_ID} ...")
    downloaded = snapshot_download(
        repo_id=_HF_REPO_ID,
        repo_type="model",
        allow_patterns=f"{_HF_RING_PATH}/**",
        local_dir=str(cache_dir / "_hf_snapshot"),
    )

    # snapshot_download puts files under <local_dir>/<path_in_repo>/...
    src = Path(downloaded) / _HF_RING_PATH
    # Move contents up to cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "test"]:
        split_src = src / split
        split_dst = cache_dir / split
        if split_src.is_dir() and not split_dst.is_dir():
            split_src.rename(split_dst)

    print(f"Ring dataset cached at: {cache_dir}")
    return str(cache_dir)
