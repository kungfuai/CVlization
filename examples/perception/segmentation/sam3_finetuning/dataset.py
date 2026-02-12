"""Dataset download, conversion, and preparation for SAM3 fine-tuning."""
import io
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

SUPERCATEGORY = "dataset"


def download_and_convert_hf_dataset(hf_dataset: str, output_dir: str = "data") -> str:
    """Download HuggingFace dataset and convert to COCO format.

    Args:
        hf_dataset: HuggingFace dataset name (e.g., 'keremberke/pcb-defect-segmentation')
        output_dir: Output directory for converted dataset

    Returns:
        Path to converted COCO dataset directory
    """
    from datasets import load_dataset
    from pycocotools import mask as mask_utils

    print(f"\n{'='*80}")
    print(f"Downloading HuggingFace dataset: {hf_dataset}")
    print(f"{'='*80}")

    dataset_name = hf_dataset.split('/')[-1]
    dataset_path = Path(output_dir) / dataset_name
    dataset_path.mkdir(parents=True, exist_ok=True)

    if (dataset_path / "train" / "_annotations.coco.json").exists():
        print(f"✓ Dataset already exists at: {dataset_path}")
        return str(dataset_path)

    print("Downloading from HuggingFace...")
    dataset = load_dataset(hf_dataset)

    for split_name in ['train', 'validation']:
        if split_name not in dataset:
            continue

        split_dir_name = 'train' if split_name == 'train' else 'valid'
        split_path = dataset_path / split_dir_name
        images_path = split_path / "images"
        images_path.mkdir(parents=True, exist_ok=True)

        print(f"\nConverting {split_name} split to COCO format...")

        split_data = dataset[split_name]

        coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }

        if len(split_data) > 0:
            first_item = split_data[0]
            if 'objects' in first_item and 'category' in first_item['objects']:
                categories = set()
                for item in split_data:
                    if 'objects' in item:
                        for cat in item['objects']['category']:
                            categories.add(cat)

                for idx, cat_name in enumerate(sorted(categories)):
                    coco_data["categories"].append({
                        "id": idx,
                        "name": str(cat_name),
                        "supercategory": "object"
                    })

        ann_id = 1
        for img_id, item in enumerate(split_data, 1):
            image = item['image']
            if isinstance(image, dict):
                image = Image.open(io.BytesIO(image['bytes']))

            image_filename = f"image_{img_id:06d}.jpg"
            image_path = images_path / image_filename
            image.save(image_path)

            w, h = image.size

            coco_data["images"].append({
                "id": img_id,
                "file_name": image_filename,
                "width": w,
                "height": h
            })

            if 'objects' in item:
                objects = item['objects']
                bboxes = objects.get('bbox', [])
                segmentations = objects.get('segmentation', [])
                categories = objects.get('category', [])

                for bbox, seg, cat in zip(bboxes, segmentations, categories):
                    if isinstance(seg, list) and len(seg) > 0:
                        poly = seg[0] if isinstance(seg[0], list) else seg
                        rle = mask_utils.frPyObjects([poly], h, w)
                        binary_mask = mask_utils.decode(rle)[..., 0]
                        rle_encoded = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
                        rle_encoded['counts'] = rle_encoded['counts'].decode('utf-8')

                        cat_id = next((c['id'] for c in coco_data['categories'] if c['name'] == str(cat)), 0)
                        area = float(mask_utils.area(rle))

                        coco_data["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cat_id,
                            "bbox": bbox,
                            "area": area,
                            "segmentation": rle_encoded,
                            "iscrowd": 0
                        })
                        ann_id += 1

        coco_json_path = split_path / "_annotations.coco.json"
        with open(coco_json_path, 'w') as f:
            json.dump(coco_data, f)

        print(f"  ✓ {split_dir_name}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")

    print(f"\n✓ Dataset converted and saved to: {dataset_path}")
    return str(dataset_path)


def verify_dataset_structure(dataset_dir: str) -> bool:
    """Verify the dataset has the expected COCO structure."""
    dataset_path = Path(dataset_dir)

    required = [
        dataset_path / "train" / "_annotations.coco.json",
        dataset_path / "valid" / "_annotations.coco.json",
    ]

    for file in required:
        if not file.exists():
            print(f"ERROR: Required file not found: {file}")
            return False

    print(f"✓ Dataset structure verified: {dataset_dir}")
    return True


def prepare_dataset_structure(dataset_dir: str) -> str:
    """Prepare dataset structure for SAM3 training.

    SAM3 expects: {dataset_root}/{supercategory}/{train|test}/
    with images directly in the train/test dirs, not in an images/ subdirectory.
    """
    dataset_path = Path(dataset_dir)

    temp_dataset = Path("/tmp/sam3_dataset")
    restructured_path = temp_dataset / SUPERCATEGORY
    restructured_path.mkdir(parents=True, exist_ok=True)

    for split_name, dst_name in [("train", "train"), ("valid", "test")]:
        src_dir = dataset_path / split_name
        dst_dir = restructured_path / dst_name

        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        ann_src = src_dir / "_annotations.coco.json"
        ann_dst = dst_dir / "_annotations.coco.json"

        with open(ann_src, 'r') as f:
            coco_data = json.load(f)

        if 'info' not in coco_data:
            coco_data['info'] = {
                'description': 'SAM3 fine-tuning dataset',
                'version': '1.0',
                'year': 2025,
                'contributor': '',
                'date_created': '2025-01-01'
            }

        # Filter out degenerate annotations (zero-area bboxes cause NaN in matcher)
        orig_count = len(coco_data.get('annotations', []))
        coco_data['annotations'] = [
            ann for ann in coco_data.get('annotations', [])
            if len(ann.get('bbox', [])) == 4
            and ann['bbox'][2] > 0 and ann['bbox'][3] > 0
            and ann.get('area', 0) > 0
        ]
        filtered = orig_count - len(coco_data['annotations'])
        if filtered > 0:
            print(f"  Filtered {filtered} degenerate annotations from {dst_name}")

        with open(ann_dst, 'w') as f:
            json.dump(coco_data, f)

        images_src = src_dir / "images"
        if images_src.exists():
            for img_file in images_src.iterdir():
                if img_file.is_file():
                    shutil.copy2(img_file, dst_dir / img_file.name)

    print(f"✓ Restructured dataset to: {temp_dataset}")
    print(f"  Structure: {SUPERCATEGORY}/train/ and {SUPERCATEGORY}/test/")
    print(f"  Images flattened (moved from images/ subdirectory)")

    return str(temp_dataset)
