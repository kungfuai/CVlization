#!/usr/bin/env python3
"""
SAM3 Fine-tuning with Actual Training

This script invokes SAM3's native training infrastructure to perform
actual fine-tuning with loss tracking and validation metrics.
"""
import argparse
import io
import os
import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch.multiprocessing
import yaml

# Patch set_start_method to always use force=True so SAM3's internal call
# doesn't fail when the context was already set (e.g. by HF dataset loading).
_orig_set_start_method = torch.multiprocessing.set_start_method
def _safe_set_start_method(method, force=False):
    _orig_set_start_method(method, force=True)
torch.multiprocessing.set_start_method = _safe_set_start_method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune SAM3 on COCO-format segmentation dataset with metric tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to COCO dataset root (train/, valid/ with _annotations.coco.json)",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="HuggingFace dataset to download (e.g., 'keremberke/pcb-defect-segmentation'). If not provided, uses synthetic test shapes.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sam3_finetuning",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=8e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to SAM3 checkpoint to resume from",
    )
    parser.add_argument(
        "--config-template",
        type=str,
        default=None,
        help="Use specific SAM3 config template (or create minimal config)",
    )

    return parser.parse_args()


def download_and_convert_hf_dataset(hf_dataset: str, output_dir: str = "data") -> str:
    """Download HuggingFace dataset and convert to COCO format.

    Args:
        hf_dataset: HuggingFace dataset name (e.g., 'keremberke/pcb-defect-segmentation')
        output_dir: Output directory for converted dataset

    Returns:
        Path to converted COCO dataset directory
    """
    from datasets import load_dataset
    import json
    from pycocotools import mask as mask_utils
    import numpy as np
    from PIL import Image

    print(f"\n{'='*80}")
    print(f"Downloading HuggingFace dataset: {hf_dataset}")
    print(f"{'='*80}")

    # Create output directory
    dataset_name = hf_dataset.split('/')[-1]
    dataset_path = Path(output_dir) / dataset_name
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if (dataset_path / "train" / "_annotations.coco.json").exists():
        print(f"✓ Dataset already exists at: {dataset_path}")
        return str(dataset_path)

    # Download dataset
    print("Downloading from HuggingFace...")
    dataset = load_dataset(hf_dataset)

    # Convert train and validation splits
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

        # Extract categories
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

        # Convert each image
        ann_id = 1
        for img_id, item in enumerate(split_data, 1):
            image = item['image']
            if isinstance(image, dict):
                image = Image.open(io.BytesIO(image['bytes']))

            # Save image
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

            # Convert annotations
            if 'objects' in item:
                objects = item['objects']
                bboxes = objects.get('bbox', [])
                segmentations = objects.get('segmentation', [])
                categories = objects.get('category', [])

                for bbox, seg, cat in zip(bboxes, segmentations, categories):
                    # Convert polygon to RLE
                    if isinstance(seg, list) and len(seg) > 0:
                        poly = seg[0] if isinstance(seg[0], list) else seg
                        rle = mask_utils.frPyObjects([poly], h, w)
                        binary_mask = mask_utils.decode(rle)[..., 0]
                        rle_encoded = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
                        rle_encoded['counts'] = rle_encoded['counts'].decode('utf-8')

                        # Find category ID
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

        # Save COCO JSON
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


def setup_sam3_environment():
    """Setup SAM3 package and environment."""
    sam3_source = Path("/opt/sam3")

    if not sam3_source.exists():
        print("ERROR: SAM3 source not found at /opt/sam3")
        print("Please ensure SAM3 is mounted in the container")
        sys.exit(1)

    # Add SAM3 to Python path
    sam3_path = str(sam3_source.absolute())
    if sam3_path not in sys.path:
        sys.path.insert(0, sam3_path)

    # Verify we can import SAM3
    try:
        import sam3
        print(f"✓ SAM3 module loaded from: {sam3.__file__}")
    except ImportError as e:
        print(f"ERROR: Cannot import SAM3: {e}")
        print("Installing SAM3 package...")

        # Copy to writable location and install
        import shutil
        sam3_copy = Path("/tmp/sam3")
        if not sam3_copy.exists():
            shutil.copytree(sam3_source, sam3_copy, symlinks=True)

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(sam3_copy)],
            check=True,
        )
        print("✓ SAM3 installed successfully")


def create_sam3_config(
    dataset_dir: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    num_gpus: int,
    config_template: str = None,
) -> str:
    """Create SAM3 training configuration based on roboflow template."""

    # Use SAM3's base config as template
    sam3_configs = Path("/opt/sam3/sam3/train/configs")

    if config_template and Path(config_template).exists():
        template_path = Path(config_template)
    else:
        # Use minimal roboflow config as base
        template_path = sam3_configs / "roboflow_v100" / "roboflow_v100_full_ft_100_images.yaml"

    if not template_path.exists():
        print(f"ERROR: Config template not found: {template_path}")
        sys.exit(1)

    # Read template
    with open(template_path) as f:
        config = yaml.safe_load(f)

    # Update paths
    config["paths"]["dataset_root"] = str(Path(dataset_dir).absolute())
    config["paths"]["experiment_log_dir"] = str(Path(output_dir).absolute())
    config["paths"]["bpe_path"] = "/opt/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

    # Update trainer settings
    config["trainer"]["max_epochs"] = epochs
    config["trainer"]["devices"] = num_gpus
    config["trainer"]["log_every_n_steps"] = 10  # More frequent logging
    config["trainer"]["val_check_interval"] = 1.0  # Validate every epoch

    # Update data settings
    config["data"]["train_batch_size"] = batch_size
    config["data"]["val_batch_size"] = 1

    # Update optimizer
    config["optim"]["learning_rate"] = lr

    # Update launcher
    config["launcher"]["num_nodes"] = 1
    config["launcher"]["gpus_per_node"] = num_gpus

    # Disable cluster submission
    config["submitit"]["use_cluster"] = False

    # Save config
    os.makedirs(output_dir, exist_ok=True)
    config_path = Path(output_dir) / "training_config.yaml"

    with open(config_path, "w") as f:
        f.write("# @package _global_\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Training configuration saved: {config_path}")
    return str(config_path)


def prepare_dataset_structure(dataset_dir: str) -> str:
    """Prepare dataset structure to match SAM3's roboflow expectations.

    SAM3's roboflow config expects: {dataset_root}/{supercategory}/{train|test}/
    with images directly in the train/test dirs, not in an images/ subdirectory.
    """
    import shutil
    import json

    dataset_path = Path(dataset_dir)

    # Create temporary restructured dataset in /tmp
    temp_dataset = Path("/tmp/sam3_dataset")
    supercategory = "-grccs"  # First supercategory in roboflow config

    restructured_path = temp_dataset / supercategory
    restructured_path.mkdir(parents=True, exist_ok=True)

    # Process train and valid (as test) directories
    for split_name, dst_name in [("train", "train"), ("valid", "test")]:
        src_dir = dataset_path / split_name
        dst_dir = restructured_path / dst_name

        # Remove existing if it exists
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Copy annotation file and add 'info' field if missing
        ann_src = src_dir / "_annotations.coco.json"
        ann_dst = dst_dir / "_annotations.coco.json"

        # Read COCO JSON and ensure 'info' field exists
        with open(ann_src, 'r') as f:
            coco_data = json.load(f)

        # Add 'info' field if missing (required by COCO evaluator)
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

        # Write updated COCO JSON
        with open(ann_dst, 'w') as f:
            json.dump(coco_data, f)

        # Copy images from images/ subdirectory to root of dst_dir
        images_src = src_dir / "images"
        if images_src.exists():
            for img_file in images_src.iterdir():
                if img_file.is_file():
                    shutil.copy2(img_file, dst_dir / img_file.name)

    print(f"✓ Restructured dataset to: {temp_dataset}")
    print(f"  Structure: {supercategory}/train/ and {supercategory}/test/")
    print(f"  Images flattened (moved from images/ subdirectory)")

    return str(temp_dataset)


def run_sam3_training(config_name: str, dataset_dir: str, output_dir: str, num_gpus: int, epochs: int = 20, lr: float = 8e-4):
    """Run SAM3 training using its native training script."""

    print("\n" + "=" * 80)
    print("STARTING SAM3 TRAINING")
    print("=" * 80)
    print(f"Config: {config_name}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Training will show:")
    print("  - Training loss per step")
    print("  - Validation metrics per epoch")
    print("  - Checkpoints saved to output directory")
    print("  - TensorBoard logs for visualization")
    print("=" * 80 + "\n")

    # Prepare dataset structure for SAM3
    dataset_dir = prepare_dataset_structure(dataset_dir)

    # Import and run SAM3 trainer
    try:
        # Initialize Hydra with absolute path to SAM3's config directory
        from hydra import initialize_config_dir, compose
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf

        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()

        # Initialize Hydra with absolute path to configs directory
        # SAM3 is installed at /opt/sam3 in the Docker image
        sam3_configs = "/opt/sam3/sam3/train/configs"
        initialize_config_dir(config_dir=sam3_configs, version_base="1.2")

        # Register OmegaConf resolvers (required by SAM3)
        from sam3.train.utils.train_utils import register_omegaconf_resolvers
        register_omegaconf_resolvers()

        # Extract config name (ensure it has the roboflow_v100/ prefix if needed)
        if '/' not in config_name:
            config_name = f"roboflow_v100/{config_name}"

        # Compose config with overrides for our custom paths
        cfg = compose(
            config_name=config_name,
            overrides=[
                f"paths.bpe_path=/opt/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                f"paths.roboflow_vl_100_root={dataset_dir}",
                f"paths.experiment_log_dir={output_dir}",
                f"launcher.gpus_per_node={num_gpus}",
                f"launcher.num_nodes=1",
                "submitit.use_cluster=false",
                f"trainer.max_epochs={epochs}",
                "trainer.skip_saving_ckpts=false",
                "trainer.val_epoch_freq=1",
            ]
        )

        print("=" * 80)
        print("Training Configuration:")
        print("=" * 80)
        print(f"  Config name: {config_name}")
        print(f"  BPE path: {cfg.paths.bpe_path}")
        print(f"  Dataset: {cfg.paths.roboflow_vl_100_root}")
        print(f"  Output: {cfg.paths.experiment_log_dir}")
        print(f"  GPUs: {cfg.launcher.gpus_per_node}")
        print(f"  Epochs: {cfg.trainer.max_epochs}")
        print("=" * 80 + "\n")

        # Import SAM3's training infrastructure and run directly
        from sam3.train.train import single_node_runner, add_pythonpath_to_sys_path
        import random

        add_pythonpath_to_sys_path()

        # Patch the matcher to handle NaN/Inf in cost matrices that cause
        # linear_sum_assignment to crash with "matrix contains invalid numeric entries".
        from sam3.train import matcher as _matcher_mod
        _orig_do_matching = _matcher_mod._do_matching
        def _safe_do_matching(cost, repeats=1, return_tgt_indices=False, do_filtering=False):
            if not np.all(np.isfinite(cost)):
                cost = np.copy(cost)
                cost[~np.isfinite(cost)] = 1e8
            return _orig_do_matching(cost, repeats=repeats, return_tgt_indices=return_tgt_indices, do_filtering=do_filtering)
        _matcher_mod._do_matching = _safe_do_matching

        # Use random port for distributed training
        main_port = random.randint(10000, 20000)

        # Run SAM3's training directly
        single_node_runner(cfg, main_port)

    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: Training requires GPU. On CPU, model loading may fail.")
        print("For actual training, run on a machine with GPU (24GB+ VRAM)")
        raise


def main():
    args = parse_args()

    print("=" * 80)
    print("SAM3 Fine-tuning - Full Training Pipeline")
    print("=" * 80)

    # Step 1: Setup dataset
    if not args.dataset_dir and not args.hf_dataset:
        # Use synthetic test shapes by default
        print("\n[1/4] Creating synthetic test shapes dataset...")
        from dataset_builder import DatasetBuilder
        DatasetBuilder()
        args.dataset_dir = "data/test_shapes"
    elif args.hf_dataset:
        # Download and convert HuggingFace dataset
        print("\n[1/4] Downloading and converting HuggingFace dataset...")
        args.dataset_dir = download_and_convert_hf_dataset(args.hf_dataset)
    else:
        # Use provided dataset directory
        print(f"\n[1/4] Using provided dataset: {args.dataset_dir}")

    # Step 2: Verify dataset
    print(f"\n[2/4] Verifying dataset structure...")
    if not verify_dataset_structure(args.dataset_dir):
        sys.exit(1)

    # Step 3: Setup SAM3
    print("\n[3/4] Setting up SAM3 environment...")
    setup_sam3_environment()

    # Step 4: Run training with SAM3's default config
    print("\n[4/4] Running SAM3 training...")
    config_name = args.config_template or "roboflow_v100_full_ft_100_images"
    print(f"Using config: {config_name}")
    print("\nThis will:")
    print("  ✓ Load SAM3 training config with Hydra")
    print("  ✓ Override paths for dataset, BPE tokenizer, and output")
    print("  ✓ Load SAM3 model (848M parameters)")
    print("  ✓ Run training loop with loss tracking")
    print("  ✓ Validate on validation set each epoch")
    print("  ✓ Save checkpoints and metrics")
    print("  ✓ Log to TensorBoard")
    print()

    try:
        run_sam3_training(
            config_name=config_name,
            dataset_dir=str(Path(args.dataset_dir).absolute()),
            output_dir=str(Path(args.output_dir).absolute()),
            num_gpus=args.num_gpus,
            epochs=args.epochs,
            lr=args.lr,
        )

        print("\n" + "=" * 80)
        print("✓ TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nOutputs saved to: {args.output_dir}")
        print(f"  - Checkpoints: {args.output_dir}/checkpoints/")
        print(f"  - TensorBoard: tensorboard --logdir {args.output_dir}")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("⚠ TRAINING FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        print("\nCommon issues:")
        print("  - No GPU available (requires 24GB+ VRAM)")
        print("  - HuggingFace authentication required for SAM3 model")
        print("  - Dataset format issues")
        print("\nFor dataset loading/processing verification, check if error is before model download.")
        print("For actual training, ensure GPU access and HF authentication.")
        sys.exit(1)


if __name__ == "__main__":
    main()
