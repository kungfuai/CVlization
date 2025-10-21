"""
Download and prepare HuggingFace captcha-images dataset for Moondream2 fine-tuning.

Dataset: https://huggingface.co/datasets/project-sloth/captcha-images
- 10,000 captcha images (200x200)
- 6-digit numbers
- Train/Val/Test splits
- License: WTFPL (Do What You Want)

This script:
1. Downloads the dataset from HuggingFace
2. Converts to Q&A format for Moondream2
3. Saves images and creates JSONL training file
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import os


def download_and_prepare_captcha(output_dir: str = "data/captcha", split: str = "train"):
    """Download captcha dataset and convert to Moondream2 format."""

    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading captcha-images dataset (split: {split})...")
    dataset = load_dataset("project-sloth/captcha-images", split=split)

    print(f"Loaded {len(dataset)} samples")
    print(f"Saving images to: {images_dir}")

    # Prepare training data in Q&A format
    train_data = []

    for idx, example in enumerate(dataset):
        # Save image
        image = example['image']
        image_filename = f"captcha_{split}_{idx:05d}.png"
        image_path = images_dir / image_filename
        image.save(image_path)

        # Get solution (the 6-digit number)
        solution = example['solution']

        # Create Q&A pairs (can create multiple questions per image)
        qa_samples = [
            {
                "image": str(image_path),
                "question": "What is the text in this captcha image?",
                "answer": solution
            },
            {
                "image": str(image_path),
                "question": "Read the numbers shown in the image.",
                "answer": solution
            },
            {
                "image": str(image_path),
                "question": "What number is displayed?",
                "answer": solution
            }
        ]

        # Use first question variant for this sample
        train_data.append(qa_samples[0])

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} images...")

    # Save as JSONL
    jsonl_path = output_path / f"{split}.jsonl"
    with open(jsonl_path, 'w') as f:
        for sample in train_data:
            f.write(json.dumps(sample) + '\n')

    print(f"\n✓ Dataset prepared!")
    print(f"  Images: {images_dir}")
    print(f"  Training data: {jsonl_path}")
    print(f"  Total samples: {len(train_data)}")
    print(f"\nTo train:")
    print(f"  ./train.sh --data {jsonl_path}")

    return jsonl_path


def create_sample_subset(input_jsonl: str, output_jsonl: str, num_samples: int = 100):
    """Create a small subset for quick testing."""

    print(f"Creating subset of {num_samples} samples...")

    samples = []
    with open(input_jsonl, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            samples.append(json.loads(line))

    with open(output_jsonl, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"✓ Created subset: {output_jsonl} ({len(samples)} samples)")


def create_train_val_split(input_jsonl: str, output_dir: str, val_ratio: float = 0.1):
    """Split data into train and validation sets."""
    import random

    print(f"Creating train/val split (val_ratio={val_ratio})...")

    samples = []
    with open(input_jsonl, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    # Shuffle and split
    random.seed(42)
    random.shuffle(samples)

    val_size = int(len(samples) * val_ratio)
    train_samples = samples[val_size:]
    val_samples = samples[:val_size]

    # Save train set
    train_path = Path(output_dir) / "train.jsonl"
    with open(train_path, 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + '\n')

    # Save val set
    val_path = Path(output_dir) / "val.jsonl"
    with open(val_path, 'w') as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + '\n')

    print(f"✓ Created splits:")
    print(f"  Train: {train_path} ({len(train_samples)} samples)")
    print(f"  Val:   {val_path} ({len(val_samples)} samples)")

    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare captcha-images dataset for Moondream2 fine-tuning"
    )
    parser.add_argument(
        "--output-dir",
        default="data/captcha",
        help="Output directory for images and JSONL"
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to download"
    )
    parser.add_argument(
        "--subset",
        type=int,
        help="Create a subset with N samples (for quick testing)"
    )
    parser.add_argument(
        "--train-val-split",
        action="store_true",
        help="Create train/val split from the data"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1 = 10%%)"
    )

    args = parser.parse_args()

    # Download and prepare full dataset
    jsonl_path = download_and_prepare_captcha(args.output_dir, args.split)

    # Create subset if requested
    if args.subset:
        subset_path = Path(args.output_dir) / f"{args.split}_subset_{args.subset}.jsonl"
        create_sample_subset(jsonl_path, subset_path, args.subset)
        print(f"\nQuick test command:")
        print(f"  ./train.sh --data {subset_path} --epochs 1 --batch-size 1")

    # Create train/val split if requested
    if args.train_val_split:
        train_path, val_path = create_train_val_split(
            jsonl_path,
            args.output_dir,
            args.val_ratio
        )
        print(f"\nTraining command with validation:")
        print(f"  ./train.sh --data {train_path} --val-data {val_path} --epochs 2 --lr 3e-6 --eval-steps 100")


if __name__ == "__main__":
    main()
