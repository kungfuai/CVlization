#!/usr/bin/env python3
"""
Sample data preparation script for Granite-Docling fine-tuning.

This script demonstrates how to prepare your document dataset for fine-tuning.
It converts documents + annotations into the required JSON/JSONL format.
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def create_sample_dataset(output_path: str, num_samples: int = 10):
    """Create a sample dataset for testing.

    In practice, you would:
    1. Load your document images
    2. Load corresponding annotations/ground truth
    3. Convert to the required format
    """
    samples = []

    for i in range(num_samples):
        sample = {
            "image_path": f"/workspace/data/images/doc_{i:04d}.png",
            "prompt": "Extract text and layout as DocTags.",
            "doctags": f"""<doc>
  <title>Sample Document {i}</title>
  <para>This is a sample document for fine-tuning demonstration.</para>
  <para>The document contains structured content that needs to be extracted.</para>
  <table>
    <row><cell>Header 1</cell><cell>Header 2</cell></row>
    <row><cell>Data 1</cell><cell>Data 2</cell></row>
  </table>
</doc>"""
        }
        samples.append(sample)

    # Save as JSON
    if output_path.endswith('.jsonl'):
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Created sample dataset with {num_samples} samples at: {output_path}")


def convert_custom_dataset(
    images_dir: str,
    annotations_file: str,
    output_path: str,
    image_ext: str = ".png"
):
    """Convert your custom dataset to the required format.

    Args:
        images_dir: Directory containing document images
        annotations_file: JSON file with annotations
        output_path: Output JSON/JSONL file
        image_ext: Image file extension
    """
    images_path = Path(images_dir)

    # Load annotations
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    samples = []

    for anno in annotations:
        # Adapt this to your annotation format
        image_id = anno.get('image_id') or anno.get('id')
        image_file = images_path / f"{image_id}{image_ext}"

        if not image_file.exists():
            print(f"Warning: Image not found: {image_file}")
            continue

        # Convert your annotation format to DocTags
        # This is a placeholder - adapt to your actual format
        doctags = anno.get('doctags') or anno.get('ground_truth')

        if not doctags:
            print(f"Warning: No doctags for image: {image_file}")
            continue

        sample = {
            "image_path": str(image_file),
            "prompt": anno.get('prompt', "Extract text and layout as DocTags."),
            "doctags": doctags
        }
        samples.append(sample)

    # Save
    if output_path.endswith('.jsonl'):
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(samples)} samples to: {output_path}")


def validate_dataset(dataset_path: str):
    """Validate a dataset file."""
    issues = []

    if dataset_path.endswith('.jsonl'):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            samples = [json.loads(line) for line in f]
    else:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)

    for i, sample in enumerate(samples):
        # Check required fields
        if 'image_path' not in sample:
            issues.append(f"Sample {i}: Missing 'image_path'")
        elif not Path(sample['image_path']).exists():
            issues.append(f"Sample {i}: Image file not found: {sample['image_path']}")

        if 'doctags' not in sample:
            issues.append(f"Sample {i}: Missing 'doctags'")

        if 'prompt' not in sample:
            print(f"Sample {i}: No prompt (will use default)")

    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
        return False
    else:
        print(f"✓ Dataset is valid ({len(samples)} samples)")
        return True


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Granite-Docling fine-tuning")
    parser.add_argument("--mode", choices=["sample", "convert", "validate"], required=True,
                       help="Mode: sample (create sample data), convert (convert custom data), validate (check dataset)")
    parser.add_argument("--output", type=str, help="Output JSON/JSONL file")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of sample documents to create")
    parser.add_argument("--images-dir", type=str, help="Directory with images (for convert mode)")
    parser.add_argument("--annotations", type=str, help="Annotations JSON file (for convert mode)")
    parser.add_argument("--image-ext", type=str, default=".png", help="Image file extension")
    parser.add_argument("--dataset", type=str, help="Dataset file to validate")

    args = parser.parse_args()

    if args.mode == "sample":
        if not args.output:
            print("Error: --output required for sample mode")
            return 1
        create_sample_dataset(args.output, args.num_samples)

    elif args.mode == "convert":
        if not all([args.images_dir, args.annotations, args.output]):
            print("Error: --images-dir, --annotations, and --output required for convert mode")
            return 1
        convert_custom_dataset(args.images_dir, args.annotations, args.output, args.image_ext)

    elif args.mode == "validate":
        if not args.dataset:
            print("Error: --dataset required for validate mode")
            return 1
        if not validate_dataset(args.dataset):
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
