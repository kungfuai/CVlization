#!/usr/bin/env python3
"""
Run InternVL3-8B on CheckboxQA benchmark.

InternVL3-8B is OpenGVLab's advanced vision-language model with strong
document understanding capabilities. Uses dynamic tiling for high-resolution
document images.

Model: OpenGVLab/InternVL3-8B
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

MODEL_ID = "OpenGVLab/InternVL3-8B"

# Image processing constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_page_cache_root():
    return Path.home() / ".cache/cvlization/data/checkbox_qa/page_images"


def get_page_images(doc_id: str, max_pages: int = 1, page_cache_root: Optional[Path] = None):
    """Get page images for a document."""
    if page_cache_root is None:
        page_cache_root = get_page_cache_root()
    doc_dir = page_cache_root / doc_id
    if not doc_dir.exists():
        return []
    pages = sorted(doc_dir.glob("page-*.png"))
    return pages[:max_pages] if max_pages else pages


def build_transform(input_size: int):
    """Build image transform for InternVL3."""
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from target ratios."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    InternVL's dynamic tiling preprocessing for high-resolution images.

    Splits the image into tiles based on aspect ratio to preserve detail.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Generate all possible aspect ratios within constraints
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find closest matching aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize image
    resized_img = image.resize((target_width, target_height))

    # Split into tiles
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    # Optionally add thumbnail
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image(image_path: Path, max_num: int = 12, image_size: int = 448):
    """Load and preprocess image with dynamic tiling."""
    image = Image.open(image_path).convert('RGB')

    # Apply dynamic tiling
    images = dynamic_preprocess(image, max_num=max_num, image_size=image_size, use_thumbnail=True)

    # Transform all tiles
    transform = build_transform(image_size)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)

    return pixel_values


def load_model(device: str = "cuda"):
    """Load InternVL3-8B model."""
    print(f"Loading model: {MODEL_ID}...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval()

    if device == "cuda":
        model = model.cuda()

    print("Model loaded.")
    return model, tokenizer


def ask_question(
    model,
    tokenizer,
    image_paths: List[Path],
    question: str,
    max_tokens: int = 256,
    max_tiles: int = 12,
    image_size: int = 448
) -> str:
    """Ask a VQA question about document images (supports multiple pages)."""

    # Load and preprocess all page images with dynamic tiling
    all_pixel_values = []
    for image_path in image_paths:
        pixel_values = load_image(image_path, max_num=max_tiles, image_size=image_size)
        all_pixel_values.append(pixel_values)

    # Concatenate all pixel values
    pixel_values = torch.cat(all_pixel_values, dim=0)
    pixel_values = pixel_values.to(model.device).to(model.dtype)

    # Build prompt with image tokens for all tiles
    num_patches = pixel_values.shape[0]
    image_tokens = '<image>' * num_patches
    num_pages = len(image_paths)
    page_info = f"This document has {num_pages} page(s). " if num_pages > 1 else ""
    prompt = f"{image_tokens}\n{page_info}Look at this document and answer the following question. Answer concisely.\n\nQuestion: {question}\n\nAnswer:"

    # Generation config
    generation_config = dict(
        max_new_tokens=max_tokens,
        do_sample=False
    )

    # Run inference
    with torch.no_grad():
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=generation_config
        )

    return response.strip()


def load_subset(subset_path: Path) -> List[Dict]:
    """Load CheckboxQA subset."""
    docs = []
    with open(subset_path) as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def run_evaluation(
    model,
    tokenizer,
    subset_path: Path,
    output_dir: Path,
    max_tokens: int = 256,
    max_tiles: int = 12,
    image_size: int = 448,
    max_pages: int = 1
):
    """Run InternVL3-8B VQA on CheckboxQA."""

    output_dir.mkdir(parents=True, exist_ok=True)

    docs = load_subset(subset_path)
    print(f"Loaded {len(docs)} documents")

    predictions = []
    page_cache_root = get_page_cache_root()

    for doc_idx, doc in enumerate(docs):
        doc_id = doc["name"]
        print(f"\n[{doc_idx+1}/{len(docs)}] Processing {doc_id}...")

        # Get page images
        page_images = get_page_images(doc_id, max_pages=max_pages, page_cache_root=page_cache_root)
        if not page_images:
            print(f"  Warning: No pages found for {doc_id}")
            continue

        print(f"  Using {len(page_images)} page(s) with dynamic tiling (max {max_tiles} tiles)")

        # Answer questions
        doc_predictions = {
            "name": doc_id,
            "extension": "pdf",
            "annotations": []
        }

        for ann in doc["annotations"]:
            question_id = ann["id"]
            question = ann["key"]

            print(f"  Q: {question[:60]}...")
            # Pass all page images for multi-page support
            answer = ask_question(
                model, tokenizer, page_images, question,
                max_tokens, max_tiles, image_size
            )
            print(f"  A: {answer[:80]}")

            doc_predictions["annotations"].append({
                "id": question_id,
                "key": question,
                "values": [{"value": answer}]
            })

        predictions.append(doc_predictions)
        print(f"  Answered {len(doc['annotations'])} questions")

    # Save predictions
    pred_file = output_dir / "predictions.jsonl"
    with open(pred_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    print(f"\nPredictions saved to {pred_file}")

    # Run evaluation
    print("\nRunning evaluation...")
    import subprocess
    eval_output = output_dir / "eval_results.json"
    subprocess.run([
        "python3", str(SCRIPT_DIR / "evaluate.py"),
        "--pred", str(pred_file),
        "--gold", str(subset_path),
        "--output", str(eval_output)
    ])

    # Print results
    if eval_output.exists():
        with open(eval_output) as f:
            results = json.load(f)
        print(f"\n{'='*60}")
        print("RESULTS (InternVL3-8B on CheckboxQA)")
        print(f"{'='*60}")
        print(f"ANLS* Score: {results['anls_score']:.4f}")
        print(f"Total Questions: {results['total_questions']}")
        print(f"{'='*60}")

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Run InternVL3-8B VQA on CheckboxQA"
    )
    parser.add_argument(
        "--subset",
        type=Path,
        default=Path.home() / ".cache/cvlization/data/checkbox_qa/subset_dev.jsonl",
        help="CheckboxQA subset file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results/internvl3_dev",
        help="Output directory"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens to generate"
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=12,
        help="Max tiles for dynamic tiling (more tiles = higher resolution)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=448,
        help="Size of each image tile"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Max pages per document (currently uses first page only)"
    )

    args = parser.parse_args()

    print("="*60)
    print("InternVL3-8B VQA on CheckboxQA")
    print("="*60)
    print(f"Model: {MODEL_ID}")
    print(f"Dynamic tiling: max {args.max_tiles} tiles of {args.image_size}x{args.image_size}")
    print(f"Max pages: {args.max_pages}")
    print("="*60 + "\n")

    model, tokenizer = load_model()

    run_evaluation(
        model, tokenizer,
        args.subset,
        args.output_dir,
        args.max_tokens,
        args.max_tiles,
        args.image_size,
        args.max_pages
    )


if __name__ == "__main__":
    main()
