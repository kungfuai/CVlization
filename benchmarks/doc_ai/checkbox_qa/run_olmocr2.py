#!/usr/bin/env python3
"""
Run olmOCR-2 on CheckboxQA benchmark.

olmOCR-2 is based on Qwen2.5-VL-7B and fine-tuned for OCR, but since it uses
the Qwen2.5-VL architecture, we can use VQA-style prompts to ask questions.

This script tests whether olmOCR-2 retains VQA capability for checkbox detection.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

MODEL_ID = "allenai/olmOCR-2-7B-1025-FP8"


def get_page_cache_root():
    return Path.home() / ".cache/cvlization/data/checkbox_qa/page_images"


def get_page_images(doc_id: str, max_pages: int = 5, page_cache_root: Optional[Path] = None):
    if page_cache_root is None:
        page_cache_root = get_page_cache_root()
    doc_dir = page_cache_root / doc_id
    if not doc_dir.exists():
        return []
    pages = sorted(doc_dir.glob("page-*.png"))
    return pages[:max_pages] if max_pages else pages


def load_model(device: str = "cuda"):
    """Load olmOCR-2 model."""
    print(f"Loading model: {MODEL_ID}...")

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("Model loaded.")
    return model, processor


def resize_image(image_path: Path, max_size: int) -> Image.Image:
    """Resize image to fit within max_size while preserving aspect ratio."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def ask_question(
    model,
    processor,
    image_paths: List[Path],
    question: str,
    max_tokens: int = 256,
    max_image_size: int = 1400
) -> str:
    """Ask a VQA question about the document images."""

    # Build message with resized images
    content = []
    for img_path in image_paths:
        img = resize_image(img_path, max_image_size)
        content.append({
            "type": "image",
            "image": img,
        })

    # Add the question
    content.append({
        "type": "text",
        "text": f"Look at this document and answer the following question. Answer concisely.\n\nQuestion: {question}\n\nAnswer:"
    })

    messages = [{"role": "user", "content": content}]

    # Prepare inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

    # Trim input tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output.strip()


def load_subset(subset_path: Path) -> List[Dict]:
    """Load CheckboxQA subset."""
    docs = []
    with open(subset_path) as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def run_evaluation(
    model,
    processor,
    subset_path: Path,
    output_dir: Path,
    max_pages: int = 5,
    max_tokens: int = 256,
    max_image_size: int = 1400
):
    """Run olmOCR-2 VQA on CheckboxQA."""

    output_dir.mkdir(parents=True, exist_ok=True)

    docs = load_subset(subset_path)
    print(f"Loaded {len(docs)} documents")

    predictions = []
    page_cache_root = get_page_cache_root()

    for doc_idx, doc in enumerate(docs):
        doc_id = doc["name"]
        print(f"\n[{doc_idx+1}/{len(docs)}] Processing {doc_id}...")

        # Get page images
        page_images = get_page_images(doc_id, max_pages, page_cache_root)
        if not page_images:
            print(f"  Warning: No pages found for {doc_id}")
            continue

        print(f"  Using {len(page_images)} pages")

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
            answer = ask_question(model, processor, page_images, question, max_tokens, max_image_size)
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
        print("RESULTS (olmOCR-2 on CheckboxQA)")
        print(f"{'='*60}")
        print(f"ANLS* Score: {results['anls_score']:.4f}")
        print(f"Total Questions: {results['total_questions']}")
        print(f"{'='*60}")

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Run olmOCR-2 VQA on CheckboxQA"
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
        default=SCRIPT_DIR / "results/olmocr2_dev",
        help="Output directory"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Max pages per document"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens to generate"
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=1400,
        help="Max image dimension (resize larger images)"
    )

    args = parser.parse_args()

    print("="*60)
    print("olmOCR-2 VQA on CheckboxQA")
    print("="*60)
    print("Testing if olmOCR-2 retains VQA capability for checkbox detection.")
    print("="*60 + "\n")

    model, processor = load_model()

    run_evaluation(
        model, processor,
        args.subset,
        args.output_dir,
        args.max_pages,
        args.max_tokens,
        args.max_image_size
    )


if __name__ == "__main__":
    main()
