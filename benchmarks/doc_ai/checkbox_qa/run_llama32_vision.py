#!/usr/bin/env python3
"""
Run LLaMA-3.2-Vision-11B on CheckboxQA benchmark.

LLaMA-3.2-Vision is Meta's vision-language model. Using the 4-bit quantized
version from Unsloth to fit within 22GB GPU memory.

Note: This model only supports single images, so we test with 1 page per document.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, MllamaProcessor

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

MODEL_ID = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"


def get_page_cache_root():
    return Path.home() / ".cache/cvlization/data/checkbox_qa/page_images"


def get_page_images(doc_id: str, max_pages: int = 1, page_cache_root: Optional[Path] = None):
    """Get page images for a document. LLaMA-3.2-Vision only supports single images."""
    if page_cache_root is None:
        page_cache_root = get_page_cache_root()
    doc_dir = page_cache_root / doc_id
    if not doc_dir.exists():
        return []
    pages = sorted(doc_dir.glob("page-*.png"))
    return pages[:max_pages] if max_pages else pages


def load_model(device: str = "cuda"):
    """Load LLaMA-3.2-Vision model (4-bit quantized)."""
    print(f"Loading model: {MODEL_ID}...")

    processor = MllamaProcessor.from_pretrained(MODEL_ID)

    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

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
    image_path: Path,
    question: str,
    max_tokens: int = 50,
    max_image_size: int = 1400
) -> str:
    """Ask a VQA question about the document image."""

    # Load and resize image
    image = resize_image(image_path, max_image_size)

    # Build message - force very short answers
    prompt = f"""Look at this document. Answer the question with ONLY the checkbox label, option text, or Yes/No. Give just the value - no explanation.

Question: {question}
Answer (1-5 words only):"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

    # Remove prompt tokens
    prompt_len = inputs["input_ids"].shape[1]
    output = processor.decode(
        output_ids[0][prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

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
    max_tokens: int = 50,
    max_image_size: int = 1400
):
    """Run LLaMA-3.2-Vision VQA on CheckboxQA."""

    output_dir.mkdir(parents=True, exist_ok=True)

    docs = load_subset(subset_path)
    print(f"Loaded {len(docs)} documents")

    predictions = []
    page_cache_root = get_page_cache_root()

    for doc_idx, doc in enumerate(docs):
        doc_id = doc["name"]
        print(f"\n[{doc_idx+1}/{len(docs)}] Processing {doc_id}...")

        # Get first page image (LLaMA-3.2-Vision is single-image only)
        page_images = get_page_images(doc_id, max_pages=1, page_cache_root=page_cache_root)
        if not page_images:
            print(f"  Warning: No pages found for {doc_id}")
            continue

        print(f"  Using 1 page (single-image model)")

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
            answer = ask_question(model, processor, page_images[0], question, max_tokens, max_image_size)
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
        print("RESULTS (LLaMA-3.2-Vision-11B on CheckboxQA)")
        print(f"{'='*60}")
        print(f"ANLS* Score: {results['anls_score']:.4f}")
        print(f"Total Questions: {results['total_questions']}")
        print(f"{'='*60}")

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Run LLaMA-3.2-Vision VQA on CheckboxQA"
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
        default=SCRIPT_DIR / "results/llama32_vision_dev",
        help="Output directory"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max tokens to generate (shorter = more concise)"
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=1400,
        help="Max image dimension (resize larger images)"
    )

    args = parser.parse_args()

    print("="*60)
    print("LLaMA-3.2-Vision-11B VQA on CheckboxQA")
    print("="*60)
    print("Note: Single-image model, using 1 page per document.")
    print("="*60 + "\n")

    model, processor = load_model()

    run_evaluation(
        model, processor,
        args.subset,
        args.output_dir,
        args.max_tokens,
        args.max_image_size
    )


if __name__ == "__main__":
    main()
