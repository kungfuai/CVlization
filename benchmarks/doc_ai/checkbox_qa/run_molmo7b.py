#!/usr/bin/env python3
"""
Run Molmo-7B on CheckboxQA dev subset.

Molmo-7B-D is a 7B parameter VLM from Allen AI with strong document understanding.
"""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers.dynamic_module_utils import get_imports

from page_cache import get_page_images, get_page_cache_root

try:
    from anls_star import anls_star
    ANLS_AVAILABLE = True
except ImportError:
    ANLS_AVAILABLE = False
    print("Warning: anls_star not available. Install with: pip install anls-star")

SCRIPT_DIR = Path(__file__).parent
DEFAULT_SUBSET = Path.home() / ".cache/cvlization/data/checkbox_qa/subset_dev.jsonl"

# Model options
MODELS = {
    "7b": "allenai/Molmo-7B-D-0924",
    "7b-o": "allenai/Molmo-7B-O-0924",
    "1b": "allenai/MolmoE-1B-0924",
}


def fixed_get_imports(filename):
    """Workaround: Remove tensorflow from imports."""
    imports = get_imports(filename)
    if "tensorflow" in imports:
        imports.remove("tensorflow")
    return imports


def load_model(model_id: str):
    """Load Molmo model and processor."""
    print(f"Loading {model_id}...")

    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='cuda:0',
            low_cpu_mem_usage=True
        )

    print("Model loaded.")
    return model, processor


def ask_question(model, processor, images: list, question: str, max_tokens: int = 100):
    """Ask a question about images using Molmo."""
    # Use first image only (Molmo doesn't do multi-image well for VQA)
    image = images[0] if images else None
    if image is None:
        return ""

    # Process inputs
    inputs = processor.process(
        images=[image],
        text=question
    )

    # Move to device
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        torch.cuda.empty_cache()
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=max_tokens, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

    # Decode
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text.strip()


def load_subset(subset_file: Path) -> list:
    """Load CheckboxQA subset."""
    documents = []
    with open(subset_file, 'r') as f:
        for line in f:
            documents.append(json.loads(line))
    return documents


def run_evaluation(
    model,
    processor,
    documents: list,
    output_dir: Path,
    max_pages: int = 1,
    max_tokens: int = 100,
    max_image_size: int = None,
):
    """Run evaluation on CheckboxQA documents."""
    predictions = {}
    page_cache_root = get_page_cache_root()

    for i, doc in enumerate(documents):
        doc_id = doc["name"]
        print(f"\n[{i+1}/{len(documents)}] Processing {doc_id}...")

        # Get page images
        page_paths = get_page_images(doc_id, max_pages=max_pages)
        # Load images and optionally resize
        page_images = []
        for p in page_paths:
            img = Image.open(p)
            if max_image_size and max(img.size) > max_image_size:
                ratio = max_image_size / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            page_images.append(img)
        if not page_images:
            print(f"  Warning: No pages for {doc_id}")
            continue

        print(f"  Using {len(page_images)} page(s)")

        predictions[doc_id] = {
            "name": doc_id,
            "extension": "pdf",
            "annotations": []
        }

        for ann in doc["annotations"]:
            question_id = ann["id"]
            question = ann["key"]

            print(f"  Q: {question[:60]}...")

            answer = ask_question(model, processor, page_images, question, max_tokens)
            print(f"  A: {answer[:80]}")

            predictions[doc_id]["annotations"].append({
                "id": question_id,
                "key": question,
                "values": [{"value": answer}]
            })

        print(f"  Answered {len(doc['annotations'])} questions")

    # Save predictions
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_file = output_dir / "predictions.jsonl"
    with open(pred_file, 'w') as f:
        for doc_id in sorted(predictions.keys()):
            predictions[doc_id]["annotations"].sort(key=lambda x: x["id"])
            f.write(json.dumps(predictions[doc_id]) + "\n")

    print(f"\nPredictions saved to {pred_file}")
    return predictions


def evaluate(predictions: dict, gold_documents: list, output_dir: Path):
    """Compute ANLS* score."""
    if not ANLS_AVAILABLE:
        print("Skipping evaluation - anls_star not available")
        return None

    gold_answers = {}
    for doc in gold_documents:
        for ann in doc["annotations"]:
            key = (doc["name"], ann["id"])
            gold_answers[key] = [v["value"] for v in ann["values"]]

    pred_answers = {}
    for doc_id, doc in predictions.items():
        for ann in doc["annotations"]:
            key = (doc_id, ann["id"])
            pred_answers[key] = ann["values"][0]["value"] if ann["values"] else ""

    scores = []
    for key, gold_vals in gold_answers.items():
        pred = pred_answers.get(key, "")
        score = max(anls_star(pred, g) for g in gold_vals) if gold_vals else 0.0
        scores.append(score)

    anls_score = sum(scores) / len(scores) if scores else 0.0

    print(f"\n{'='*60}")
    print(f"ANLS* Score: {anls_score:.4f}")
    print(f"Total Questions: {len(scores)}")
    print(f"{'='*60}")

    # Save results
    results = {
        "anls_score": anls_score,
        "total_questions": len(scores),
    }

    results_file = output_dir / "eval_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_file}")
    return anls_score


def main():
    parser = argparse.ArgumentParser(description="Run Molmo-7B on CheckboxQA")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="7b",
                        help="Model variant (default: 7b)")
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET,
                        help="Path to subset JSONL file")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory")
    parser.add_argument("--max-pages", type=int, default=1,
                        help="Max pages per document (default: 1)")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Max tokens to generate (default: 100)")
    parser.add_argument("--max-image-size", type=int, default=None,
                        help="Max image dimension (resize if larger)")

    args = parser.parse_args()

    model_id = MODELS[args.model]

    if args.output_dir is None:
        args.output_dir = SCRIPT_DIR / f"results/molmo_{args.model}_{args.max_pages}p"

    print("="*60)
    print(f"Molmo VQA on CheckboxQA")
    print("="*60)
    print(f"Model: {model_id}")
    print(f"Max pages: {args.max_pages}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Max image size: {args.max_image_size}")
    print("="*60)

    # Load model
    model, processor = load_model(model_id)

    # Load data
    documents = load_subset(args.subset)
    print(f"Loaded {len(documents)} documents")

    # Run evaluation
    predictions = run_evaluation(
        model, processor, documents, args.output_dir,
        max_pages=args.max_pages,
        max_tokens=args.max_tokens,
        max_image_size=args.max_image_size,
    )

    # Evaluate
    print("\nRunning evaluation...")
    evaluate(predictions, documents, args.output_dir)


if __name__ == "__main__":
    main()
