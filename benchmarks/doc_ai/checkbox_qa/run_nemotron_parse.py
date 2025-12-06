#!/usr/bin/env python3
"""
Run Nemotron Parse on CheckboxQA benchmark.

This script documents that Nemotron Parse is NOT suitable for checkbox QA,
but we run it anyway to keep a record of its performance.

Approach:
1. Parse each page with Nemotron Parse
2. Extract markdown content
3. Use heuristic matching to find answers (will perform poorly)
4. Evaluate with ANLS* metric

See nemotron_notes.md for detailed limitations.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, GenerationConfig

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Try to import page_cache, fallback to direct file lookup
try:
    from page_cache import get_page_images, get_page_cache_root
except ImportError:
    # Fallback for environments without pypdfium2
    def get_page_cache_root():
        return Path.home() / ".cache/cvlization/data/checkbox_qa/page_images"

    def get_page_images(doc_id: str, max_pages: int = 20, page_cache_root: Optional[Path] = None):
        if page_cache_root is None:
            page_cache_root = get_page_cache_root()
        doc_dir = page_cache_root / doc_id
        if not doc_dir.exists():
            return []
        pages = sorted(doc_dir.glob("page-*.png"))
        return pages[:max_pages] if max_pages else pages

MODEL_ID = "nvidia/NVIDIA-Nemotron-Parse-v1.1"
DEFAULT_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown>"


def load_model(device: str = "cuda"):
    """Load Nemotron Parse model."""
    print(f"Loading model: {MODEL_ID}...")

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    gen_config = GenerationConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("Model loaded.")
    return model, processor, gen_config


def parse_page(model, processor, gen_config, image: Image.Image, max_tokens: int = 4096) -> str:
    """Run Nemotron Parse on a single page."""
    inputs = processor(images=[image], text=DEFAULT_PROMPT, return_tensors="pt")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config, max_new_tokens=max_tokens)

    text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return text


def clean_markdown(text: str) -> str:
    """Remove bbox/class tags from Nemotron output."""
    # Remove coordinate and class tags
    text = re.sub(r'<x_[\d.]+>', '', text)
    text = re.sub(r'<y_[\d.]+>', '', text)
    text = re.sub(r'<class_[^>]+>', '', text)
    # Clean up LaTeX
    text = text.replace('\\(', '').replace('\\)', '')
    text = text.replace('\\square', '□')
    text = text.replace('\\boxtimes', '☑')
    text = text.replace('\\checkmark', '✓')
    return text.strip()


def extract_answer_heuristic(markdown: str, question: str) -> str:
    """
    Heuristic answer extraction from markdown.

    This is expected to perform poorly - Nemotron Parse doesn't capture
    checkbox states, so we can only do keyword matching.
    """
    markdown_clean = clean_markdown(markdown).lower()
    question_lower = question.lower()

    # Try to find question text in markdown
    # Look for patterns like "question: answer" or nearby text

    # Extract key terms from question
    key_terms = []
    for term in question_lower.split():
        if len(term) > 3 and term not in ['what', 'which', 'does', 'the', 'for', 'and', 'is']:
            key_terms.append(term)

    # Search for Yes/No patterns near question terms
    for term in key_terms[:3]:
        if term in markdown_clean:
            # Find position and look for Yes/No nearby
            idx = markdown_clean.find(term)
            context = markdown_clean[max(0, idx-100):min(len(markdown_clean), idx+200)]

            # Check for checkbox patterns
            if '□ yes' in context or 'yes □' in context:
                if '☑ yes' in context or 'yes ☑' in context:
                    return "Yes"
                if '□ no' not in context:
                    return "No"  # Yes is unchecked, No might be checked
            if '□ no' in context or 'no □' in context:
                if '☑ no' in context or 'no ☑' in context:
                    return "No"
                if '□ yes' not in context:
                    return "Yes"  # No is unchecked, Yes might be checked

            # Look for direct mentions
            if ' yes' in context and ' no' not in context:
                return "Yes"
            if ' no' in context and ' yes' not in context:
                return "No"

    # Default: return empty (will get 0 ANLS)
    return ""


def load_subset(subset_path: Path) -> List[Dict]:
    """Load CheckboxQA subset."""
    docs = []
    with open(subset_path) as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def run_evaluation(
    model, processor, gen_config,
    subset_path: Path,
    output_dir: Path,
    max_pages: int = 1,
    max_tokens: int = 4096
):
    """Run Nemotron Parse on CheckboxQA and evaluate."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load subset
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

        # Parse all pages and combine markdown
        all_markdown = []
        for page_path in page_images:
            print(f"  Parsing {page_path.name}...")
            image = Image.open(page_path).convert("RGB")
            markdown = parse_page(model, processor, gen_config, image, max_tokens)
            all_markdown.append(markdown)

        combined_markdown = "\n\n".join(all_markdown)

        # Save raw output for debugging
        (output_dir / f"{doc_id}_raw.txt").write_text(combined_markdown)

        # Answer questions
        doc_predictions = {
            "name": doc_id,
            "extension": "pdf",
            "annotations": []
        }

        for ann in doc["annotations"]:
            question_id = ann["id"]
            question = ann["key"]

            answer = extract_answer_heuristic(combined_markdown, question)

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
        print("RESULTS (Nemotron Parse on CheckboxQA)")
        print(f"{'='*60}")
        print(f"ANLS* Score: {results['anls_score']:.4f}")
        print(f"Total Questions: {results['total_questions']}")
        print(f"\nNote: This poor performance is expected.")
        print("See nemotron_notes.md for limitations.")
        print(f"{'='*60}")

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Run Nemotron Parse on CheckboxQA (for documentation purposes)"
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
        default=SCRIPT_DIR / "results/nemotron_parse",
        help="Output directory"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Max pages per document (Nemotron only supports 1 image per call)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens to generate"
    )

    args = parser.parse_args()

    print("="*60)
    print("Nemotron Parse on CheckboxQA")
    print("="*60)
    print("WARNING: Nemotron Parse is NOT designed for checkbox QA.")
    print("This run is for documentation purposes only.")
    print("See nemotron_notes.md for details.")
    print("="*60 + "\n")

    model, processor, gen_config = load_model()

    run_evaluation(
        model, processor, gen_config,
        args.subset,
        args.output_dir,
        args.max_pages,
        args.max_tokens
    )


if __name__ == "__main__":
    main()
