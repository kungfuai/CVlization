#!/usr/bin/env python3
"""
Run DeepSeek-OCR on CheckboxQA benchmark with VQA prompts.

DeepSeek-OCR is primarily an OCR model but supports custom prompts.
Testing if it can handle VQA-style questions about documents.

Model: deepseek-ai/DeepSeek-OCR
- 12GB VRAM
- Uses model.infer() API
"""

import argparse
import json
import os
import sys
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

MODEL_ID = "deepseek-ai/DeepSeek-OCR"


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


def load_model():
    """Load DeepSeek-OCR model."""
    print(f"Loading model: {MODEL_ID}...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )

    model = model.eval()
    print("Model loaded.")
    return model, tokenizer


def ask_question(
    model,
    tokenizer,
    image_path: Path,
    question: str,
) -> str:
    """Ask a VQA question about the document image using DeepSeek-OCR."""

    # VQA prompt - use MiniCPM's "default" prompt style that scored best
    prompt = f"<image>\nLook at this document and answer the following question. Answer concisely.\n\nQuestion: {question}\n\nAnswer:"

    # Create temp output dir
    temp_output_dir = "/tmp/deepseek_ocr_output"
    os.makedirs(temp_output_dir, exist_ok=True)

    # Capture stdout since model.infer() prints output
    captured_output = StringIO()
    original_stdout = sys.stdout
    sys.stdout = captured_output

    try:
        res = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=str(image_path),
            output_path=temp_output_dir,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=False,
        )
    finally:
        sys.stdout = original_stdout

    # Get captured output
    stdout_text = captured_output.getvalue()

    # Parse output - remove debug info
    if "===============save results:===============" in stdout_text:
        output = stdout_text.split("===============save results:===============")[0].strip()
    else:
        output = stdout_text.strip()

    # Filter debug lines
    lines = output.split('\n')
    clean_lines = []
    for line in lines:
        if any(x in line for x in ['torch.Size', '=====', 'BASE:', 'PATCHES:', 'Loading']):
            continue
        clean_lines.append(line)

    output = '\n'.join(clean_lines).strip()

    # If empty, try return value
    if not output and res:
        output = str(res)

    return output.strip() if output else ""


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
    max_pages: int = 1,
):
    """Run DeepSeek-OCR VQA on CheckboxQA."""

    output_dir.mkdir(parents=True, exist_ok=True)

    docs = load_subset(subset_path)
    print(f"Loaded {len(docs)} documents")

    predictions = []
    page_cache_root = get_page_cache_root()

    for doc_idx, doc in enumerate(docs):
        doc_id = doc["name"]
        print(f"\n[{doc_idx+1}/{len(docs)}] Processing {doc_id}...")

        # Get page images (single image only for DeepSeek-OCR)
        page_images = get_page_images(doc_id, max_pages=1, page_cache_root=page_cache_root)
        if not page_images:
            print(f"  Warning: No pages found for {doc_id}")
            continue

        print(f"  Using 1 page (DeepSeek-OCR is single-image)")

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
            answer = ask_question(model, tokenizer, page_images[0], question)
            print(f"  A: {answer[:80] if answer else '(empty)'}")

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
        print("RESULTS (DeepSeek-OCR on CheckboxQA)")
        print(f"{'='*60}")
        print(f"ANLS* Score: {results['anls_score']:.4f}")
        print(f"Total Questions: {results['total_questions']}")
        print(f"{'='*60}")

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Run DeepSeek-OCR VQA on CheckboxQA"
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
        default=SCRIPT_DIR / "results/deepseek_ocr_dev",
        help="Output directory"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Max pages (DeepSeek-OCR only supports 1)"
    )

    args = parser.parse_args()

    print("="*60)
    print("DeepSeek-OCR VQA on CheckboxQA")
    print("="*60)
    print("Note: Testing OCR model with VQA prompts")
    print("="*60 + "\n")

    model, tokenizer = load_model()

    run_evaluation(
        model, tokenizer,
        args.subset,
        args.output_dir,
        args.max_pages,
    )


if __name__ == "__main__":
    main()
