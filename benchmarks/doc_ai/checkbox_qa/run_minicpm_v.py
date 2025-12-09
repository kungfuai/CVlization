#!/usr/bin/env python3
"""
Run MiniCPM-V-2.6 on CheckboxQA benchmark.

MiniCPM-V-2.6 is OpenBMB's efficient 8B multimodal model with
strong OCR and document understanding capabilities.

Model: openbmb/MiniCPM-V-2_6
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

MODEL_ID = "openbmb/MiniCPM-V-2_6"


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


def load_model(device: str = "cuda"):
    """Load MiniCPM-V-2.6 model."""
    print(f"Loading model: {MODEL_ID}...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16
    )

    if device == "cuda":
        model = model.eval().cuda()
    else:
        model = model.eval()

    print("Model loaded.")
    return model, tokenizer


PROMPT_STYLES = {
    "default": "Look at this document and answer the following question. Answer concisely.\n\nQuestion: {question}\n\nAnswer:",
    "checkbox": "This is a form with checkboxes. Look carefully at the checkboxes and their labels. Answer the following question about which checkbox(es) are checked/selected. Be brief and specific.\n\nQuestion: {question}\n\nAnswer:",
    "ocr_checkbox": "Extract all checkbox states from this document. For the following question, identify which option(s) are marked/checked.\n\nQuestion: {question}\n\nAnswer:",
    "direct": "{question}",
}


def ask_question(
    model,
    tokenizer,
    image_paths: List[Path],
    question: str,
    max_tokens: int = 256,
    max_image_size: int = None,
    prompt_style: str = "default"
) -> str:
    """Ask a VQA question about document images."""

    # Load images
    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        if max_image_size:
            # Resize if larger than max
            w, h = img.size
            if max(w, h) > max_image_size:
                scale = max_image_size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
        images.append(img)

    # Build prompt
    num_pages = len(images)
    if num_pages > 1:
        page_info = f"This document has {num_pages} pages. "
    else:
        page_info = ""

    template = PROMPT_STYLES.get(prompt_style, PROMPT_STYLES["default"])
    prompt = page_info + template.format(question=question)

    # Build message with images
    content = images + [prompt]
    msgs = [{'role': 'user', 'content': content}]

    # Run inference
    with torch.no_grad():
        response = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer
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
    max_pages: int = 1,
    max_image_size: int = None,
    prompt_style: str = "default"
):
    """Run MiniCPM-V VQA on CheckboxQA."""

    output_dir.mkdir(parents=True, exist_ok=True)

    docs = load_subset(subset_path)
    print(f"Loaded {len(docs)} documents")
    print(f"Prompt style: {prompt_style}")
    if max_image_size:
        print(f"Max image size: {max_image_size}")

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

        print(f"  Using {len(page_images)} page(s)")

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
            answer = ask_question(
                model, tokenizer, page_images, question, max_tokens,
                max_image_size=max_image_size, prompt_style=prompt_style
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
        print("RESULTS (MiniCPM-V-2.6 on CheckboxQA)")
        print(f"{'='*60}")
        print(f"ANLS* Score: {results['anls_score']:.4f}")
        print(f"Total Questions: {results['total_questions']}")
        print(f"{'='*60}")

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Run MiniCPM-V-2.6 VQA on CheckboxQA"
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
        default=SCRIPT_DIR / "results/minicpm_v_dev",
        help="Output directory"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens to generate"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Max pages per document"
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=None,
        help="Max image size (resize if larger)"
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="default",
        choices=list(PROMPT_STYLES.keys()),
        help="Prompt style for checkbox questions"
    )

    args = parser.parse_args()

    print("="*60)
    print("MiniCPM-V-2.6 VQA on CheckboxQA")
    print("="*60)
    print(f"Model: {MODEL_ID}")
    print(f"Max pages: {args.max_pages}")
    print("="*60 + "\n")

    model, tokenizer = load_model()

    run_evaluation(
        model, tokenizer,
        args.subset,
        args.output_dir,
        args.max_tokens,
        args.max_pages,
        args.max_image_size,
        args.prompt_style
    )


if __name__ == "__main__":
    main()
