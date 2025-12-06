#!/usr/bin/env python3
"""
Quick test script for Nemotron Parse on CheckboxQA documents.

Explores:
1. Default document parsing prompt
2. Custom prompts to extract checkboxes
3. Constrained generation options

Usage:
    python test_nemotron_parse.py --image path/to/page.png
    python test_nemotron_parse.py  # Uses first CheckboxQA dev page
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, GenerationConfig

# Add repo root for imports
sys.path.insert(0, str(Path(__file__).parent))
from postprocessing_nemotron import extract_classes_bboxes, postprocess_text

MODEL_ID = "nvidia/NVIDIA-Nemotron-Parse-v1.1"

# Different prompts to try
PROMPTS = {
    # Default: full document parsing with bboxes and classes
    "default": "</s><s><predict_bbox><predict_classes><output_markdown>",

    # No text output, just bboxes and classes (faster, structured)
    "no_text": "</s><s><predict_bbox><predict_classes><output_no_text>",

    # Just markdown, no bboxes/classes
    "markdown_only": "</s><s><output_markdown>",
}

# Page cache location
PAGE_CACHE = Path.home() / ".cache/cvlization/data/checkbox_qa/page_images"


def load_model(device: str = "cuda"):
    """Load Nemotron Parse model and processor."""
    print(f"Loading model: {MODEL_ID}...")

    # Use float16 for better compatibility (bfloat16 can cause dtype mismatches)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

    print(f"Model loaded on {device}")
    return model, processor, generation_config


def run_inference(model, processor, generation_config, image: Image.Image, prompt: str, max_new_tokens: int = 4096):
    """Run inference with a given prompt."""
    inputs = processor(images=[image], text=prompt, return_tensors="pt")
    # Move to model device and convert pixel_values to model dtype
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens
        )

    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return generated_text


def extract_checkbox_patterns(text: str) -> list:
    """
    Extract checkbox-like patterns from text.

    Returns list of (pattern, context) tuples.
    """
    patterns = [
        # Unicode checkboxes
        (r'☐', 'unchecked_unicode'),
        (r'☑', 'checked_unicode'),
        (r'✓', 'checkmark'),
        (r'✗', 'x_mark'),
        # ASCII checkboxes
        (r'\[[ ]\]', 'unchecked_ascii'),
        (r'\[[xX]\]', 'checked_ascii'),
        (r'\([ ]\)', 'unchecked_paren'),
        (r'\([xX]\)', 'checked_paren'),
        # Radio buttons
        (r'○', 'radio_unchecked'),
        (r'●', 'radio_checked'),
        (r'◯', 'circle_unchecked'),
        (r'◉', 'circle_checked'),
        # LaTeX-style (Nemotron outputs these)
        (r'\\square', 'latex_square'),
        (r'\\Box', 'latex_box'),
        (r'\\checkbox', 'latex_checkbox'),
        (r'\\boxtimes', 'latex_checked'),
        (r'\\checkmark', 'latex_checkmark'),
        # Common OCR outputs
        (r'\bYes\s*□', 'yes_checkbox'),
        (r'\bNo\s*□', 'no_checkbox'),
        (r'□\s*Yes', 'checkbox_yes'),
        (r'□\s*No', 'checkbox_no'),
    ]

    found = []
    for pattern, name in patterns:
        matches = list(re.finditer(pattern, text))
        for m in matches:
            # Get surrounding context (50 chars before/after)
            start = max(0, m.start() - 50)
            end = min(len(text), m.end() + 50)
            context = text[start:end].replace('\n', ' ')
            found.append({
                'type': name,
                'match': m.group(),
                'position': m.start(),
                'context': context
            })

    return found


def main():
    parser = argparse.ArgumentParser(description="Test Nemotron Parse on CheckboxQA")
    parser.add_argument("--image", type=Path, help="Path to image file")
    parser.add_argument("--doc-id", type=str, default="016f73a5", help="CheckboxQA doc ID")
    parser.add_argument("--page", type=int, default=1, help="Page number (1-indexed)")
    parser.add_argument("--prompt", type=str, choices=list(PROMPTS.keys()), default="default",
                        help="Prompt type to use")
    parser.add_argument("--all-prompts", action="store_true", help="Try all prompts")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens to generate")

    args = parser.parse_args()

    # Resolve image path
    if args.image:
        image_path = args.image
    else:
        image_path = PAGE_CACHE / args.doc_id / f"page-{args.page:03d}.png"

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return 1

    print(f"Image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"Size: {image.size}")

    # Load model
    model, processor, generation_config = load_model(args.device)

    # Determine prompts to try
    prompts_to_try = list(PROMPTS.items()) if args.all_prompts else [(args.prompt, PROMPTS[args.prompt])]

    for prompt_name, prompt in prompts_to_try:
        print(f"\n{'='*80}")
        print(f"PROMPT: {prompt_name}")
        print(f"{'='*80}")
        print(f"Prompt string: {prompt}")

        # Run inference
        print("\nRunning inference...")
        raw_output = run_inference(model, processor, generation_config, image, prompt, args.max_tokens)

        # Parse output
        print(f"\n--- Raw output (first 2000 chars) ---")
        print(raw_output[:2000])
        if len(raw_output) > 2000:
            print(f"... ({len(raw_output)} total chars)")

        # Extract structured data if using default prompt
        if "predict_bbox" in prompt and "predict_classes" in prompt:
            print(f"\n--- Parsed structure ---")
            try:
                classes, bboxes, texts = extract_classes_bboxes(raw_output)
                print(f"Elements found: {len(classes)}")
                print(f"Classes: {set(classes)}")

                # Show first few elements
                for i, (cls, bbox, text) in enumerate(zip(classes[:5], bboxes[:5], texts[:5])):
                    text_preview = text[:100].replace('\n', ' ')
                    print(f"  [{i}] {cls}: {text_preview}...")
            except Exception as e:
                print(f"Parse error: {e}")

        # Look for checkbox patterns
        print(f"\n--- Checkbox patterns found ---")
        checkboxes = extract_checkbox_patterns(raw_output)
        if checkboxes:
            for cb in checkboxes[:10]:
                print(f"  {cb['type']}: '{cb['match']}' -> ...{cb['context']}...")
        else:
            print("  No checkbox patterns found")

    # Constrained generation notes
    print(f"\n{'='*80}")
    print("CONSTRAINED GENERATION OPTIONS")
    print("="*80)
    print("""
Nemotron Parse doesn't natively support constrained/structured output for VQA.
The model outputs a fixed format: <x_><y_>text<x_><y_><class_>

Options for checkbox extraction:

1. POST-PROCESSING (current approach):
   - Parse the markdown output
   - Regex search for checkbox patterns
   - Use bbox proximity to associate checkboxes with labels

2. vLLM + OUTLINES (experimental):
   - vLLM supports structured output via outlines library
   - But requires model to be trained for JSON output
   - Nemotron isn't trained this way

3. TWO-STAGE PIPELINE:
   - Stage 1: Nemotron Parse → markdown + bboxes
   - Stage 2: LLM (Claude/GPT) answers questions from markdown
   - More accurate but higher latency/cost

4. FINE-TUNING (not recommended):
   - Fine-tune Nemotron for checkbox-specific output
   - Requires training data and compute
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
