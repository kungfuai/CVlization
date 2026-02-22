#!/usr/bin/env python3
"""
Zero-shot OMR probing with Qwen3-VL-8B.

Sends 5 music-theory prompts to Qwen3-VL-8B-Instruct on a single sheet-music
image and writes all responses to a structured output file.  This is the B2a
baseline (zero-shot large VLM) in the Bucket 3 comparison matrix.

Expected results:
  - Key signature:    good (coarse visual question)
  - Time signature:   good
  - Musical era:      moderate
  - Dynamics/tempo:   moderate
  - Full ekern:       will fail / hallucinate (transcription needs fine-tuning)

Usage:
  python predict.py
  python predict.py --image my_score.jpg
  python predict.py --model Qwen/Qwen3-VL-2B-Instruct --no-quantize
  python predict.py --image score.png --output result.txt
"""

import argparse
import os
import sys
from pathlib import Path

# Silence noisy libraries before heavy imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

try:
    from cvlization.paths import resolve_input_path, resolve_output_path
    CVL_AVAILABLE = True
except ImportError:
    CVL_AVAILABLE = False

    def resolve_input_path(p):
        return p

    def resolve_output_path(p):
        return p


# ---------------------------------------------------------------------------
# Prompts — ordered from most likely to succeed to least likely
# ---------------------------------------------------------------------------
PROMPTS = [
    {
        "id": "key_signature",
        "label": "Key signature",
        "text": (
            "Look at this sheet music image. What is the key signature? "
            "Count the sharps or flats shown at the beginning of each staff line "
            "and name the major or minor key (e.g. 'G major', 'D minor', 'no sharps or flats — C major')."
        ),
        "expectation": "Expected to work well — key signature is a clear visual feature.",
    },
    {
        "id": "time_signature",
        "label": "Time signature",
        "text": (
            "Look at this sheet music image. What is the time signature? "
            "Read the two stacked numbers shown at the start of the piece "
            "(e.g. 4/4, 3/4, 6/8) and explain what it means rhythmically."
        ),
        "expectation": "Expected to work well — time signature digits are visually salient.",
    },
    {
        "id": "musical_era",
        "label": "Musical era",
        "text": (
            "Looking at the engraving style, paper quality, typography, and notation conventions "
            "in this sheet music image, what musical era or approximate time period does it appear "
            "to be from? Give your best estimate (e.g. Baroque, Classical, Romantic, early 20th century)."
        ),
        "expectation": "Moderate — depends on whether the image has clear period-specific features.",
    },
    {
        "id": "dynamics_tempo",
        "label": "Dynamics and tempo markings",
        "text": (
            "List all dynamics markings (e.g. p, f, mf, pp, ff, crescendo, decrescendo) and "
            "tempo / expression markings (e.g. Allegro, Andante, poco a poco, dolce) visible "
            "in this sheet music image. For each, note approximately where it appears."
        ),
        "expectation": "Moderate — depends on image resolution and marking visibility.",
    },
    {
        "id": "ekern_transcription",
        "label": "Full ekern transcription",
        "text": (
            "Transcribe this piano sheet music image into ekern notation (**ekern_1.0). "
            "Ekern is a two-column tab-separated format: left column = bass clef staff, "
            "right column = treble clef staff. Each row is a beat position. "
            "Use standard kern tokens: note durations (1, 2, 4, 8, 16), pitch (C, D, E, F, G, A, B "
            "with octave suffix), rests (r), barlines (=), clef (*clefG2, *clefF4), "
            "key (*k[b-]), time (*M4/4), voice beaming (L=beam start, J=beam end). "
            "Example first line: **ekern_1.0\t**ekern_1.0\n"
            "Transcribe as many measures as you can read."
        ),
        "expectation": (
            "Expected to fail / hallucinate — full symbolic transcription requires fine-tuning. "
            "Included as a lower-bound probe to show what zero-shot cannot do."
        ),
    },
]

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_OUTPUT = "omr_probe_output.txt"
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "cvl_data" / "qwen3_omr"

# Vintage sample: Biddle's Piano Waltz (1884), Library of Congress public domain
# Cover page shows ornate Victorian typography and visible aging (good for era prompt)
# Score page shows grand staff notation with dynamics (good for OMR prompts)
SAMPLE_IMAGES = {
    "score": "qwen3_omr/vintage_score_1884.jpg",   # music notation page (default)
    "cover": "qwen3_omr/vintage_cover_1884.jpg",   # decorative cover page
}


def ensure_sample_image(variant: str = "score") -> Path:
    """Download the vintage sample image from HuggingFace (zzsi/cvl).

    variant: "score" (music notation page, default) or "cover" (decorative title page)
    Source: Biddle's Piano Waltz (Robert D. Biddle, 1884), Library of Congress.
    Public domain. High-resolution scan (~2289×2878 px at 50% of original).
    """
    filename = SAMPLE_IMAGES[variant]
    cache_path = CACHE_DIR / Path(filename).name
    if cache_path.exists():
        return cache_path
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading vintage sample ({variant}) from HuggingFace (zzsi/cvl) ...")
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(
        repo_id="zzsi/cvl",
        repo_type="dataset",
        filename=filename,
        local_dir=str(CACHE_DIR.parent),
    )
    return Path(local)


def load_model(model_id: str, quantize: bool):
    print(f"Loading {model_id} (4-bit quantization: {quantize}) ...")
    if quantize and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    elif torch.cuda.is_available():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


def run_prompt(model, processor, image: Image.Image, prompt_text: str, max_new_tokens: int = 512) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    # Move inputs to model device (works with device_map="auto" too)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Strip the prompt tokens before decoding
    trimmed = generated[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return response.strip()


def format_output(image_path: str, model_id: str, results: list[dict]) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("Qwen3-VL Zero-Shot OMR Probe")
    lines.append("=" * 70)
    lines.append(f"Model : {model_id}")
    lines.append(f"Image : {image_path}")
    lines.append("")

    for r in results:
        lines.append("-" * 70)
        lines.append(f"[{r['id']}] {r['label']}")
        lines.append(f"Expectation: {r['expectation']}")
        lines.append("")
        lines.append("Prompt:")
        lines.append(r["prompt"])
        lines.append("")
        lines.append("Response:")
        lines.append(r["response"])
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot OMR probing with Qwen3-VL")
    parser.add_argument(
        "--image",
        default=None,
        help="Path to a sheet music image. If omitted, auto-downloads the vintage sample.",
    )
    parser.add_argument(
        "--sample-cover",
        action="store_true",
        help="Use the decorative cover page instead of the music notation page as the default sample.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable 4-bit quantization (requires more VRAM; use bf16 on GPU or fp32 on CPU)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens per response (default: 512)",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        choices=[p["id"] for p in PROMPTS],
        default=None,
        help="Run only specific prompts by ID (default: all 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve image path
    if args.image:
        image_path = resolve_input_path(args.image)
    else:
        variant = "cover" if args.sample_cover else "score"
        image_path = str(ensure_sample_image(variant))
        print(f"Using vintage sample ({variant}): {image_path}")
        print("  Source: Biddle's Piano Waltz (Robert D. Biddle, 1884), Library of Congress")

    # Resolve output path
    output_path = Path(resolve_output_path(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"Image loaded: {image.size[0]}x{image.size[1]} px")

    # Select prompts
    selected_prompts = PROMPTS
    if args.prompts:
        selected_prompts = [p for p in PROMPTS if p["id"] in args.prompts]

    # Load model
    quantize = not args.no_quantize
    model, processor = load_model(args.model, quantize)

    # Run each prompt
    results = []
    for i, prompt in enumerate(selected_prompts):
        print(f"\n[{i+1}/{len(selected_prompts)}] {prompt['label']} ...")
        response = run_prompt(model, processor, image, prompt["text"], args.max_new_tokens)
        results.append({
            "id": prompt["id"],
            "label": prompt["label"],
            "expectation": prompt["expectation"],
            "prompt": prompt["text"],
            "response": response,
        })
        print(f"  → {response[:120]}{'...' if len(response) > 120 else ''}")

    # Write output
    output_text = format_output(image_path, args.model, results)
    output_path.write_text(output_text, encoding="utf-8")
    print(f"\nFull output saved to: {output_path}")
    print("\n" + output_text)


if __name__ == "__main__":
    main()
