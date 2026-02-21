#!/usr/bin/env python3
"""
Sheet Music Transformer (SMT) — Optical Music Recognition for piano scores.

Transcribes scanned sheet music images into bekern notation, a structured
token encoding that captures pitches, durations, voices, dynamics, and structure.
The output can be converted to MusicXML or rendered to SVG via Verovio.

Model: PRAIG/smt-grandstaff (default — system-level piano, matches current SMT codebase)
Repository: https://github.com/antoniorv6/SMT
Paper: https://arxiv.org/abs/2402.07596
License: MIT
"""

import os
import sys
import logging
import warnings

# Suppress verbose logging by default (before heavy imports)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
for _logger_name in ["transformers", "torch", "lightning", "loguru"]:
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

# CVL dual-mode execution support
try:
    from cvlization.paths import (
        get_input_dir,
        get_output_dir,
        resolve_input_path,
        resolve_output_path,
    )
    CVL_AVAILABLE = True
except ImportError:
    CVL_AVAILABLE = False

    def get_input_dir():
        return os.getcwd()

    def get_output_dir():
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def resolve_input_path(path, base_dir):
        return path if os.path.isabs(path) else os.path.join(base_dir, path)

    def resolve_output_path(path, base_dir):
        return path if os.path.isabs(path) else os.path.join(base_dir, path)


DEFAULT_MODEL = "PRAIG/smt-grandstaff"
# CACHE_DIR parent is used as local_dir for hf_hub_download so that the file
# lands at CACHE_DIR/sample_score.jpg (hf_hub_download preserves filename subdirs)
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "cvl_data" / "smt_omr"


def ensure_sample_data() -> Path:
    """
    Download a sample piano score image for testing.

    Tries zzsi/cvl first, then falls back to the antoniorv6/grandstaff test split.
    Caches locally so subsequent calls are instant.
    """
    cache_path = CACHE_DIR / "sample_score.jpg"
    if cache_path.exists():
        return cache_path

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Try zzsi/cvl (uploaded sample for this example).
    # Use CACHE_DIR.parent as local_dir so that filename="smt_omr/sample_score.jpg"
    # resolves to CACHE_DIR/sample_score.jpg on disk.
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id="zzsi/cvl",
            repo_type="dataset",
            filename="smt_omr/sample_score.jpg",
            local_dir=str(CACHE_DIR.parent),
        )
        return Path(local)
    except Exception:
        pass

    # Fallback: pull first image from antoniorv6/grandstaff test split
    print("Fetching sample from antoniorv6/grandstaff test split...")
    import datasets as hf_datasets
    ds = hf_datasets.load_dataset(
        "antoniorv6/grandstaff", split="test[:1]", trust_remote_code=False
    )
    sample = ds[0]
    img_array = np.array(sample["image"])
    cv2.imwrite(str(cache_path), img_array)
    print(f"Sample cached at: {cache_path}")
    return cache_path


def ensure_sample_gt() -> Path:
    """
    Download the ground truth for the sample piano score image.

    Fetches smt_omr/sample_score_gt.txt from zzsi/cvl on HuggingFace.
    Caches locally so subsequent calls are instant.
    """
    cache_path = CACHE_DIR / "sample_score_gt.txt"
    if cache_path.exists():
        return cache_path

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id="zzsi/cvl",
            repo_type="dataset",
            filename="smt_omr/sample_score_gt.txt",
            local_dir=str(CACHE_DIR.parent),
        )
        return Path(local)
    except Exception as e:
        print(f"Warning: could not download sample ground truth: {e}")
        return None


def detect_device() -> tuple:
    """Auto-detect device and appropriate dtype."""
    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        return "cuda", dtype
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    else:
        return "cpu", torch.float32


def load_model(model_id: str, device: str):
    """Load SMT model from HuggingFace."""
    from smt_model import SMTModelForCausalLM
    print(f"Loading {model_id}...")
    model = SMTModelForCausalLM.from_pretrained(model_id)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model


def load_image(image_path: str) -> np.ndarray:
    """Load image with OpenCV. Raises FileNotFoundError if not found."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    return img


def run_inference(model, image: np.ndarray, device: str) -> str:
    """
    Run SMT OMR on a score image.

    Args:
        model: Loaded SMTModelForCausalLM
        image: BGR numpy array (as returned by cv2.imread)
        device: "cuda", "mps", or "cpu"

    Returns:
        Bekern notation string with human-readable newlines/tabs substituted.
    """
    from data_augmentation.data_augmentation import convert_img_to_tensor
    tensor = convert_img_to_tensor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions, _ = model.predict(tensor, convert_to_str=True)
    bekern = (
        "".join(predictions)
        .replace("<b>", "\n")
        .replace("<s>", " ")
        .replace("<t>", "\t")
    )
    return bekern


# ---------------------------------------------------------------------------
# Metrics (CER / SER / LER) — inlined from SMT eval_functions + utils
# ---------------------------------------------------------------------------

def _levenshtein(a, b):
    """Levenshtein edit distance between two sequences."""
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n
    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add = previous[j] + 1
            delete = current[j - 1] + 1
            change = previous[j - 1] + (0 if a[j - 1] == b[i - 1] else 1)
            current[j] = min(add, delete, change)
    return current[n]


def _normalize_bekern(s: str) -> str:
    """Strip extra whitespace around tab separators and line boundaries."""
    lines = []
    for line in s.strip().split("\n"):
        cols = [col.strip() for col in line.split("\t")]
        lines.append("\t".join(cols))
    return "\n".join(lines)


def compute_metrics(prediction: str, ground_truth: str) -> dict:
    """
    Compute CER, SER, and LER between a prediction and ground truth bekern string.

    Metrics match those reported in the SMT paper (Levenshtein-based):
      CER  — Character Error Rate   (character-level)
      SER  — Symbol Error Rate      (space-separated token level)
      LER  — Line Error Rate        (line-level, averaged over lines)
    """
    pred = _normalize_bekern(prediction)
    gt = _normalize_bekern(ground_truth)

    # CER: character-level edit distance over the full string
    h_chars = list(pred)
    g_chars = list(gt)
    cer = 100.0 * _levenshtein(h_chars, g_chars) / len(g_chars) if g_chars else 0.0

    # SER: token-level; tabs/newlines become explicit tokens
    def _ser_tokens(s):
        s = s.replace("\n", " <b> ").replace("\t", " <t> ")
        return [t for t in s.split(" ") if t]

    h_ser = _ser_tokens(pred)
    g_ser = _ser_tokens(gt)
    ser = 100.0 * _levenshtein(h_ser, g_ser) / len(g_ser) if g_ser else 0.0

    # LER: per-line Levenshtein (token-level within each line), averaged
    def _ler_tokens(line):
        return [t for t in line.replace("\t", " <t> ").split(" ") if t]

    h_lines = [_ler_tokens(l) for l in pred.split("\n")]
    g_lines = [_ler_tokens(l) for l in gt.split("\n")]
    ed_total = sum(_levenshtein(h, g) for h, g in zip(h_lines, g_lines))
    len_total = sum(len(g) for g in g_lines)
    ler = 100.0 * ed_total / len_total if len_total > 0 else 0.0

    return {"CER": round(cer, 2), "SER": round(ser, 2), "LER": round(ler, 2)}


def save_output(
    bekern: str,
    output_path: str,
    fmt: str = "txt",
    metadata: dict = None,
):
    """Save bekern notation to file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        data = {"bekern": bekern, **(metadata or {})}
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        out.write_text(bekern)


def main():
    parser = argparse.ArgumentParser(
        description="SMT: Optical Music Recognition for piano sheet music",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use auto-downloaded sample image + ground truth (computes CER/SER/LER)
  python predict.py

  # Transcribe a vintage scan (no ground truth)
  python predict.py --image vintage_score.jpg

  # Evaluate against a ground truth file
  python predict.py --image score.jpg --ground-truth score_gt.txt

  # Output as JSON with metadata and metrics
  python predict.py --format json --output result.json
        """,
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input score image (default: auto-download sample from HuggingFace)",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        dest="ground_truth",
        help=(
            "Path to ground truth bekern file for accuracy evaluation. "
            "When omitted and --image is also omitted, the sample ground truth "
            "is auto-downloaded and CER/SER/LER are computed automatically."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: outputs/result.{format})",
    )
    parser.add_argument(
        "--format",
        choices=["txt", "json"],
        default="txt",
        help="Output format (default: txt)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=(
            f"HuggingFace model ID (default: {DEFAULT_MODEL}). "
            "Alternatives: PRAIG/smt-fp-grandstaff (full-page), "
            "PRAIG/smt-fp-mozarteum, PRAIG/smt-fp-polish-scores"
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)
        for _n in ["transformers", "torch"]:
            logging.getLogger(_n).setLevel(logging.INFO)

    INP = get_input_dir()
    OUT = get_output_dir()

    # Resolve image path; when using sample data also fetch ground truth
    using_sample = args.image is None
    if using_sample:
        image_path = str(ensure_sample_data())
        if args.ground_truth is None:
            gt_file = ensure_sample_gt()
            gt_path = str(gt_file) if gt_file else None
        else:
            gt_path = resolve_input_path(args.ground_truth, INP)
    else:
        image_path = resolve_input_path(args.image, INP)
        gt_path = resolve_input_path(args.ground_truth, INP) if args.ground_truth else None

    if not Path(image_path).exists():
        print(f"Error: image not found: {image_path}")
        return 1

    ext = "json" if args.format == "json" else "txt"
    output_path = resolve_output_path(args.output or f"result.{ext}", OUT)

    device, _ = (args.device, None) if args.device else detect_device()

    print(f"\n{'='*60}")
    print("Sheet Music Transformer — OMR")
    print("="*60)
    print(f"  Image:        {image_path}")
    print(f"  Model:        {args.model}")
    print(f"  Device:       {device}")
    print(f"  Output:       {output_path}")
    if gt_path:
        print(f"  Ground truth: {gt_path}")
    print("="*60 + "\n")

    # Load model
    try:
        model = load_model(args.model, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Load image
    try:
        image = load_image(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return 1

    print(f"Image size: {image.shape[1]}x{image.shape[0]} (WxH)")

    # Run inference
    try:
        print("Running OMR inference...")
        bekern = run_inference(model, image, device)
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print preview
    print("\n" + "="*60)
    print("BEKERN OUTPUT (preview):")
    print("="*60)
    preview = bekern[:600] + ("..." if len(bekern) > 600 else "")
    print(preview)
    print("="*60 + "\n")

    # Compute accuracy metrics when ground truth is available
    metrics = None
    if gt_path and Path(gt_path).exists():
        ground_truth = Path(gt_path).read_text()
        metrics = compute_metrics(bekern, ground_truth)
        print("="*60)
        print("ACCURACY METRICS (vs ground truth):")
        print("="*60)
        print(f"  CER (Character Error Rate): {metrics['CER']:.2f}%")
        print(f"  SER (Symbol Error Rate):    {metrics['SER']:.2f}%")
        print(f"  LER (Line Error Rate):      {metrics['LER']:.2f}%")
        print("="*60 + "\n")
    elif gt_path:
        print(f"Warning: ground truth file not found: {gt_path}")

    # Save output
    metadata = {
        "model": args.model,
        "image": image_path,
        "image_size": [image.shape[1], image.shape[0]],
    }
    if metrics:
        metadata["metrics"] = metrics
    save_output(bekern, output_path, args.format, metadata)
    print(f"Output saved to: {output_path}")
    print(f"Output length: {len(bekern)} characters")
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
