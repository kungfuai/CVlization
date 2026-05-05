"""LLaTiSA: VLM-based time series reasoning with dual-view input.

Renders a time series as both a line plot and a precision-calibrated
numerical table, then feeds both images to a Vision Language Model for
chain-of-thought reasoning over the series.

Reference:
  Ding et al., "LLaTiSA: Towards Difficulty-Stratified Time Series
  Reasoning from Visual Perception to Semantics", ACL 2026 Findings.
  https://arxiv.org/abs/2604.17295
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Dual-view image generation
# ---------------------------------------------------------------------------

def render_line_plot(
    series: np.ndarray,
    out_path: str | Path,
    title: str = "Time Series",
    dpi: int = 150,
) -> Path:
    """Render a line plot of one or more univariate series.

    Args:
        series: Array of shape (T,) or (T, D) where T is time steps, D dimensions.
        out_path: Destination PNG path.
        title: Plot title.
        dpi: Resolution.

    Returns:
        The output path.
    """
    out_path = Path(out_path)
    series = np.asarray(series, dtype=float)
    if series.ndim == 1:
        series = series[:, None]
    T, D = series.shape

    fig, axes = plt.subplots(D, 1, figsize=(10, max(3 * D, 4)), squeeze=False)
    for i in range(D):
        ax = axes[i, 0]
        ax.plot(range(T), series[:, i], color="blue", linewidth=1.2)
        ax.set_xlabel("Time Step")
        ax.set_ylabel(f"Dim {i}" if D > 1 else "Value")
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_numeric_table(
    series: np.ndarray,
    out_path: str | Path,
    rows_per_block: int = 50,
    dpi: int = 150,
) -> Path:
    """Render a high-precision index-value table as an image.

    The table shows each time index alongside its value(s) formatted with
    full double-precision fidelity, mirroring the approach from the LLaTiSA
    paper that bridges qualitative visual perception with quantitative
    numerical precision.

    Args:
        series: Array of shape (T,) or (T, D).
        out_path: Destination PNG path.
        rows_per_block: Maximum rows per column block.
        dpi: Resolution.

    Returns:
        The output path.
    """
    out_path = Path(out_path)
    series = np.asarray(series, dtype=float)
    if series.ndim == 1:
        series = series[:, None]
    T, D = series.shape

    def fmt(v: float) -> str:
        return f"{v:.6g}"

    # Build text lines: "idx | v0 | v1 | ..."
    header_parts = ["Index"] + [f"Dim{d}" if D > 1 else "Value" for d in range(D)]
    header = "  ".join(f"{h:>12s}" for h in header_parts)
    lines = [header, "-" * len(header)]
    for t in range(T):
        parts = [f"{t:>12d}"] + [f"{fmt(series[t, d]):>12s}" for d in range(D)]
        lines.append("  ".join(parts))

    # Split into blocks for wide rendering
    n_blocks = max(1, (T + rows_per_block - 1) // rows_per_block)
    block_lines: list[list[str]] = []
    for b in range(n_blocks):
        start = b * rows_per_block
        end = min(start + rows_per_block, T)
        block = [lines[0], lines[1]] + lines[2 + start : 2 + end]
        block_lines.append(block)

    # Render as image using matplotlib text
    fig_width = 6 * n_blocks
    fig_height = max(4, 0.18 * (rows_per_block + 2))
    fig, axes = plt.subplots(
        1, n_blocks, figsize=(fig_width, fig_height), squeeze=False
    )
    for b, block in enumerate(block_lines):
        ax = axes[0, b]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(block))
        ax.axis("off")
        for row_idx, line in enumerate(block):
            y = len(block) - 1 - row_idx
            bg = "#f0f0f0" if row_idx % 2 == 0 else "#ffffff"
            ax.axhspan(y, y + 1, color=bg, zorder=0)
            ax.text(
                0.02,
                y + 0.5,
                line,
                fontsize=7,
                fontfamily="monospace",
                verticalalignment="center",
                zorder=1,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a time-series reasoning assistant. You are given two images of "
    "the same time series: a line plot showing visual patterns, and a "
    "numerical table with precise index-value pairs. Use both views to "
    "answer the question. Think step by step."
)

EXAMPLE_QUESTIONS = {
    "l1_minmax": (
        "What is the maximum value in this time series, and at which "
        "time step does it occur? Also report the minimum value and its "
        "time step."
    ),
    "l1_read": (
        "What is the value of this time series at time step {idx}?"
    ),
    "l2_trend": (
        "Describe the overall trend of this time series. Is it increasing, "
        "decreasing, stationary, or does it exhibit a more complex pattern? "
        "Identify any notable local patterns such as peaks, troughs, or "
        "change points."
    ),
    "l3_anomaly": (
        "Analyze this time series for anomalies. Are there any unusual "
        "spikes, drops, or deviations from the expected pattern? If so, "
        "at which time steps do they occur, and what might explain them?"
    ),
}


def build_messages(
    plot_path: Path,
    table_path: Path,
    question: str,
) -> list[dict]:
    """Build a multi-image chat message list for the VLM."""
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is the line plot of the time series:"},
                {"type": "image", "image": f"file://{plot_path.resolve()}"},
                {
                    "type": "text",
                    "text": (
                        "Here is the numerical table with precise index-value "
                        "pairs for the same series:"
                    ),
                },
                {"type": "image", "image": f"file://{table_path.resolve()}"},
                {"type": "text", "text": question},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# VLM inference
# ---------------------------------------------------------------------------

def run_inference(
    messages: list[dict],
    model_path: str,
    max_new_tokens: int = 1024,
    device: str = "auto",
) -> str:
    """Run inference with Qwen2.5-VL / Qwen3-VL."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    print(f"Loading model from {model_path} ...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_path)

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
    ).to(model.device)

    print("Generating response ...", flush=True)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # Trim the prompt tokens from the output
    generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_series_from_json(path: str) -> np.ndarray:
    """Load a time series from a JSON file.

    Accepts either a plain array ``[1.0, 2.0, ...]`` or a dict with a
    ``"timeseries"`` key (HiTSR format).
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        ts = data.get("timeseries", data.get("values", data.get("data")))
    else:
        ts = data
    return np.asarray(ts, dtype=float)


def load_series_from_csv(path: str, column: int | str = 1) -> np.ndarray:
    """Load a time series column from a CSV file."""
    import pandas as pd

    df = pd.read_csv(path)
    if isinstance(column, int):
        return df.iloc[:, column].values.astype(float)
    return df[column].values.astype(float)


def generate_synthetic_series(length: int = 200, seed: int = 42) -> np.ndarray:
    """Generate a synthetic time series with trend, seasonality, and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=float)
    trend = 0.02 * t
    seasonal = 2.0 * np.sin(2 * np.pi * t / 50)
    noise = rng.normal(0, 0.3, size=length)
    # Add an anomaly spike
    anomaly_idx = int(length * 0.7)
    noise[anomaly_idx] += 5.0
    return trend + seasonal + noise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLaTiSA: VLM time-series reasoning with dual-view input",
    )
    p.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input time series (JSON or CSV). Uses synthetic data if omitted.",
    )
    p.add_argument(
        "--csv-column",
        type=str,
        default="1",
        help="Column index (int) or name (str) for CSV input. Default: 1.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model ID or local path. Default: Qwen/Qwen2.5-VL-7B-Instruct.",
    )
    p.add_argument(
        "--question",
        type=str,
        default=None,
        help="Custom question to ask about the series. If omitted, a default question is used.",
    )
    p.add_argument(
        "--question-preset",
        type=str,
        choices=list(EXAMPLE_QUESTIONS.keys()),
        default="l2_trend",
        help="Preset question type. Ignored if --question is given.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum tokens in VLM response.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="./artifacts",
        help="Directory for output images and results.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device map for model loading (auto, cuda, cpu).",
    )
    p.add_argument(
        "--series-length",
        type=int,
        default=200,
        help="Length of synthetic series (only used when --input is omitted).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data.",
    )
    p.add_argument(
        "--images-only",
        action="store_true",
        help="Only generate dual-view images, skip VLM inference.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load or generate series ---
    if args.input is not None:
        input_path = args.input
        print(f"Loading time series from {input_path} ...", flush=True)
        if input_path.endswith(".csv"):
            try:
                col = int(args.csv_column)
            except ValueError:
                col = args.csv_column
            series = load_series_from_csv(input_path, column=col)
        else:
            series = load_series_from_json(input_path)
    else:
        print(
            f"No input provided; generating synthetic series "
            f"(length={args.series_length}, seed={args.seed}) ...",
            flush=True,
        )
        series = generate_synthetic_series(
            length=args.series_length, seed=args.seed
        )

    print(
        f"Series shape: {series.shape}, "
        f"range: [{series.min():.4f}, {series.max():.4f}]",
        flush=True,
    )

    # --- Generate dual-view images ---
    plot_path = render_line_plot(series, out_dir / "plot.png")
    table_path = render_numeric_table(series, out_dir / "numeric_table.png")
    print(f"Saved line plot:       {plot_path}", flush=True)
    print(f"Saved numeric table:   {table_path}", flush=True)

    if args.images_only:
        print("Images-only mode; skipping VLM inference.", flush=True)
        return

    # --- Build question ---
    if args.question:
        question = args.question
    else:
        question = EXAMPLE_QUESTIONS[args.question_preset]
        # Fill in template placeholders if any
        if "{idx}" in question:
            idx = len(series) // 2
            question = question.format(idx=idx)

    print(f"\nQuestion: {question}\n", flush=True)

    # --- Run VLM inference ---
    messages = build_messages(plot_path, table_path, question)
    response = run_inference(
        messages,
        model_path=args.model,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )

    print("=" * 60)
    print("RESPONSE:")
    print("=" * 60)
    print(response)
    print("=" * 60)

    # --- Save results ---
    result = {
        "model": args.model,
        "question": question,
        "response": response,
        "series_shape": list(series.shape),
        "series_range": [float(series.min()), float(series.max())],
    }
    result_path = out_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {result_path}", flush=True)


if __name__ == "__main__":
    main()
