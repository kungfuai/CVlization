from __future__ import annotations

import argparse
import json
import re
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from scipy.signal import resample

from cvlization.dataset.sleep_edf import SleepEDFBuilder


def parse_front_matter(spec_path: Path) -> Tuple[Dict[str, Any], str]:
    text = spec_path.read_text()
    matches = list(re.finditer(r"^---\s*$", text, re.MULTILINE))
    if len(matches) < 2:
        raise ValueError(
            f"Spec must begin with YAML front matter enclosed by '---' lines (found {len(matches)} markers)"
        )
    start = matches[0].end()
    end = matches[1].start()
    yaml_block = text[start:end].strip()
    metadata = yaml.safe_load(yaml_block) or {}
    body = text[matches[1].end() :]
    return metadata, body


def compute_stats(signals: np.ndarray, clip_threshold: float | None) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "channels": signals.shape[0],
        "samples": int(signals.shape[1]),
        "mean": signals.mean(axis=1).tolist(),
        "std": signals.std(axis=1).tolist(),
    }
    if clip_threshold is not None:
        over_limit = np.abs(signals) > clip_threshold
        stats["clip_fraction"] = float(over_limit.sum() / over_limit.size)
    return stats


def resample_signals(signals: np.ndarray, orig_hz: float, target_hz: Optional[float]) -> Tuple[np.ndarray, float]:
    if not target_hz or np.isclose(target_hz, orig_hz):
        return signals, orig_hz
    new_n = int(round(signals.shape[1] * target_hz / orig_hz))
    if new_n <= 0:
        raise ValueError("Resample target produced zero samples; check target_hz")
    resampled = resample(signals, new_n, axis=1)
    return resampled, target_hz


def normalize_signals(signals: np.ndarray, mode: Optional[str]) -> np.ndarray:
    if not mode:
        return signals
    mode = mode.lower()
    if mode == "zscore_per_channel":
        mean = signals.mean(axis=1, keepdims=True)
        std = signals.std(axis=1, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return (signals - mean) / std
    if mode == "minmax_per_channel":
        min_v = signals.min(axis=1, keepdims=True)
        max_v = signals.max(axis=1, keepdims=True)
        denom = np.where(np.abs(max_v - min_v) < 1e-6, 1.0, max_v - min_v)
        return (signals - min_v) / denom
    return signals


def window_signals(
    signals: np.ndarray,
    sampling_hz: float,
    length_seconds: Optional[float],
    overlap_seconds: Optional[float],
) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    if not length_seconds or length_seconds <= 0:
        return windows
    win_samples = int(round(length_seconds * sampling_hz))
    if win_samples <= 0 or win_samples > signals.shape[1]:
        return windows
    overlap = overlap_seconds or 0.0
    step = max(1, win_samples - int(round(overlap * sampling_hz)))
    for start in range(0, signals.shape[1] - win_samples + 1, step):
        end = start + win_samples
        windows.append(
            {
                "window_index": len(windows),
                "start_sample": start,
                "end_sample": end,
                "start_time_sec": start / sampling_hz,
                "end_time_sec": end / sampling_hz,
            }
        )
    return windows


def _resolve_export_dir(base_output_dir: Path, export_cfg: Dict[str, Any]) -> Path:
    subdir_value = export_cfg.get("output_dir") or "processed"
    subdir_path = Path(subdir_value)
    export_dir = subdir_path if subdir_path.is_absolute() else base_output_dir / subdir_path
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def export_timeseries(
    signals: np.ndarray,
    sampling_hz: float,
    channel_names: List[str],
    output_dir: Path,
    export_cfg: Dict[str, Any],
) -> Tuple[Path, Path]:
    n_samples = signals.shape[1]
    base_ts = pd.Timestamp("2000-01-01T00:00:00Z")
    times = base_ts + pd.to_timedelta(np.arange(n_samples) / sampling_hz, unit="s")
    records = []
    for idx, ch in enumerate(channel_names):
        df = pd.DataFrame(
            {
                "unique_id": ch,
                "timestamp": times,
                "target": signals[idx],
            }
        )
        records.append(df)
    timeseries_df = pd.concat(records, ignore_index=True)
    export_dir = _resolve_export_dir(output_dir, export_cfg)
    timeseries_df["timestamp"] = pd.to_datetime(timeseries_df["timestamp"])
    export_path = export_dir / "timeseries.csv"
    timeseries_df.to_csv(export_path, index=False)
    return export_path, export_dir


def _maybe_load_env() -> None:
    # Try repo root and current directory .env files (matches other agentic patterns).
    candidates = {Path("/cvlization_repo/.env")}
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidates.add(parent / ".env")
    for candidate in candidates:
        if candidate.is_file():
            load_dotenv(candidate, override=False)
    load_dotenv(override=False)


def summarize_with_llm(
    summary: Dict[str, Any],
    provider: str,
    model: Optional[str],
) -> str:
    provider = provider.lower()
    if provider == "mock":
        return "LLM summary (mock): dataset preprocessed with no anomalies detected."
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for --llm-provider openai")
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        prompt = (
            "You are a data-quality assistant. Given the preprocessing spec and record stats, "
            "summarize key lineage notes, potential issues, and next steps in 2-3 short paragraphs.\n\n"
            f"SPEC:\n{json.dumps(summary['spec'], indent=2)}\n\n"
            f"RECORD_STATS:\n{json.dumps(summary['records'], indent=2)}\n\n"
            f"NOTES:\n{summary['notes']}"
        )
        completion = client.chat.completions.create(
            model=model or "gpt-5-nano",
            messages=[
                {
                    "role": "system",
                    "content": "You analyze physiological signal preprocessing workflows.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content.strip()
    raise ValueError(f"Unsupported LLM provider '{provider}'.")


def preprocess_from_spec(args: argparse.Namespace) -> None:
    _maybe_load_env()
    spec_path = Path(args.spec)
    metadata, body = parse_front_matter(spec_path)

    source = metadata.get("source", {})
    record_ids = source.get("record_ids")
    if not record_ids:
        raise ValueError("spec.source.record_ids is required")
    subset = source.get("subset", "sleep-cassette")

    channels = metadata.get("channels", {}).get("include")
    sampling = metadata.get("sampling", {})
    target_hz = sampling.get("target_hz")

    qc = metadata.get("qc", {})
    clip_threshold = qc.get("amplitude_clip_uv")

    builder = SleepEDFBuilder(
        subset=subset,
        records=[f"{record}-PSG.edf" if not record.endswith(".edf") else record for record in record_ids],
        load_signals=True,
        channels=channels,
    )
    train_ds = builder.training_dataset()

    processed: List[Dict[str, Any]] = []
    window_manifest: List[Dict[str, Any]] = []
    last_export_dir: Optional[Path] = None
    export_cfg = metadata.get("export", {})
    window_cfg = metadata.get("windowing", {})
    normalization_mode = metadata.get("normalization")

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(len(train_ds)):
        example = train_ds[idx]
        signals = example.get("signals")
        if signals is None:
            continue
        sampling_rate = example.get("sampling_rate") or sampling.get("original_hz")
        if sampling_rate is None:
            raise ValueError("Sampling rate missing; include sampling.original_hz in the spec.")

        resampled, effective_hz = resample_signals(signals, sampling_rate, target_hz)
        normalized = normalize_signals(resampled, normalization_mode)
        stats = compute_stats(normalized, clip_threshold)
        stats["effective_hz"] = effective_hz

        channel_names = example.get("channel_names") or [f"ch_{i}" for i in range(normalized.shape[0])]
        export_path, export_dir = export_timeseries(
            normalized,
            effective_hz,
            channel_names,
            outputs_dir,
            export_cfg,
        )
        last_export_dir = export_dir
        windows = window_signals(
            normalized,
            effective_hz,
            window_cfg.get("length_seconds"),
            window_cfg.get("overlap_seconds"),
        )
        window_manifest.append(
            {
                "record_id": example["record_id"],
                "windows": windows,
            }
        )

        processed.append(
            {
                "record_id": example["record_id"],
                "sampling_rate": sampling_rate,
                "stats": stats,
                "timeseries_path": str(export_path),
                "window_count": len(windows),
                "channel_names": channel_names,
            }
        )

    if window_manifest:
        manifest_dir = last_export_dir or _resolve_export_dir(outputs_dir, export_cfg)
        window_manifest_path = manifest_dir / "window_manifest.json"
        window_manifest_path.write_text(json.dumps(window_manifest, indent=2))
        print(f"Stored window metadata at {window_manifest_path}")

    summary = {
        "spec": metadata,
        "notes": body.strip(),
        "records": processed,
    }
    llm_note = None
    if args.llm_provider:
        print(
            f"[preprocess] invoking LLM provider '{args.llm_provider}' "
            f"with model '{args.llm_model or 'default'}'"
        )
        llm_note = summarize_with_llm(summary, args.llm_provider, args.llm_model)
        summary["llm_summary"] = llm_note
    else:
        print("[preprocess] no LLM provider specified; skipping narrative summary")

    summary_path = outputs_dir / "preprocess_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Sleep-EDF data from spec")
    parser.add_argument(
        "--spec",
        default="specs/data_spec.sample.md",
        help="Path to data spec markdown with YAML front matter",
    )
    parser.add_argument(
        "--llm-provider",
        default=None,
        help="Optional LLM provider (openai, mock). If unspecified, no LLM call is made.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Provider-specific model identifier (e.g., gpt-5-nano).",
    )
    args = parser.parse_args()
    preprocess_from_spec(args)


if __name__ == "__main__":
    main()
