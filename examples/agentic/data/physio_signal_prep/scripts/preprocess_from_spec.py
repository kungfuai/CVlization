from __future__ import annotations

import argparse
import glob
import json
import re
import os
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


def _matches_any(value: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch(value, pat) for pat in patterns)


def _resolve_records_from_source(source: Dict[str, Any]) -> List[str]:
    """Resolve record IDs from explicit list, globs, or record name patterns."""
    if source.get("record_ids"):
        return [r if r.endswith(".edf") else f"{r}.edf" for r in source["record_ids"]]

    include_globs: Iterable[str] = source.get("include_globs") or []
    exclude_globs: Iterable[str] = source.get("exclude_globs") or []
    record_patterns: Iterable[str] = source.get("record_patterns") or []

    candidates: List[str] = []
    for pattern in include_globs:
        for path_str in glob.glob(os.path.expanduser(pattern), recursive=True):
            if exclude_globs and _matches_any(path_str, exclude_globs):
                continue
            if path_str.lower().endswith(".edf"):
                candidates.append(path_str)

    if not candidates:
        raise ValueError(
            "No EDF files found. Provide source.record_ids or populate source.include_globs with paths to .edf files."
        )

    record_ids: List[str] = []
    for path_str in candidates:
        name = Path(path_str).name
        if record_patterns and not _matches_any(name, record_patterns):
            continue
        record_ids.append(name)

    if not record_ids:
        raise ValueError(
            "No EDF files matched record_patterns; adjust source.record_patterns/include_globs or provide record_ids."
        )
    return sorted(set(record_ids))


def _resolve_source_hz(record_id: str, sampling_cfg: Dict[str, Any]) -> Optional[float]:
    source_hz_cfg = sampling_cfg.get("source_hz", {})
    overrides = source_hz_cfg.get("overrides") or []
    for override in overrides:
        pattern = override.get("pattern")
        if pattern and _matches_any(record_id, pattern):
            return override.get("hz")
    return source_hz_cfg.get("default") or sampling_cfg.get("original_hz")


def _select_channels(
    signals: np.ndarray,
    channel_names: Sequence[str],
    channel_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """Select, order, and optionally pad channels.

    Returns (selected_signals, selected_names, missing_optional, padded_optional).
    """
    requested_required: List[str] = channel_cfg.get("required") or channel_cfg.get("include") or []
    requested_optional: List[str] = channel_cfg.get("optional") or []
    exclude_patterns: List[str] = channel_cfg.get("exclude") or []
    pad_missing: bool = bool(channel_cfg.get("pad_missing_channels", False))

    if not requested_required and not requested_optional and not exclude_patterns:
        # Nothing to filter; keep all channels as-is.
        return signals, list(channel_names), [], []

    def _resolve_one(ch: str) -> Optional[str]:
        if ch in channel_names:
            return ch
        alias = f"EEG {ch}" if not ch.startswith("EEG ") else ch
        if alias in channel_names:
            return alias
        return None

    selected_arrays: List[np.ndarray] = []
    selected_names: List[str] = []
    missing_optional: List[str] = []
    padded_optional: List[str] = []

    # Preserve order: required first, then optional.
    for ch in requested_required + requested_optional:
        resolved = _resolve_one(ch)
        if resolved and not _matches_any(resolved, exclude_patterns):
            idx = channel_names.index(resolved)
            selected_arrays.append(signals[idx:idx+1, :])
            selected_names.append(ch)
        else:
            if ch in requested_required:
                raise ValueError(f"Channel '{ch}' not found (checked aliases) in record; available: {channel_names}")
            missing_optional.append(ch)
            if pad_missing:
                padded_optional.append(ch)

    if pad_missing and missing_optional:
        n_samples = signals.shape[1]
        for ch in missing_optional:
            selected_arrays.append(np.full((1, n_samples), np.nan, dtype=signals.dtype))
            selected_names.append(ch)

    if not selected_arrays:
        raise ValueError("No channels selected after applying required/optional/exclude rules.")

    selected = np.vstack(selected_arrays)
    return selected, selected_names, missing_optional, padded_optional


def preprocess_from_spec(args: argparse.Namespace) -> None:
    _maybe_load_env()
    spec_path = Path(args.spec)
    metadata, body = parse_front_matter(spec_path)

    source = metadata.get("source", {})
    record_ids = _resolve_records_from_source(source)
    subset = source.get("subset", "sleep-cassette")

    channel_cfg = metadata.get("channels", {}) or {}
    sampling = metadata.get("sampling", {})
    target_hz = sampling.get("target_hz")

    qc = metadata.get("qc", {})
    clip_threshold = qc.get("amplitude_clip_uv")

    builder = SleepEDFBuilder(
        subset=subset,
        records=[f"{record}-PSG.edf" if not record.endswith(".edf") else record for record in record_ids],
        load_signals=True,
        channels=None,  # load all channels; selection/padding handled downstream
    )
    train_ds = builder.training_dataset()

    processed: List[Dict[str, Any]] = []
    window_manifest: List[Dict[str, Any]] = []
    last_export_dir: Optional[Path] = None
    export_cfg = metadata.get("export", {})
    window_cfg = metadata.get("windowing", {})
    normalization_cfg = metadata.get("normalization")
    normalization_mode = (
        normalization_cfg.get("strategy") if isinstance(normalization_cfg, dict) else normalization_cfg
    )

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(len(train_ds)):
        example = train_ds[idx]
        signals = example.get("signals")
        if signals is None:
            continue
        sampling_rate = example.get("sampling_rate") or _resolve_source_hz(example["record_id"], sampling)
        if sampling_rate is None:
            raise ValueError(
                "Sampling rate missing; include sampling.source_hz.default or sampling.original_hz in the spec."
            )

        selected_signals, selected_names, missing_optional, padded_optional = _select_channels(
            signals, example.get("channel_names") or [], channel_cfg
        )

        resampled, effective_hz = resample_signals(selected_signals, sampling_rate, target_hz)
        normalized = normalize_signals(resampled, normalization_mode)
        stats = compute_stats(normalized, clip_threshold)
        stats["effective_hz"] = effective_hz

        export_path, export_dir = export_timeseries(
            normalized,
            effective_hz,
            selected_names,
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
                "channel_names": selected_names,
                "present_channels": selected_names,
                "missing_optional_channels": missing_optional,
                "padded_optional_channels": padded_optional,
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
