from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

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


def preprocess_from_spec(args: argparse.Namespace) -> None:
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
    for idx in range(len(train_ds)):
        example = train_ds[idx]
        signals = example.get("signals")
        if signals is None:
            continue
        stats = compute_stats(signals, clip_threshold)
        if target_hz:
            stats["target_hz"] = target_hz
        processed.append({
            "record_id": example["record_id"],
            "sampling_rate": example.get("sampling_rate"),
            "stats": stats,
        })

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = outputs_dir / "preprocess_summary.json"
    summary = {
        "spec": metadata,
        "notes": body.strip(),
        "records": processed,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Sleep-EDF data from spec")
    parser.add_argument(
        "--spec",
        default="specs/data_spec.sample.md",
        help="Path to data spec markdown with YAML front matter",
    )
    args = parser.parse_args()
    preprocess_from_spec(args)


if __name__ == "__main__":
    main()
