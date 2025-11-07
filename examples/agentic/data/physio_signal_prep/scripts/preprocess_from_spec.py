from __future__ import annotations

import argparse
import json
import re
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from dotenv import load_dotenv

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
