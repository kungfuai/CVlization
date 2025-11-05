#!/usr/bin/env python3
"""
Fetch a subset of the Hugging Face `maifeeulasad/askubuntu-data` dataset and write
markdown files under `data/docs/askubuntu/` for the LangGraph helpdesk example.

Usage:
  python fetch_askubuntu_subset.py --limit 1000

Requires the `datasets` library (`pip install datasets`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit("Install the 'datasets' package first: pip install datasets") from exc


ROOT = Path(__file__).resolve().parent
DOC_DIR = ROOT / "data" / "docs" / "askubuntu"


def iter_rows(limit: int | None) -> Iterable[dict]:
    dataset = load_dataset(
        "maifeeulasad/askubuntu-data",
        split="train",
        streaming=True,
    )
    for idx, row in enumerate(dataset):
        if limit is not None and idx >= limit:
            break
        yield row


def normalize_tags(tags: object) -> List[str]:
    if tags is None:
        return []
    if isinstance(tags, list):
        return [str(t) for t in tags]
    if isinstance(tags, str):
        try:
            parsed = json.loads(tags)
            if isinstance(parsed, list):
                return [str(t) for t in parsed]
        except json.JSONDecodeError:
            return [tags]
    return [str(tags)]


def write_markdown(idx: int, row: dict) -> Path:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    title = row.get("title") or row.get("question") or "Untitled"
    question = row.get("body_markdown") or row.get("body") or ""
    accepted = row.get("acceptedAnswer") or {}
    answer = accepted.get("body") if isinstance(accepted, dict) else ""
    if not answer:
        answers = row.get("answers") or []
        if isinstance(answers, list) and answers:
            answer = answers[0].get("body", "")
    tags = normalize_tags(row.get("tags"))

    slug = f"askubuntu_{idx:05d}.md"
    path = DOC_DIR / slug
    sections = [
        f"# {title.strip()}",
        "",
        "## Question",
        (question.strip() or "No description provided."),
        "",
        "## Answer",
        (answer.strip() or "Answer not available in subset."),
    ]
    if tags:
        sections.extend(["", "## Tags", ", ".join(tags)])
    path.write_text("\n".join(sections), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a subset of AskUbuntu data into markdown docs."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of posts to download (<=0 to stream entire dataset).",
    )
    args = parser.parse_args()

    limit = None if args.limit <= 0 else args.limit
    count = 0
    for count, row in enumerate(iter_rows(limit), start=1):
        write_markdown(count, row)

    print(f"Wrote {count} documents to {DOC_DIR}")


if __name__ == "__main__":
    main()
