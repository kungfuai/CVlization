from __future__ import annotations

import argparse
import json
import random
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Tuple


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DATA = DATA_DIR / "raw" / "penguins_in_a_table.json"
RAW_URL = "https://raw.githubusercontent.com/XinyuanWangCS/PromptAgent/main/datasets/penguins_in_a_table.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/dev/test splits for penguins_in_a_table.")
    parser.add_argument("--train", type=int, default=80)
    parser.add_argument("--dev", type=int, default=35)
    parser.add_argument("--test", type=int, default=34)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_examples() -> List[dict]:
    ensure_raw_dataset()
    data = json.loads(RAW_DATA.read_text())
    return list(data.get("examples", []))


def ensure_raw_dataset() -> None:
    if RAW_DATA.exists():
        return
    RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading raw dataset from {RAW_URL}...")
    try:
        with urllib.request.urlopen(RAW_URL) as response:
            RAW_DATA.write_bytes(response.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Failed to download dataset from {RAW_URL}. "
            "Download it manually and place it at "
            f"{RAW_DATA}."
        ) from exc


def split_examples(
    examples: List[dict], train_size: int, dev_size: int, test_size: int, seed: int
) -> Tuple[List[dict], List[dict], List[dict]]:
    total_requested = train_size + dev_size + test_size
    if total_requested > len(examples):
        raise ValueError(f"Requested {total_requested} examples but only {len(examples)} available.")

    rng = random.Random(seed)
    shuffled = examples.copy()
    rng.shuffle(shuffled)

    train = shuffled[:train_size]
    dev = shuffled[train_size : train_size + dev_size]
    test = shuffled[train_size + dev_size : train_size + dev_size + test_size]
    return train, dev, test


def write_split(records: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        for record in records:
            fp.write(json.dumps(record) + "\n")


def main() -> None:
    args = parse_args()
    examples = load_examples()
    train, dev, test = split_examples(examples, args.train, args.dev, args.test, args.seed)

    write_split(train, DATA_DIR / "train.jsonl")
    write_split(dev, DATA_DIR / "dev.jsonl")
    write_split(test, DATA_DIR / "test.jsonl")

    print("Wrote splits:")
    print("  train:", len(train))
    print("  dev:", len(dev))
    print("  test:", len(test))


if __name__ == "__main__":
    main()
