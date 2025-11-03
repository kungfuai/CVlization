import argparse
import ast
import hashlib
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx
from datasets import Dataset, DatasetDict, Features, Image, Sequence as HFSequence, Value, load_dataset
from PIL import Image as PILImage
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


TIME_MULTIPLIERS = {
    "day": 24 * 60,
    "days": 24 * 60,
    "d": 24 * 60,
    "hour": 60,
    "hours": 60,
    "hr": 60,
    "hrs": 60,
    "h": 60,
    "minute": 1,
    "minutes": 1,
    "min": 1,
    "mins": 1,
    "m": 1,
    "second": 1 / 60,
    "seconds": 1 / 60,
    "sec": 1 / 60,
    "secs": 1 / 60,
    "s": 1 / 60,
}


def _parse_time_to_minutes(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not math.isnan(value):
        return float(value)
    value = str(value).strip().lower()
    if not value:
        return None
    total_minutes = 0.0
    for number, unit in re.findall(r"(\d+)\s*([a-z]+)", value):
        multiplier = TIME_MULTIPLIERS.get(unit)
        if multiplier is None:
            continue
        total_minutes += int(number) * multiplier
    return total_minutes if total_minutes > 0 else None


def _safe_literal_eval(value: Optional[str]) -> Sequence[str]:
    if value is None:
        return []
    value = value.strip()
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return parsed
    except Exception:  # pylint: disable=broad-except
        pass
    if ";" in value:
        return [item.strip() for item in value.split(";") if item.strip()]
    if "\n" in value:
        return [item.strip() for item in value.split("\n") if item.strip()]
    return [value]


def _build_features() -> Features:
    return Features(
        {
            "title": Value("string"),
            "url": Value("string"),
            "category": Value("string"),
            "rating": Value("float32"),
            "rating_count": Value("int32"),
            "review_count": Value("int32"),
            "servings": Value("float32"),
            "prep_time_minutes": Value("float32"),
            "cook_time_minutes": Value("float32"),
            "total_time_minutes": Value("float32"),
            "calories": Value("float32"),
            "carbohydrates_g": Value("float32"),
            "sugars_g": Value("float32"),
            "fat_g": Value("float32"),
            "saturated_fat_g": Value("float32"),
            "protein_g": Value("float32"),
            "dietary_fiber_g": Value("float32"),
            "sodium_mg": Value("float32"),
            "ingredients": HFSequence(Value("string")),
            "instructions": HFSequence(Value("string")),
            "image": Image(),
        }
    )


def _sanitize_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _sanitize_int(value) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _download_image(client: httpx.Client, url: str, output_path: Path) -> bool:
    try:
        response = client.get(url, timeout=10.0)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        with PILImage.open(output_path) as img:
            img.convert("RGB")
        return True
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.debug("Failed to download %s: %s", url, exc)
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        return False


def _hash_url(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8"), usedforsecurity=False).hexdigest()


def _prepare_records(
    raw_dataset,
    max_examples: int,
    image_dir: Path,
    require_total_time: bool,
    client: httpx.Client,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    image_dir.mkdir(parents=True, exist_ok=True)

    iterator = tqdm(raw_dataset, total=max_examples or len(raw_dataset), desc="Processing recipes")
    for example in iterator:
        if max_examples is not None and len(records) >= max_examples:
            break

        image_url = example.get("image")
        calories = example.get("calories")
        total_time_minutes = _parse_time_to_minutes(example.get("total_time"))

        if not image_url or calories is None:
            continue
        if require_total_time and total_time_minutes is None:
            continue

        image_filename = _hash_url(image_url) + ".jpg"
        image_path = image_dir / image_filename
        if not image_path.exists():
            if not _download_image(client, image_url, image_path):
                continue

        record = {
            "title": example.get("title") or "",
            "url": example.get("url") or "",
            "category": example.get("category") or "",
            "rating": _sanitize_float(example.get("rating")),
            "rating_count": _sanitize_int(example.get("rating_count")),
            "review_count": _sanitize_int(example.get("review_count")),
            "servings": _sanitize_float(example.get("servings")),
            "prep_time_minutes": _parse_time_to_minutes(example.get("prep_time")) or float("nan"),
            "cook_time_minutes": _parse_time_to_minutes(example.get("cook_time")) or float("nan"),
            "total_time_minutes": total_time_minutes or float("nan"),
            "calories": _sanitize_float(calories),
            "carbohydrates_g": _sanitize_float(example.get("carbohydrates_g")),
            "sugars_g": _sanitize_float(example.get("sugars_g")),
            "fat_g": _sanitize_float(example.get("fat_g")),
            "saturated_fat_g": _sanitize_float(example.get("saturated_fat_g")),
            "protein_g": _sanitize_float(example.get("protein_g")),
            "dietary_fiber_g": _sanitize_float(example.get("dietary_fiber_g")),
            "sodium_mg": _sanitize_float(example.get("sodium_mg")),
            "ingredients": list(_safe_literal_eval(example.get("ingredients"))),
            "instructions": list(_safe_literal_eval(example.get("instructions_list"))),
            "image": str(image_path.resolve()),
        }
        records.append(record)

    return records


def build_subset(max_examples: int, val_ratio: float, output_dir: Path, require_total_time: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    dataset_path = output_dir / "hf_recipe_subset"

    LOGGER.info("Loading Shengtao/recipe split…")
    source_dataset = load_dataset("Shengtao/recipe")["train"]

    LOGGER.info("Collecting up to %d examples (val_ratio=%.2f)…", max_examples, val_ratio)
    features = _build_features()

    with httpx.Client(follow_redirects=True, timeout=10.0) as client:
        records = _prepare_records(
            raw_dataset=source_dataset,
            max_examples=max_examples,
            image_dir=image_dir,
            require_total_time=require_total_time,
            client=client,
        )

    if not records:
        raise RuntimeError("No records were downloaded. Relax filters or check network connectivity.")

    dataset = Dataset.from_list(records, features=features)
    dataset = dataset.filter(lambda example: example["image"] is not None)
    dataset = dataset.shuffle(seed=20241027)

    test_size = max(1, int(len(dataset) * val_ratio))
    splits = dataset.train_test_split(test_size=test_size, seed=20241027)
    hf_dataset = DatasetDict({"train": splits["train"], "validation": splits["test"]})

    LOGGER.info("Saving subset with %d train / %d validation examples to %s", len(hf_dataset["train"]), len(hf_dataset["validation"]), dataset_path)
    hf_dataset.save_to_disk(dataset_path)

    return dataset_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a local recipe subset with downloaded images.")
    parser.add_argument("--max-examples", type=int, default=5000, help="Maximum number of examples to keep.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--output-dir", type=Path, default=Path("var/datasets/recipe_subset"), help="Output directory.")
    parser.add_argument(
        "--require-total-time",
        action="store_true",
        help="Discard examples without total_time.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    dataset_path = build_subset(
        max_examples=args.max_examples,
        val_ratio=args.val_ratio,
        output_dir=args.output_dir,
        require_total_time=args.require_total_time,
    )
    LOGGER.info("Dataset saved at %s", dataset_path)


if __name__ == "__main__":
    main()
