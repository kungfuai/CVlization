import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Training config to validate")
    parser.add_argument("--dataset-path", help="Override dataset.path from config")
    parser.add_argument("--data-files", help="Override dataset.data_files from config")
    parser.add_argument("--data-dir", help="Override dataset.data_dir from config")
    parser.add_argument("--max-examples", type=int, help="Only inspect the first N non-empty rows")
    parser.add_argument("--strict-json-target", action="store_true", help="Require target_json values to parse as JSON")
    return parser.parse_args()


def expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "$" in expanded:
            raise ValueError(f"Unexpanded environment variable in config value {value!r}")
        return expanded
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    return value


def maybe_json_loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def iter_jsonl(path: Path):
    with path.open() as f:
        for line_number, line in enumerate(f, 1):
            if line.strip():
                yield line_number, json.loads(line)


def data_files_from_config(dataset_config: dict[str, Any]) -> list[Path]:
    data_files = expand_env_vars(dataset_config.get("data_files"))
    if not data_files:
        raise ValueError("dataset.data_files is required for local JSONL validation")
    if isinstance(data_files, str):
        return [Path(data_files)]
    if isinstance(data_files, list):
        return [Path(item) for item in data_files]
    if isinstance(data_files, dict):
        return [Path(item) for values in data_files.values() for item in ([values] if isinstance(values, str) else values)]
    raise ValueError(f"Unsupported dataset.data_files type: {type(data_files).__name__}")


def validate_row(row: dict[str, Any], target_column: str, strict_json_target: bool) -> list[str]:
    errors = []
    if "messages" not in row:
        errors.append("missing messages")
    else:
        messages = maybe_json_loads(row["messages"])
        if not isinstance(messages, list) or not messages:
            errors.append("messages is not a non-empty list")
        elif not all(isinstance(message, dict) and message.get("role") and message.get("content") for message in messages):
            errors.append("messages entries must contain role and content")

    target_value = row.get(target_column)
    if target_value is None and target_column != "target_json":
        target_value = row.get("target_json")
    if target_value is None:
        target_value = row.get("target_text")
    if target_value is None:
        errors.append(f"missing target column {target_column!r} and no target_json/target_text fallback")
    elif strict_json_target or target_column == "target_json":
        parsed = maybe_json_loads(target_value)
        if not isinstance(parsed, (dict, list)):
            errors.append("target_json is not JSON object/list")

    return errors


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    dataset_config = dict(config["dataset"])
    if args.dataset_path:
        dataset_config["path"] = args.dataset_path
    if args.data_files:
        dataset_config["data_files"] = args.data_files
    if args.data_dir:
        dataset_config["data_dir"] = args.data_dir
    if dataset_config.get("path") != "json":
        raise ValueError("validate_dataset.py currently validates local JSON/JSONL datasets only")

    target_sources = set(dataset_config.get("target_sources") or [])
    target_column = dataset_config.get("target_column", "target_json")
    files = data_files_from_config(dataset_config)

    rows = 0
    documents = set()
    target_source_counts = Counter()
    prompt_style_counts = Counter()
    schema_style_counts = Counter()
    skipped_counts = Counter()
    error_counts = Counter()
    error_examples = []

    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
        for line_number, row in iter_jsonl(path):
            if args.max_examples and rows >= args.max_examples:
                break
            rows += 1
            if row.get("document_id") is not None:
                documents.add(str(row["document_id"]))
            source = row.get("target_source")
            target_source_counts[source] += 1
            if row.get("prompt_style") is not None:
                prompt_style_counts[row["prompt_style"]] += 1
            if row.get("schema_style") is not None:
                schema_style_counts[row["schema_style"]] += 1
            if target_sources and source not in target_sources:
                skipped_counts["target_source not selected"] += 1
                continue
            for error in validate_row(row, target_column, args.strict_json_target):
                error_counts[error] += 1
                if len(error_examples) < 10:
                    error_examples.append({"line": line_number, "error": error, "id": row.get("id")})
        if args.max_examples and rows >= args.max_examples:
            break

    print(f"Files: {', '.join(str(path) for path in files)}")
    print(f"Rows: {rows}")
    print(f"Documents: {len(documents)}")
    print(f"Target sources: {dict(sorted(target_source_counts.items(), key=lambda item: str(item[0])))}")
    if prompt_style_counts:
        print(f"Prompt styles: {dict(prompt_style_counts)}")
    if schema_style_counts:
        print(f"Schema styles: {dict(schema_style_counts)}")
    if skipped_counts:
        print(f"Skipped rows: {dict(skipped_counts)}")

    if error_counts:
        print(f"Errors: {dict(error_counts)}")
        print(f"Error examples: {json.dumps(error_examples, ensure_ascii=False)}")
        raise SystemExit(1)

    print("Validation passed.")


if __name__ == "__main__":
    main()
