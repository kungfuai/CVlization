from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", help="JSONL file from predict_eval.py")
    parser.add_argument("--output", help="Optional JSON summary path")
    parser.add_argument("--row-output", help="Optional JSONL row-level metrics path")
    parser.add_argument("--field-output", help="Optional JSONL field-level score path")
    parser.add_argument("--error-output", help="Optional JSON diagnostic error report path")
    parser.add_argument(
        "--max-error-examples",
        type=int,
        default=20,
        help="Maximum missing/extra/mismatched field examples to write per row",
    )
    parser.add_argument(
        "--repair-json-string-newlines",
        action="store_true",
        help="Repair literal newlines/tabs inside JSON strings before parsing",
    )
    parser.add_argument(
        "--normalize-values",
        action="store_true",
        help="Collapse whitespace in scalar values before exact field comparison",
    )
    parser.add_argument(
        "--exclude-field-types",
        default="checkbox",
        help=(
            "Comma-separated detected field types to exclude from metrics, "
            "for example 'checkbox' or 'checkbox,null'. Defaults to 'checkbox' "
            "because text OCR usually does not reliably encode checkbox state."
        ),
    )
    parser.add_argument(
        "--include-checkbox-fields",
        action="store_true",
        help="Include checkbox fields in metrics. This disables the default checkbox exclusion.",
    )
    parser.add_argument(
        "--exclude-path-regex",
        action="append",
        default=[],
        help=(
            "Regex for flattened field paths to exclude from metrics. "
            "May be passed multiple times."
        ),
    )
    return parser.parse_args()


# Borrowed and adapted from the internal document-extraction evaluator's
# comparator utilities.
DATE_FORMATS = [
    "%m/%d/%Y",
    "%m/%d/%y",
    "%Y-%m-%d",
    "%B %d, %Y",
    "%b %d, %Y",
    "%B %d %Y",
    "%b %d %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%m-%d-%Y",
    "%m-%d-%y",
    "%Y%m%d",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
]
TRUTHY_VALUES = {"yes", "true", "1", "x", "checked", "on", "y"}
FALSY_VALUES = {"no", "false", "0", "", "unchecked", "off", "n", "none", "null"}
EMPTY_EQUIVALENT_STRINGS = {"", "none", "null"}
MISSING = object()

# These fields are generated workflow/header metadata rather than values a text-only
# form extractor should read from the OCR prompt. Keep them out of both target and
# prediction path sets so they are neither rewarded nor penalized.
DEFAULT_IGNORED_PATHS = {"body.documentCode"}
DEFAULT_IGNORED_PREFIXES = ("header.",)
DEFAULT_IGNORED_PATH_PATTERNS = (
    re.compile(r"^body\.formsData\[\d+\]\.formsDescription$"),
)


def ignored_path(path: str) -> bool:
    return (
        path in DEFAULT_IGNORED_PATHS
        or any(pattern.match(path) for pattern in DEFAULT_IGNORED_PATH_PATTERNS)
        or any(
            path == prefix[:-1] or path.startswith(prefix)
            for prefix in DEFAULT_IGNORED_PREFIXES
        )
    )


def filter_ignored_paths(values: dict[str, Any]) -> dict[str, Any]:
    return {path: value for path, value in values.items() if not ignored_path(path)}


def parse_excluded_field_types(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def excluded_path(path: str, patterns: list[re.Pattern]) -> bool:
    return any(pattern.search(path) for pattern in patterns)


def exclude_metric_paths(
    flat_target: dict[str, str],
    flat_prediction: dict[str, str],
    raw_target: dict[str, Any],
    raw_prediction: dict[str, Any],
    excluded_field_types: set[str],
    excluded_path_patterns: list[re.Pattern],
) -> tuple[dict[str, str], dict[str, str], dict[str, Any], dict[str, Any], int, int]:
    excluded_target_paths = set()
    excluded_prediction_paths = set()

    for path, value in raw_target.items():
        field_type = detect_field_type(path, value)
        if field_type in excluded_field_types or excluded_path(path, excluded_path_patterns):
            excluded_target_paths.add(path)

    for path, value in raw_prediction.items():
        field_type = detect_field_type(path, value)
        if field_type in excluded_field_types or excluded_path(path, excluded_path_patterns):
            excluded_prediction_paths.add(path)

    excluded_paths = excluded_target_paths | excluded_prediction_paths
    return (
        {path: value for path, value in flat_target.items() if path not in excluded_paths},
        {path: value for path, value in flat_prediction.items() if path not in excluded_paths},
        {path: value for path, value in raw_target.items() if path not in excluded_paths},
        {path: value for path, value in raw_prediction.items() if path not in excluded_paths},
        len(excluded_target_paths),
        len(excluded_prediction_paths),
    )


def token_sort_ratio(a: str, b: str) -> float:
    """Return a rapidfuzz-style token-sort ratio in [0, 1]."""
    try:
        from rapidfuzz import fuzz

        return fuzz.token_sort_ratio(a, b) / 100.0
    except ImportError:
        a_sorted = " ".join(sorted(a.split()))
        b_sorted = " ".join(sorted(b.split()))
        return SequenceMatcher(None, a_sorted, b_sorted).ratio()


def normalize_string(value: str) -> str:
    return value.lower().strip()


def parse_date(value: str) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def compare_dates(extracted: str, expected: str) -> float | None:
    ext_date = parse_date(str(extracted))
    exp_date = parse_date(str(expected))
    if ext_date is None or exp_date is None:
        return None
    return 1.0 if ext_date.date() == exp_date.date() else 0.0


def normalize_numeric(value: str) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", str(value))
    if not cleaned or cleaned in (".", "-", "-."):
        return None
    if "(" in str(value) and ")" in str(value) and not cleaned.startswith("-"):
        cleaned = "-" + cleaned
    return cleaned


def compare_currency(extracted: str, expected: str) -> float | None:
    ext_num = normalize_numeric(str(extracted))
    exp_num = normalize_numeric(str(expected))
    if ext_num is None or exp_num is None:
        return None
    try:
        return 1.0 if abs(float(ext_num) - float(exp_num)) < 0.01 else 0.0
    except ValueError:
        return None


def normalize_digits(value: str) -> str:
    return re.sub(r"\D", "", str(value))


def compare_ssn_ein(extracted: str, expected: str) -> float:
    ext_digits = normalize_digits(extracted)
    exp_digits = normalize_digits(expected)
    if len(ext_digits) == 9 and len(exp_digits) == 9:
        return 1.0 if ext_digits == exp_digits else 0.0
    if len(ext_digits) == 4 and len(exp_digits) == 9:
        return 0.5 if exp_digits.endswith(ext_digits) else 0.0
    if len(exp_digits) == 4 and len(ext_digits) == 9:
        return 0.5 if ext_digits.endswith(exp_digits) else 0.0
    return 0.0


def compare_phone(extracted: str, expected: str) -> float | None:
    ext_digits = normalize_digits(extracted)
    exp_digits = normalize_digits(expected)
    if len(ext_digits) == 11 and ext_digits.startswith("1"):
        ext_digits = ext_digits[1:]
    if len(exp_digits) == 11 and exp_digits.startswith("1"):
        exp_digits = exp_digits[1:]
    if len(ext_digits) != 10 or len(exp_digits) != 10:
        return None
    return 1.0 if ext_digits == exp_digits else 0.0


def compare_checkbox(extracted: str, expected: str) -> float | None:
    ext_norm = normalize_string(str(extracted)) if extracted is not None else ""
    exp_norm = normalize_string(str(expected)) if expected is not None else ""
    ext_truthy = ext_norm in TRUTHY_VALUES
    ext_falsy = ext_norm in FALSY_VALUES or ext_norm == ""
    exp_truthy = exp_norm in TRUTHY_VALUES
    exp_falsy = exp_norm in FALSY_VALUES or exp_norm == ""
    if (ext_truthy or ext_falsy) and (exp_truthy or exp_falsy):
        return 1.0 if ext_truthy == exp_truthy else 0.0
    return None


def compare_zip(extracted: str, expected: str) -> float | None:
    ext_digits = normalize_digits(extracted)
    exp_digits = normalize_digits(expected)
    if len(ext_digits) >= 5 and len(exp_digits) >= 5:
        if ext_digits[:5] == exp_digits[:5]:
            if len(ext_digits) >= 9 and len(exp_digits) >= 9:
                return 1.0 if ext_digits[:9] == exp_digits[:9] else 0.5
            return 1.0
        return 0.0
    return None


def compare_name(extracted: str, expected: str) -> float:
    def normalize_name(value: str) -> str:
        value = value.lower().strip()
        for prefix in ["mr ", "mr. ", "mrs ", "mrs. ", "ms ", "ms. ", "dr ", "dr. "]:
            if value.startswith(prefix):
                value = value[len(prefix) :]
        for suffix in [" jr", " jr.", " sr", " sr.", " ii", " iii", " iv"]:
            if value.endswith(suffix):
                value = value[: -len(suffix)]
        return " ".join(value.split())

    ext_norm = normalize_name(extracted)
    exp_norm = normalize_name(expected)
    if ext_norm == exp_norm:
        return 1.0
    return token_sort_ratio(ext_norm, exp_norm)


def compare_list(extracted: list, expected: list) -> float:
    if not extracted and not expected:
        return 1.0
    if not extracted or not expected:
        return 0.0
    ext_set = {str(x).strip().lower() for x in extracted}
    exp_set = {str(x).strip().lower() for x in expected}
    if ext_set == exp_set:
        return 1.0
    union = ext_set | exp_set
    return len(ext_set & exp_set) / len(union) if union else 0.0


def is_empty_equivalent(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in EMPTY_EQUIVALENT_STRINGS
    return value in ([], {})


def compare_address(extracted: str, expected: str) -> float:
    def normalize_address(value: str) -> str:
        value = value.lower().strip()
        replacements = {
            " street": " st",
            " avenue": " ave",
            " boulevard": " blvd",
            " drive": " dr",
            " road": " rd",
            " lane": " ln",
            " court": " ct",
            " place": " pl",
            " suite": " ste",
            " apartment": " apt",
            " p.o. box": " po box",
            " p.o.box": " po box",
            "p.o. box": "po box",
            "p.o.box": "po box",
        }
        for old, new in replacements.items():
            value = value.replace(old, new)
        value = re.sub(r"[,.]", " ", value)
        return " ".join(value.split())

    def meaningful_tokens(value: str) -> set[str]:
        fillers = {"apt", "ste", "box", "po", "st", "ave", "blvd", "dr", "rd", "ln", "ct", "pl"}
        tokens = set()
        for token in value.split():
            if token.isdigit():
                tokens.add(token)
            elif len(token) > 2 and token not in fillers:
                tokens.add(token)
        return tokens

    ext_norm = normalize_address(extracted)
    exp_norm = normalize_address(expected)
    if ext_norm == exp_norm:
        return 1.0
    if "po box" in ext_norm and "po box" in exp_norm:
        ext_match = re.search(r"po box (\d+)", ext_norm)
        exp_match = re.search(r"po box (\d+)", exp_norm)
        if ext_match and exp_match and ext_match.group(1) != exp_match.group(1):
            return 0.0
    ext_words = meaningful_tokens(ext_norm)
    exp_words = meaningful_tokens(exp_norm)
    if not ext_words and not exp_words:
        return 1.0 if ext_norm == exp_norm else 0.0
    if not ext_words or not exp_words:
        return 0.0
    return len(ext_words & exp_words) / len(ext_words | exp_words)


def detect_field_type(field_path: str, value: Any) -> str:
    field_lower = field_path.lower()
    value_str = str(value).strip() if value is not None else ""

    if isinstance(value, list):
        return "list"
    if "address" in field_lower:
        if "zip" in field_lower:
            return "zip"
        if any(part in field_lower for part in [".state", "stateabbreviation"]):
            return "text"
        if ".city" in field_lower:
            return "text"
        if "recipient" in field_lower:
            return "name"
        return "address"
    if any(kw in field_lower for kw in ["date", "_date", "dob", "birth", "receiveddate", "scanneddate", "postmarkdate"]):
        return "date"
    if any(kw in field_lower for kw in ["ssn", "social_security"]):
        return "ssn"
    if field_lower.endswith("tin") and "type" not in field_lower:
        return "ssn"
    if any(kw in field_lower for kw in ["ein", "employer_id", "fein"]):
        return "ein"
    if any(kw in field_lower for kw in ["phone", "telephone", "fax", "mobile", "cell"]):
        return "phone"
    if any(kw in field_lower for kw in ["zip", "postal"]):
        return "zip"
    if any(
        kw in field_lower
        for kw in ["amount", "total", "balance", "credit", "payment", "price", "cost", "fee", "due", "subtotal"]
    ) and normalize_numeric(value_str):
        return "currency"
    if any(kw in field_lower for kw in ["checkbox", "checked", "selected", "is_", "has_", "enabled"]):
        return "checkbox"
    if value_str.lower() in TRUTHY_VALUES | FALSY_VALUES:
        return "checkbox"
    if parse_date(value_str):
        return "date"
    if any(
        kw in field_lower
        for kw in [
            "first_name",
            "last_name",
            "middle_name",
            "full_name",
            "firstname",
            "lastname",
            "middlename",
            "business_name",
            "company_name",
            "employer_name",
            "customer_name",
            "vendor_name",
            "sender_name",
            "recipient_name",
            "applicant_name",
        ]
    ):
        return "name"
    if "count" in field_lower:
        return "integer"
    return "text"


def type_aware_compare(field_path: str, extracted: Any, expected: Any) -> dict[str, Any]:
    if extracted is MISSING and is_empty_equivalent(expected):
        return {"score": 1.0, "field_type": "null", "method": "missing_expected_null"}
    if extracted is MISSING:
        return {"score": 0.0, "field_type": detect_field_type(field_path, expected), "method": "missing"}
    if is_empty_equivalent(extracted) and is_empty_equivalent(expected):
        return {"score": 1.0, "field_type": "null", "method": "both_empty"}
    if expected is None:
        return {"score": 0.0, "field_type": "null", "method": "expected_null"}
    if extracted is None:
        return {"score": 0.0, "field_type": detect_field_type(field_path, expected), "method": "one_null"}

    field_type = detect_field_type(field_path, expected)
    if field_type == "list" or isinstance(extracted, list) or isinstance(expected, list):
        ext_list = extracted if isinstance(extracted, list) else [extracted]
        exp_list = expected if isinstance(expected, list) else [expected]
        return {"score": compare_list(ext_list, exp_list), "field_type": "list", "method": "type_aware"}
    if field_type == "integer":
        try:
            return {
                "score": 1.0 if int(extracted) == int(expected) else 0.0,
                "field_type": "integer",
                "method": "type_aware",
            }
        except (ValueError, TypeError):
            pass

    ext_str = str(extracted).strip()
    exp_str = str(expected).strip()
    if ext_str == exp_str:
        return {"score": 1.0, "field_type": field_type, "method": "exact"}
    if ext_str.lower() == exp_str.lower():
        return {"score": 1.0, "field_type": field_type, "method": "case_insensitive"}

    comparators = {
        "date": compare_dates,
        "currency": compare_currency,
        "ssn": compare_ssn_ein,
        "ein": compare_ssn_ein,
        "phone": compare_phone,
        "checkbox": compare_checkbox,
        "zip": compare_zip,
        "name": compare_name,
        "address": compare_address,
    }
    if field_type in comparators:
        result = comparators[field_type](ext_str, exp_str)
        if result is not None:
            return {"score": result, "field_type": field_type, "method": "type_aware"}
    return {"score": token_sort_ratio(ext_str, exp_str), "field_type": field_type, "method": "fuzzy"}


def truncate_value(value: str, max_length: int = 160) -> str:
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."


def repair_json_string_controls(text: str) -> str:
    out = []
    in_string = False
    escape = False
    for char in text:
        if in_string:
            if escape:
                out.append(char)
                escape = False
            elif char == "\\":
                out.append(char)
                escape = True
            elif char == '"':
                out.append(char)
                in_string = False
            elif char == "\n":
                out.append("\\n")
            elif char == "\r":
                out.append("\\r")
            elif char == "\t":
                out.append("\\t")
            else:
                out.append(char)
            continue

        out.append(char)
        if char == '"':
            in_string = True
    return "".join(out)


def parse_json_candidate(candidate: str, repair_json_string_newlines: bool) -> Any:
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        if not repair_json_string_newlines:
            raise
        return json.loads(repair_json_string_controls(candidate))


def extract_json_object(text: str, repair_json_string_newlines: bool = False) -> Any:
    text = text.strip()
    try:
        return parse_json_candidate(text, repair_json_string_newlines)
    except json.JSONDecodeError:
        pass

    start_candidates = [pos for pos in (text.find("{"), text.find("[")) if pos != -1]
    if not start_candidates:
        raise ValueError("No JSON object or array start found")

    start = min(start_candidates)
    for end in range(len(text), start, -1):
        candidate = text[start:end].strip()
        try:
            return parse_json_candidate(candidate, repair_json_string_newlines)
        except json.JSONDecodeError:
            continue
    raise ValueError("Could not parse JSON substring")


def normalize_scalar(value: Any, normalize_values: bool) -> str:
    if normalize_values and is_empty_equivalent(value):
        text = ""
    elif value is None:
        text = "null"
    elif isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value).strip()
    if normalize_values:
        text = re.sub(r"\s+", " ", text)
    return text


def flatten(value: Any, prefix: str = "", normalize_values: bool = False) -> dict[str, str]:
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(child, child_prefix, normalize_values=normalize_values))
        return out
    if isinstance(value, list):
        out = {}
        for i, child in enumerate(value):
            out.update(flatten(child, f"{prefix}[{i}]", normalize_values=normalize_values))
        return out
    return {prefix: normalize_scalar(value, normalize_values=normalize_values)}


def flatten_raw(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_raw(child, child_prefix))
        return out
    if isinstance(value, list):
        out = {}
        for i, child in enumerate(value):
            out.update(flatten_raw(child, f"{prefix}[{i}]"))
        return out
    return {prefix: value}


def f1_score(matches: int, predicted: int, target: int) -> tuple[float, float, float]:
    precision = matches / predicted if predicted else 0.0
    recall = matches / target if target else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score_record(
    record: dict[str, Any],
    repair_json_string_newlines: bool = False,
    normalize_values: bool = False,
    include_diagnostics: bool = False,
    max_error_examples: int = 20,
    excluded_field_types: set[str] | None = None,
    excluded_path_patterns: list[re.Pattern] | None = None,
) -> dict[str, Any]:
    target = extract_json_object(
        record["target"],
        repair_json_string_newlines=repair_json_string_newlines,
    )
    prediction = extract_json_object(
        record["prediction"],
        repair_json_string_newlines=repair_json_string_newlines,
    )

    flat_target_all = flatten(target, normalize_values=normalize_values)
    flat_prediction_all = flatten(prediction, normalize_values=normalize_values)
    raw_target_all = flatten_raw(target)
    raw_prediction_all = flatten_raw(prediction)

    ignored_target_field_count = sum(1 for path in flat_target_all if ignored_path(path))
    ignored_prediction_field_count = sum(1 for path in flat_prediction_all if ignored_path(path))
    flat_target = filter_ignored_paths(flat_target_all)
    flat_prediction = filter_ignored_paths(flat_prediction_all)
    raw_target = filter_ignored_paths(raw_target_all)
    raw_prediction = filter_ignored_paths(raw_prediction_all)

    excluded_target_field_count = 0
    excluded_prediction_field_count = 0
    if excluded_field_types or excluded_path_patterns:
        (
            flat_target,
            flat_prediction,
            raw_target,
            raw_prediction,
            excluded_target_field_count,
            excluded_prediction_field_count,
        ) = exclude_metric_paths(
            flat_target,
            flat_prediction,
            raw_target,
            raw_prediction,
            excluded_field_types or set(),
            excluded_path_patterns or [],
        )

    target_items = set(flat_target.items())
    prediction_items = set(flat_prediction.items())
    exact_items = target_items & prediction_items
    field_precision, field_recall, field_f1 = f1_score(
        len(exact_items),
        len(prediction_items),
        len(target_items),
    )

    target_paths = set(flat_target)
    prediction_paths = set(flat_prediction)
    exact_paths = target_paths & prediction_paths
    path_precision, path_recall, path_f1 = f1_score(
        len(exact_paths),
        len(prediction_paths),
        len(target_paths),
    )

    field_scores = []
    for path in sorted(target_paths):
        expected_value = raw_target[path]
        extracted_value = raw_prediction.get(path, MISSING)
        comparison = type_aware_compare(path, extracted_value, expected_value)
        score_value = float(comparison["score"])
        field_scores.append(
            {
                "path": path,
                "expected": expected_value,
                "prediction": None if extracted_value is MISSING else extracted_value,
                "missing": extracted_value is MISSING,
                "score": score_value,
                "match": score_value >= 0.9,
                "field_type": comparison["field_type"],
                "method": comparison["method"],
            }
        )
    score_sum = sum(field["score"] for field in field_scores)
    match_count = sum(1 for field in field_scores if field["match"])
    evaluated_field_count = len(field_scores)

    score = {
        "json_valid": True,
        "ignored_target_field_count": ignored_target_field_count,
        "ignored_prediction_field_count": ignored_prediction_field_count,
        "excluded_target_field_count": excluded_target_field_count,
        "excluded_prediction_field_count": excluded_prediction_field_count,
        "evaluated_field_count": evaluated_field_count,
        "field_score_sum": score_sum,
        "mean_field_score": score_sum / evaluated_field_count if evaluated_field_count else 0.0,
        "field_match_count": match_count,
        "field_match_rate": match_count / evaluated_field_count if evaluated_field_count else 0.0,
        "target_field_count": len(target_items),
        "prediction_field_count": len(prediction_items),
        "exact_field_matches": len(exact_items),
        "field_precision": field_precision,
        "field_recall": field_recall,
        "field_f1": field_f1,
        "target_path_count": len(target_paths),
        "prediction_path_count": len(prediction_paths),
        "exact_path_matches": len(exact_paths),
        "path_precision": path_precision,
        "path_recall": path_recall,
        "path_f1": path_f1,
    }
    if not include_diagnostics:
        score["_field_scores"] = field_scores
        return score

    missing_paths = sorted(target_paths - prediction_paths)
    extra_paths = sorted(prediction_paths - target_paths)
    low_score_fields = sorted(
        (field for field in field_scores if field["score"] < 0.9),
        key=lambda field: (field["score"], field["path"]),
    )
    value_mismatch_paths = sorted(path for path in exact_paths if flat_target[path] != flat_prediction[path])
    score["_diagnostics"] = {
        "missing_path_count": len(missing_paths),
        "extra_path_count": len(extra_paths),
        "low_score_field_count": len(low_score_fields),
        "value_mismatch_count": len(value_mismatch_paths),
        "missing_paths": missing_paths[:max_error_examples],
        "extra_paths": extra_paths[:max_error_examples],
        "low_score_fields": [
            {
                "path": field["path"],
                "score": field["score"],
                "field_type": field["field_type"],
                "method": field["method"],
                "expected": truncate_value(normalize_scalar(field["expected"], True)),
                "prediction": None
                if field["missing"]
                else truncate_value(normalize_scalar(field["prediction"], True)),
            }
            for field in low_score_fields[:max_error_examples]
        ],
        "value_mismatches": [
            {
                "path": path,
                "target": truncate_value(flat_target[path]),
                "prediction": truncate_value(flat_prediction[path]),
            }
            for path in value_mismatch_paths[:max_error_examples]
        ],
        "_missing_paths_all": missing_paths,
        "_extra_paths_all": extra_paths,
        "_value_mismatch_paths_all": value_mismatch_paths,
    }
    score["_field_scores"] = field_scores
    return score


def main() -> None:
    args = parse_args()
    excluded_field_types = parse_excluded_field_types(args.exclude_field_types)
    if args.include_checkbox_fields:
        excluded_field_types.discard("checkbox")
    excluded_path_patterns = [re.compile(pattern) for pattern in args.exclude_path_regex]
    rows = []
    field_rows = []
    diagnostic_rows = []
    errors = Counter()
    missing_path_counts = Counter()
    extra_path_counts = Counter()
    value_mismatch_path_counts = Counter()
    field_type_scores = {}
    method_counts = Counter()

    with Path(args.predictions).open() as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                score = score_record(
                    record,
                    repair_json_string_newlines=args.repair_json_string_newlines,
                    normalize_values=args.normalize_values,
                    include_diagnostics=bool(args.error_output),
                    max_error_examples=args.max_error_examples,
                    excluded_field_types=excluded_field_types,
                    excluded_path_patterns=excluded_path_patterns,
                )
            except Exception as exc:
                score = {
                    "json_valid": False,
                    "ignored_target_field_count": 0,
                    "ignored_prediction_field_count": 0,
                    "excluded_target_field_count": 0,
                    "excluded_prediction_field_count": 0,
                    "evaluated_field_count": 0,
                    "field_score_sum": 0.0,
                    "mean_field_score": 0.0,
                    "field_match_count": 0,
                    "field_match_rate": 0.0,
                    "target_field_count": 0,
                    "prediction_field_count": 0,
                    "exact_field_matches": 0,
                    "field_precision": 0.0,
                    "field_recall": 0.0,
                    "field_f1": 0.0,
                    "target_path_count": 0,
                    "prediction_path_count": 0,
                    "exact_path_matches": 0,
                    "path_precision": 0.0,
                    "path_recall": 0.0,
                    "path_f1": 0.0,
                    "error": str(exc),
                }
                errors[type(exc).__name__] += 1
                if args.error_output:
                    diagnostic_rows.append(
                        {
                            "line": line_number,
                            "row_id": record.get("row_id"),
                            "target_source": record.get("target_source"),
                            "json_valid": False,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
            field_scores = score.pop("_field_scores", [])
            for field_score in field_scores:
                field_type_scores.setdefault(field_score["field_type"], []).append(field_score["score"])
                method_counts[field_score["method"]] += 1
                if args.field_output:
                    field_rows.append(
                        {
                            "line": line_number,
                            "row_id": record.get("row_id"),
                            "target_source": record.get("target_source"),
                            **field_score,
                        }
                    )
            diagnostics = score.pop("_diagnostics", None)
            if diagnostics is not None:
                for path in diagnostics.pop("_missing_paths_all"):
                    missing_path_counts[path] += 1
                for path in diagnostics.pop("_extra_paths_all"):
                    extra_path_counts[path] += 1
                for path in diagnostics.pop("_value_mismatch_paths_all"):
                    value_mismatch_path_counts[path] += 1
                diagnostic_rows.append(
                    {
                        "line": line_number,
                        "row_id": record.get("row_id"),
                        "target_source": record.get("target_source"),
                        "json_valid": True,
                        **diagnostics,
                    }
                )
            rows.append(
                {
                    "line": line_number,
                    "row_id": record.get("row_id"),
                    "target_source": record.get("target_source"),
                    **score,
                }
            )

    total = len(rows)
    valid = sum(1 for row in rows if row["json_valid"])
    valid_rows = [row for row in rows if row["json_valid"]]
    total_evaluated_fields = sum(row.get("evaluated_field_count", 0) for row in rows)
    total_ignored_target_fields = sum(row.get("ignored_target_field_count", 0) for row in rows)
    total_ignored_prediction_fields = sum(
        row.get("ignored_prediction_field_count", 0) for row in rows
    )
    total_excluded_target_fields = sum(row.get("excluded_target_field_count", 0) for row in rows)
    total_excluded_prediction_fields = sum(
        row.get("excluded_prediction_field_count", 0) for row in rows
    )
    total_field_score = sum(row.get("field_score_sum", 0.0) for row in rows)
    total_field_matches = sum(row.get("field_match_count", 0) for row in rows)
    summary = {
        "prediction_file": args.predictions,
        "count": total,
        "json_valid_count": valid,
        "json_valid_rate": valid / total if total else 0.0,
        "total_evaluated_fields": total_evaluated_fields,
        "total_ignored_target_fields": total_ignored_target_fields,
        "total_ignored_prediction_fields": total_ignored_prediction_fields,
        "total_excluded_target_fields": total_excluded_target_fields,
        "total_excluded_prediction_fields": total_excluded_prediction_fields,
        "overall_field_score": total_field_score / total_evaluated_fields
        if total_evaluated_fields
        else 0.0,
        "overall_field_match_rate": total_field_matches / total_evaluated_fields
        if total_evaluated_fields
        else 0.0,
        "mean_submission_field_score": sum(row.get("mean_field_score", 0.0) for row in rows) / total
        if total
        else 0.0,
        "mean_valid_submission_field_score": sum(row.get("mean_field_score", 0.0) for row in valid_rows) / valid
        if valid
        else 0.0,
        "mean_submission_field_match_rate": sum(row.get("field_match_rate", 0.0) for row in rows) / total
        if total
        else 0.0,
        "mean_field_precision": sum(row["field_precision"] for row in rows) / total if total else 0.0,
        "mean_field_recall": sum(row["field_recall"] for row in rows) / total if total else 0.0,
        "mean_field_f1": sum(row["field_f1"] for row in rows) / total if total else 0.0,
        "mean_valid_field_f1": sum(row["field_f1"] for row in valid_rows) / valid if valid else 0.0,
        "mean_path_precision": sum(row["path_precision"] for row in rows) / total if total else 0.0,
        "mean_path_recall": sum(row["path_recall"] for row in rows) / total if total else 0.0,
        "mean_path_f1": sum(row["path_f1"] for row in rows) / total if total else 0.0,
        "mean_valid_path_f1": sum(row["path_f1"] for row in valid_rows) / valid if valid else 0.0,
        "repair_json_string_newlines": args.repair_json_string_newlines,
        "normalize_values": args.normalize_values,
        "excluded_field_types": sorted(excluded_field_types),
        "excluded_path_regex": args.exclude_path_regex,
        "errors": dict(errors),
        "field_type_scores": {
            field_type: {
                "count": len(scores),
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
                "match_rate": sum(1 for score in scores if score >= 0.9) / len(scores)
                if scores
                else 0.0,
            }
            for field_type, scores in sorted(field_type_scores.items())
        },
        "comparison_methods": dict(method_counts),
    }
    if args.error_output:
        summary["top_missing_paths"] = missing_path_counts.most_common(20)
        summary["top_extra_paths"] = extra_path_counts.most_common(20)
        summary["top_value_mismatch_paths"] = value_mismatch_path_counts.most_common(20)

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.row_output:
        row_output_path = Path(args.row_output)
        row_output_path.parent.mkdir(parents=True, exist_ok=True)
        with row_output_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    if args.field_output:
        field_output_path = Path(args.field_output)
        field_output_path.parent.mkdir(parents=True, exist_ok=True)
        with field_output_path.open("w") as f:
            for row in field_rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    if args.error_output:
        error_output_path = Path(args.error_output)
        error_output_path.parent.mkdir(parents=True, exist_ok=True)
        error_report = {
            "summary": summary,
            "rows": diagnostic_rows,
        }
        error_output_path.write_text(
            json.dumps(error_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
