import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and score held-out predictions for a base/full checkpoint or "
            "a LoRA adapter. Primary metrics exclude checkbox fields by default."
        )
    )
    parser.add_argument("--config", default="config.yaml", help="Eval/training config")
    parser.add_argument("--dataset-path", help="Override dataset.path from config")
    parser.add_argument("--data-files", help="Override dataset.data_files from config")
    parser.add_argument("--data-dir", help="Override dataset.data_dir from config")
    parser.add_argument(
        "--checkpoint",
        help=(
            "Base/full model checkpoint or Hugging Face model id. Omit to use "
            "model.name from the config."
        ),
    )
    parser.add_argument(
        "--model-name",
        help="Alias for --checkpoint; kept for parity with predict_eval.py.",
    )
    parser.add_argument(
        "--adapter",
        help="Optional LoRA adapter path. If set, the base model comes from config or --checkpoint.",
    )
    parser.add_argument("--attn-implementation", help="Override attention implementation")
    parser.add_argument("--max-seq-length", type=int, help="Override model.max_seq_length")
    parser.add_argument("--load-in-4bit", action="store_true", help="Force 4-bit model loading")
    parser.add_argument("--no-load-in-4bit", action="store_true", help="Disable 4-bit model loading")
    parser.add_argument("--output-dir", default="outputs/eval", help="Directory for eval artifacts")
    parser.add_argument("--name", default="checkpoint", help="Artifact name prefix")
    parser.add_argument(
        "--predictions",
        help=(
            "Existing predictions JSONL to score. If omitted, predictions are generated "
            "with predict_eval.py."
        ),
    )
    parser.add_argument(
        "--predict-script",
        default="predict_eval.py",
        choices=["predict_eval.py", "predict_eval_unsloth.py"],
        help="Prediction backend used when --predictions is omitted.",
    )
    parser.add_argument("--max-samples", type=int, default=64, help="Max eval rows to generate")
    parser.add_argument("--max-new-tokens", type=int, default=8192, help="Generation output budget")
    parser.add_argument("--max-target-tokens", type=int)
    parser.add_argument(
        "--sample-strategy",
        choices=["first", "document_id"],
        default="document_id",
        help="How to choose eval rows for generation",
    )
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--no-stop-on-balanced-json", action="store_true")
    parser.add_argument("--no-truncate-prompt", action="store_true")
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        help="Unsloth eval only: left-truncate prompt input IDs to this many tokens.",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Unsloth eval only: pass use_cache=False during generation.",
    )
    parser.add_argument(
        "--no-for-inference",
        action="store_true",
        help="Unsloth eval only: do not call FastLanguageModel.for_inference(model).",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        help="Unsloth eval only: use chunked prompt prefill for greedy generation.",
    )
    parser.add_argument(
        "--exclude-field-types",
        default="checkbox",
        help=(
            "Primary metric excluded field types. Defaults to checkbox. "
            "Use '' to disable field-type exclusions."
        ),
    )
    parser.add_argument(
        "--include-checkbox-fields",
        action="store_true",
        help="Include checkbox fields in the primary metrics.",
    )
    parser.add_argument(
        "--exclude-path-regex",
        action="append",
        default=[],
        help="Regex for flattened field paths to exclude from primary metrics.",
    )
    parser.add_argument(
        "--no-all-fields-metrics",
        action="store_true",
        help="Do not write the secondary checkbox-included metrics file.",
    )
    parser.add_argument(
        "--write-details",
        action="store_true",
        help="Write row/field/error diagnostics for primary metrics.",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def add_if_present(command: list[str], flag: str, value) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def generate_predictions(args: argparse.Namespace, predictions_path: Path) -> None:
    if args.predict_script == "predict_eval_unsloth.py":
        if not args.adapter:
            raise SystemExit("--adapter is required when --predict-script predict_eval_unsloth.py")
        command = [
            sys.executable,
            args.predict_script,
            "--config",
            args.config,
            "--output",
            str(predictions_path),
            "--adapter",
            args.adapter,
            "--max-samples",
            str(args.max_samples),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--sample-strategy",
            args.sample_strategy,
            "--sample-offset",
            str(args.sample_offset),
            "--sample-stride",
            str(args.sample_stride),
        ]
        add_if_present(command, "--dataset-path", args.dataset_path)
        add_if_present(command, "--data-files", args.data_files)
        add_if_present(command, "--data-dir", args.data_dir)
        add_if_present(command, "--attn-implementation", args.attn_implementation)
        add_if_present(command, "--max-target-tokens", args.max_target_tokens)
        add_if_present(command, "--max-prompt-tokens", args.max_prompt_tokens)
        add_if_present(command, "--prefill-chunk-size", args.prefill_chunk_size)
        if args.disable_cache:
            command.append("--disable-cache")
        if args.no_for_inference:
            command.append("--no-for-inference")
        if args.no_stop_on_balanced_json:
            command.append("--no-stop-on-balanced-json")
        run(command)
        return

    command = [
        sys.executable,
        args.predict_script,
        "--config",
        args.config,
        "--output",
        str(predictions_path),
        "--max-samples",
        str(args.max_samples),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--sample-strategy",
        args.sample_strategy,
        "--sample-offset",
        str(args.sample_offset),
        "--sample-stride",
        str(args.sample_stride),
    ]
    add_if_present(command, "--dataset-path", args.dataset_path)
    add_if_present(command, "--data-files", args.data_files)
    add_if_present(command, "--data-dir", args.data_dir)
    add_if_present(command, "--checkpoint", args.checkpoint or args.model_name)
    add_if_present(command, "--adapter", args.adapter)
    add_if_present(command, "--attn-implementation", args.attn_implementation)
    add_if_present(command, "--max-seq-length", args.max_seq_length)
    add_if_present(command, "--max-target-tokens", args.max_target_tokens)
    if args.load_in_4bit:
        command.append("--load-in-4bit")
    if args.no_load_in_4bit:
        command.append("--no-load-in-4bit")
    if args.no_stop_on_balanced_json:
        command.append("--no-stop-on-balanced-json")
    if args.no_truncate_prompt:
        command.append("--no-truncate-prompt")
    run(command)


def score_predictions(
    args: argparse.Namespace,
    predictions_path: Path,
    metrics_path: Path,
    *,
    include_checkbox_fields: bool,
    primary: bool,
) -> None:
    command = [
        sys.executable,
        "evaluate_predictions.py",
        str(predictions_path),
        "--output",
        str(metrics_path),
        "--repair-json-string-newlines",
        "--normalize-values",
    ]
    if include_checkbox_fields:
        command.append("--include-checkbox-fields")
    else:
        command.extend(["--exclude-field-types", args.exclude_field_types])
    if primary:
        for pattern in args.exclude_path_regex:
            command.extend(["--exclude-path-regex", pattern])
    if args.write_details and primary:
        stem = metrics_path.with_suffix("")
        command.extend(
            [
                "--row-output",
                str(stem.with_suffix(".rows.jsonl")),
                "--field-output",
                str(stem.with_suffix(".fields.jsonl")),
                "--error-output",
                str(stem.with_suffix(".errors.json")),
            ]
        )
    run(command)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = Path(args.predictions) if args.predictions else output_dir / f"{args.name}.predictions.jsonl"

    if args.predictions:
        print(f"Scoring existing predictions: {predictions_path}", flush=True)
    else:
        generate_predictions(args, predictions_path)

    primary_metrics_path = output_dir / f"{args.name}.metrics.json"
    score_predictions(
        args,
        predictions_path,
        primary_metrics_path,
        include_checkbox_fields=args.include_checkbox_fields,
        primary=True,
    )

    all_fields_metrics_path = None
    if not args.no_all_fields_metrics and not args.include_checkbox_fields:
        all_fields_metrics_path = output_dir / f"{args.name}.all-fields.metrics.json"
        score_predictions(
            args,
            predictions_path,
            all_fields_metrics_path,
            include_checkbox_fields=True,
            primary=False,
        )

    summary = {
        "predictions": str(predictions_path),
        "primary_metrics": str(primary_metrics_path),
        "primary_metric_excludes": {
            "field_types": []
            if args.include_checkbox_fields
            else [part.strip() for part in args.exclude_field_types.split(",") if part.strip()],
            "path_regex": args.exclude_path_regex,
        },
        "all_fields_metrics": str(all_fields_metrics_path) if all_fields_metrics_path else None,
    }
    summary_path = output_dir / f"{args.name}.eval-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
