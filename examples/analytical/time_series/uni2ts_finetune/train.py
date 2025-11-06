#!/usr/bin/env python3
"""
Fine-tune Salesforce Uni2TS on a GluonTS benchmark dataset.

The script prepares a small slice of the selected dataset, converts it into the
wide CSV format expected by Uni2TS, and then launches the packaged training CLI.
"""

import argparse
import json
import os
import shutil
import subprocess
from itertools import islice
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
from dotenv import load_dotenv
from datasets import load_from_disk
from gluonts.dataset.repository import datasets as gluon_datasets
from uni2ts.data.builder.simple import SimpleDatasetBuilder, SimpleEvalDatasetBuilder

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
UNI2TS_DATA_DIR = DATA_DIR / "uni2ts_cache"
WIDE_CSV_PATH = DATA_DIR / "train_wide.csv"
METADATA_PATH = ARTIFACTS_DIR / "dataset_metadata.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
HYDRA_OUTPUT_DIR = ARTIFACTS_DIR / "uni2ts_runs"
UNI2TS_REPO_DIR = DATA_DIR / "uni2ts_src"
UNI2TS_REPO_URL = os.environ.get("UNI2TS_REPO_URL", "https://github.com/SalesforceAIResearch/uni2ts.git")
UNI2TS_REPO_REF = os.environ.get("UNI2TS_REPO_REF", "1.2.0")
DATA_CONFIG_TEMPLATE = """_target_: uni2ts.data.builder.simple.SimpleDatasetBuilder\ndataset: {dataset_name}\nweight: 1\n"""
VAL_CONFIG_TEMPLATE = """_target_: uni2ts.data.builder.simple.SimpleEvalDatasetBuilder\ndataset: {eval_dataset}\noffset: {offset}\nwindows: {windows}\ndistance: {distance}\nprediction_length: {prediction_length}\ncontext_length: {context_length}\npatch_size: {patch_size}\n"""
DEFAULT_TRAIN_OVERRIDES = [
    "trainer.max_epochs=1",
    "train_dataloader.num_batches_per_epoch=10",
    "train_dataloader.batch_size=32",
    "train_dataloader.num_workers=1",
    "train_dataloader.prefetch_factor=1",
    "val_dataloader.num_workers=1",
    "val_dataloader.prefetch_factor=1",
    "+trainer.num_sanity_val_steps=0",
]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Fine-tune Uni2TS on a GluonTS dataset.")
    parser.add_argument(
        "--dataset",
        default="m4_hourly",
        help="Name of the GluonTS repository dataset to materialize locally.",
    )
    parser.add_argument(
        "--model-config",
        default="moirai_1.0_R_small",
        help="Hydra model config name from uni2ts/cli/conf/finetune/model.",
    )
    parser.add_argument(
        "--max-series",
        type=int,
        default=100,
        help="Maximum number of training series to materialize locally.",
    )
    parser.add_argument(
        "--context-steps",
        type=int,
        default=384,
        help="Number of most recent timesteps to keep per series (<=0 disables truncation).",
    )
    parser.add_argument(
        "--val-patch-size",
        type=int,
        default=16,
        help="Patch size (timesteps) for validation windows (<=0 lets the script choose).",
    )
    parser.add_argument(
        "--val-context-length",
        type=int,
        default=0,
        help="Context length for validation windows (<=0 lets the script choose).",
    )
    return parser.parse_known_args()


def entry_to_series(entry: Dict, freq: str) -> pd.Series:
    freq = freq.upper()
    start = pd.Period(entry["start"], freq=freq)
    index = pd.period_range(start=start, periods=len(entry["target"]), freq=freq)
    return pd.Series(entry["target"], index=index)


def load_env_vars() -> None:
    # Attempt to load a local .env first, then fall back to the mounted repo root.
    candidate_paths = [SCRIPT_DIR / ".env"]
    candidate_paths.extend(parent / ".env" for parent in SCRIPT_DIR.parents)
    candidate_paths.append(Path("/cvlization_repo/.env"))
    for path in candidate_paths:
        if path and path.exists():
            load_dotenv(path, override=False)
            break


def materialize_dataset(dataset_name: str, max_series: int, context_steps: int) -> Tuple[str, str, int, int]:
    print(f"[dataset] Loading {dataset_name!r} from GluonTS repository …", flush=True)
    dataset = gluon_datasets.get_dataset(dataset_name, regenerate=False)
    freq = str(dataset.metadata.freq).upper()
    prediction_length = dataset.metadata.prediction_length

    frames: Dict[str, pd.Series] = {}
    for idx, entry in enumerate(islice(dataset.train, max_series)):
        series = entry_to_series(entry, freq)
        if context_steps > 0:
            series = series.iloc[-context_steps:]
        frames[f"series_{idx:04d}"] = series

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(frames)
    df.index.name = "timestamp"
    df.to_csv(WIDE_CSV_PATH)
    print(f"[dataset] Wrote {len(frames)} series to {WIDE_CSV_PATH}", flush=True)
    series_length = len(df)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": dataset_name,
        "frequency": freq,
        "prediction_length": prediction_length,
        "series_count": len(frames),
        "series_length": series_length,
        "csv_path": str(WIDE_CSV_PATH),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return dataset_name, freq, prediction_length, series_length


def run_command(command: Sequence[str], cwd: Path, env_updates: Optional[Dict[str, str]] = None) -> None:
    print(f"[command] {' '.join(command)}", flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    if env_updates:
        env.update(env_updates)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def build_training_dataset(dataset_name: str, freq: str) -> None:
    dataset_path = UNI2TS_DATA_DIR / dataset_name
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    builder = SimpleDatasetBuilder(dataset=dataset_name, storage_path=UNI2TS_DATA_DIR)
    builder.build_dataset(Path(WIDE_CSV_PATH), dataset_type="wide", freq=freq)


def build_validation_dataset(
    dataset_name: str,
    freq: str,
    prediction_length: int,
    series_length: int,
    patch_size_arg: int,
    context_length_arg: int,
) -> Tuple[str, Dict[str, object]]:
    eval_dataset = f"{dataset_name}_eval"
    dataset_path = UNI2TS_DATA_DIR / eval_dataset
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    auto_patch = max(8, min(32, prediction_length))
    patch_size = patch_size_arg if patch_size_arg > 0 else auto_patch
    patch_size = max(1, min(patch_size, prediction_length * 2))
    windows = 1
    if context_length_arg > 0:
        context_length = context_length_arg
    else:
        context_length = min(series_length - prediction_length - patch_size, prediction_length * 4)

    max_context = max(patch_size + 1, series_length - prediction_length - 1)
    context_length = max(patch_size, min(context_length, max_context))
    offset = context_length
    builder = SimpleEvalDatasetBuilder(
        dataset=eval_dataset,
        offset=offset,
        windows=windows,
        distance=prediction_length,
        prediction_length=prediction_length,
        context_length=context_length,
        patch_size=patch_size,
        storage_path=UNI2TS_DATA_DIR,
    )
    builder.build_dataset(Path(WIDE_CSV_PATH), dataset_type="wide", freq=freq)

    params = {
        "eval_dataset": eval_dataset,
        "offset": offset,
        "windows": windows,
        "distance": prediction_length,
        "prediction_length": prediction_length,
        "context_length": context_length,
        "patch_size": patch_size,
    }
    return eval_dataset, params


def normalize_dataset_freq(dataset_name: str) -> None:
    dataset_path = UNI2TS_DATA_DIR / dataset_name
    if not dataset_path.exists():
        return

    ds = load_from_disk(str(dataset_path))
    unique_freqs = {str(freq) for freq in ds.unique("freq")}
    if all(freq.upper() == freq for freq in unique_freqs):
        return

    ds = ds.map(lambda example: {"freq": str(example["freq"]).upper()})
    tmp_path = dataset_path.parent / f"{dataset_name}_tmp"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    ds.save_to_disk(str(tmp_path))
    shutil.rmtree(dataset_path)
    tmp_path.rename(dataset_path)
    print(f"[dataset] Normalized frequency tokens to uppercase for {dataset_name}", flush=True)


def ensure_uni2ts_cli_repo() -> Path:
    if (UNI2TS_REPO_DIR / "cli").exists():
        return UNI2TS_REPO_DIR

    if UNI2TS_REPO_DIR.exists():
        print(f"[repo] Using pre-existing Uni2TS checkout at {UNI2TS_REPO_DIR}", flush=True)
        return UNI2TS_REPO_DIR

    print(f"[repo] Cloning Uni2TS ({UNI2TS_REPO_REF}) from {UNI2TS_REPO_URL}", flush=True)
    run_command(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            UNI2TS_REPO_REF,
            UNI2TS_REPO_URL,
            str(UNI2TS_REPO_DIR),
        ],
        cwd=DATA_DIR,
    )
    return UNI2TS_REPO_DIR


def ensure_dataset_config(
    repo_root: Path, dataset_name: str, eval_params: Dict[str, object]
) -> None:
    data_dir = repo_root / "cli" / "conf" / "finetune" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_path = data_dir / f"{dataset_name}.yaml"
    desired = DATA_CONFIG_TEMPLATE.format(dataset_name=dataset_name)
    if not config_path.exists() or config_path.read_text() != desired:
        config_path.write_text(desired)
        print(f"[config] Wrote dataset config {config_path}", flush=True)
    val_dir = repo_root / "cli" / "conf" / "finetune" / "val_data"
    val_dir.mkdir(parents=True, exist_ok=True)
    val_config_path = val_dir / f"{dataset_name}.yaml"
    desired_val = VAL_CONFIG_TEMPLATE.format(**eval_params)
    if not val_config_path.exists() or val_config_path.read_text() != desired_val:
        val_config_path.write_text(desired_val)
        print(f"[config] Wrote validation config {val_config_path}", flush=True)


def run_finetuning(
    dataset_name: str, model_config: str, repo_root: Path, extra_overrides: Sequence[str]
) -> None:
    HYDRA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_name = f"uni2ts_finetune_{dataset_name}"
    pythonpath = os.pathsep.join(
        filter(None, [str(SCRIPT_DIR), str(repo_root), os.environ.get("PYTHONPATH", "")])
    )
    overrides = [
        f"run_name={run_name}",
        f"model={model_config}",
        f"data={dataset_name}",
        f"val_data={dataset_name}",
        f"hydra.run.dir={HYDRA_OUTPUT_DIR}",
        *DEFAULT_TRAIN_OVERRIDES,
        *extra_overrides,
    ]
    run_command(
        [
            "python",
            "-m",
            "cli.train",
            "-cp",
            "conf/finetune",
            *overrides,
        ],
        cwd=repo_root,
        env_updates={"PYTHONPATH": pythonpath},
    )


def main() -> None:
    args, extra_overrides = parse_args()
    load_env_vars()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UNI2TS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUSTOM_DATA_PATH", str(UNI2TS_DATA_DIR))
    repo_root = ensure_uni2ts_cli_repo()
    dataset_name, freq, prediction_length, series_length = materialize_dataset(
        args.dataset, args.max_series, args.context_steps
    )
    build_training_dataset(dataset_name, freq)
    eval_dataset, eval_params = build_validation_dataset(
        dataset_name,
        freq,
        prediction_length,
        series_length,
        args.val_patch_size,
        args.val_context_length,
    )
    normalize_dataset_freq(dataset_name)
    normalize_dataset_freq(eval_dataset)
    ensure_dataset_config(repo_root, dataset_name, eval_params)
    run_finetuning(dataset_name, args.model_config, repo_root, extra_overrides)

    metrics = {
        "status": "completed",
        "dataset": dataset_name,
        "frequency": freq,
        "model_config": args.model_config,
        "prepared_dataset_csv": str(WIDE_CSV_PATH),
        "hydra_output_dir": str(HYDRA_OUTPUT_DIR),
        "custom_data_path": str(UNI2TS_DATA_DIR),
        "uni2ts_repo": str(repo_root),
        "uni2ts_ref": UNI2TS_REPO_REF,
        "series_length": series_length,
        "context_steps": args.context_steps,
        "validation_dataset": eval_dataset,
        "validation_patch_size": eval_params["patch_size"],
        "validation_context_length": eval_params["context_length"],
        "hydra_overrides": DEFAULT_TRAIN_OVERRIDES + list(extra_overrides),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
