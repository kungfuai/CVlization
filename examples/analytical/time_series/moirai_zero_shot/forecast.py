import argparse
import json
import os
from itertools import islice
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.repository import datasets
from gluonts.dataset.split import split
from gluonts.evaluation import Evaluator
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

ARTIFACTS_DIR = Path("artifacts")
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
SAMPLE_FORECAST_PATH = ARTIFACTS_DIR / "sample_forecast.csv"
PLOT_PATH = ARTIFACTS_DIR / "sample_forecast.png"


def entry_to_series(entry: Dict, freq: str, *, as_period: bool = False) -> pd.Series:
    """Convert a GluonTS data entry into a pandas Series with either datetime or period index."""
    start = entry["start"]
    if as_period:
        if not isinstance(start, pd.Period):
            start = pd.Period(start, freq=freq)
        index = pd.period_range(start=start, periods=len(entry["target"]), freq=freq)
    else:
        if hasattr(start, "to_timestamp"):
            start = start.to_timestamp()
        index = pd.date_range(start=start, periods=len(entry["target"]), freq=freq)
    return pd.Series(entry["target"], index=index)


def select_series(dataset, series_limit: int) -> pd.DataFrame:
    freq = dataset.metadata.freq
    series_frames: Dict[str, pd.Series] = {}

    for idx, entry in enumerate(islice(dataset.train, series_limit)):
        series_frames[f"series_{idx}"] = entry_to_series(entry, freq)

    if not series_frames:
        raise RuntimeError("Dataset did not yield any series. Try increasing series_limit.")

    df = pd.DataFrame(series_frames)
    df.index.name = "timestamp"
    return df


def build_forecast_dataset(df: pd.DataFrame, prediction_length: int, windows: int) -> tuple:
    ds = PandasDataset(dict(df))
    test_horizon = prediction_length * windows
    max_len = len(df.index)
    if test_horizon >= max_len:
        raise ValueError(
            f"Requested {windows} windows of length {prediction_length}, "
            f"but series length is only {max_len}. Reduce windows or prediction length."
        )
    train, test_template = split(ds, offset=-test_horizon)
    test_data = test_template.generate_instances(
        prediction_length=prediction_length,
        windows=windows,
        distance=prediction_length,
    )
    return train, test_data


def compute_metrics(target_series: List[pd.Series], forecasts: List) -> Dict[str, float]:
    evaluator = Evaluator()
    agg_metrics, _ = evaluator(target_series, forecasts)
    # Retain a small subset of metrics to keep JSON compact
    metrics_of_interest = {
        "mean_wQuantileLoss": agg_metrics.get("mean_wQuantileLoss"),
        "MASE": agg_metrics.get("MASE"),
        "sMAPE": agg_metrics.get("sMAPE"),
        "OWA": agg_metrics.get("OWA"),
    }
    return {k: float(v) if v is not None and not np.isnan(v) else None for k, v in metrics_of_interest.items()}


def save_sample_forecast(target_series: List[pd.Series], forecasts: List, path_csv: Path, path_png: Path, dataset_name: str) -> None:
    if not forecasts:
        raise RuntimeError("No forecasts generated")
    truth_period = target_series[0]
    fc = forecasts[0]
    horizon = len(fc.mean)
    truth = truth_period.to_timestamp()
    timestamps = truth.index[-horizon:]
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "actual": truth.values[-horizon:],
            "p10": np.asarray(fc.quantile(0.1)),
            "p50": np.asarray(fc.quantile(0.5)),
            "p90": np.asarray(fc.quantile(0.9)),
        }
    )
    df.to_csv(path_csv, index=False)

    plt.figure(figsize=(8, 4))
    plt.plot(truth.index, truth.values, label="actual", linewidth=1.3)
    mean_pred = np.asarray(fc.quantile(0.5))
    lower = np.asarray(fc.quantile(0.1))
    upper = np.asarray(fc.quantile(0.9))

    plt.plot(timestamps, mean_pred, label="forecast", linewidth=1.5)
    plt.fill_between(timestamps, lower, upper, alpha=0.2, label="10-90% interval")
    plt.title(f"Moirai zero-shot forecast ({dataset_name})")
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-shot forecasting with Salesforce Moirai")
    parser.add_argument("--dataset", default="m4_hourly", help="GluonTS dataset name (default: m4_hourly)")
    parser.add_argument(
        "--model-family",
        default="moirai",
        choices=["moirai", "moirai2"],
        help="Pretrained model family to use (moirai 1.1-R or moirai 2.0-R)",
    )
    parser.add_argument("--model-size", default="small", choices=["small", "base", "large"], help="Moirai model size")
    parser.add_argument("--prediction-length", type=int, default=None, help="Forecast horizon. Defaults to dataset metadata")
    parser.add_argument("--context-length", type=int, default=512, help="Context window length passed to Moirai")
    parser.add_argument("--windows", type=int, default=2, help="Rolling windows to evaluate")
    parser.add_argument("--series-limit", type=int, default=4, help="Number of univariate series to sample from train set")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of forecast samples for quantiles")
    parser.add_argument("--patch-size", default="auto", help="Patch size (auto or integer)")
    parser.add_argument("--device", default="auto", help="Device for predictor (auto, cpu, cuda)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    dataset = datasets.get_dataset(args.dataset, regenerate=False)
    df = select_series(dataset, args.series_limit)
    default_horizon = dataset.metadata.prediction_length
    prediction_length = args.prediction_length or default_horizon
    train, test_data = build_forecast_dataset(df, prediction_length, args.windows)

    effective_context = min(args.context_length, len(df) - prediction_length * args.windows)
    effective_context = max(effective_context, prediction_length)

    patch_size = args.patch_size
    if isinstance(patch_size, str) and patch_size != "auto":
        patch_size = int(patch_size)

    if args.model_family == "moirai2":
        if args.model_size != "small":
            raise ValueError("Moirai 2.0 currently provides only the 'small' checkpoint.")
        module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
        model = Moirai2Forecast(
            module=module,
            prediction_length=prediction_length,
            context_length=effective_context,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
    else:
        module = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{args.model_size}")
        model = MoiraiForecast(
            module=module,
            prediction_length=prediction_length,
            context_length=effective_context,
            patch_size=patch_size,
            num_samples=args.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

    predictor = model.create_predictor(batch_size=args.batch_size, device=args.device)
    forecasts = list(predictor.predict(test_data.input))
    target_entries = list(test_data.label)
    target_series = [entry_to_series(entry, dataset.metadata.freq, as_period=True) for entry in target_entries]

    metrics = compute_metrics(target_series, forecasts)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_sample_forecast(target_series, forecasts, SAMPLE_FORECAST_PATH, PLOT_PATH, args.dataset)

    summary = {
        "dataset": args.dataset,
        "model_family": args.model_family,
        "model_size": args.model_size,
        "model": (
            "Salesforce/moirai-2.0-R-small"
            if args.model_family == "moirai2"
            else f"Salesforce/moirai-1.1-R-{args.model_size}"
        ),
        "prediction_length": prediction_length,
        "context_length": effective_context,
        "windows": args.windows,
        "series_used": args.series_limit,
    }
    with (ARTIFACTS_DIR / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Zero-shot forecasting complete. Metrics saved to", METRICS_PATH)


if __name__ == "__main__":
    main()
