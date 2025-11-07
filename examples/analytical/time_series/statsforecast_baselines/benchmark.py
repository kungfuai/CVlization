"""
Statistical Forecasting Baselines Benchmark

Comprehensive benchmark of classical statistical forecasting methods using StatsForecast.
Compares AutoARIMA, AutoETS, Theta, and SeasonalNaive across multiple datasets.

Models: AutoARIMA, AutoETS, Theta, SeasonalNaive
Datasets: M4 (hourly), ETT-h1, Electricity (LTSF suite)
Metrics: sMAPE, MASE, RMSE
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from datasetsforecast.m4 import M4
from statsforecast.core import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    SeasonalNaive,
    Theta,
)

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark classical statistical forecasting methods"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["m4_hourly"],
        choices=["m4_hourly", "m4_daily", "ett_h1", "electricity"],
        help="Datasets to benchmark",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["AutoARIMA", "AutoETS", "Theta", "SeasonalNaive"],
        choices=["AutoARIMA", "AutoETS", "Theta", "SeasonalNaive", "AutoCES"],
        help="Models to benchmark",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Forecast horizon (uses dataset default if not specified)",
    )
    parser.add_argument(
        "--n-series",
        type=int,
        default=10,
        help="Number of series to sample from each dataset (for faster testing)",
    )
    parser.add_argument(
        "--season-length",
        type=int,
        default=None,
        help="Seasonal period (uses dataset default if not specified)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./artifacts",
        help="Output directory for results",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/root/.cache/cvlization/statsforecast_data",
        help="Directory to cache downloaded data",
    )
    return parser.parse_args()


def load_dataset(dataset_name: str, data_dir: str, n_series: int = None):
    """Load and prepare dataset."""
    print(f"\n{'='*60}")
    print(f"Loading {dataset_name} dataset...")
    print(f"{'='*60}")

    os.makedirs(data_dir, exist_ok=True)

    if dataset_name.startswith("m4_"):
        # M4 dataset
        freq_map = {"m4_hourly": "Hourly", "m4_daily": "Daily"}
        freq = freq_map.get(dataset_name, "Hourly")

        Y_df, *_ = M4.load(directory=data_dir, group=freq)

        # M4 default horizons and seasons
        horizon_map = {"Hourly": 48, "Daily": 14}
        season_map = {"Hourly": 24, "Daily": 7}

        horizon = horizon_map[freq]
        season_length = season_map[freq]
        # M4 converts to datetime internally
        freq_str = "H" if freq == "Hourly" else "D"

    else:
        # LTSF datasets (ETT, Electricity, etc.)
        raise NotImplementedError(f"Dataset {dataset_name} not yet implemented")

    # Sample n_series if specified
    if n_series is not None:
        unique_ids = Y_df["unique_id"].unique()
        if len(unique_ids) > n_series:
            sample_ids = np.random.choice(unique_ids, size=n_series, replace=False)
            Y_df = Y_df[Y_df["unique_id"].isin(sample_ids)]

    print(f"✓ Loaded {len(Y_df['unique_id'].unique())} series")
    print(f"✓ Date range: {Y_df['ds'].min()} to {Y_df['ds'].max()}")
    print(f"✓ Horizon: {horizon} periods")
    print(f"✓ Season length: {season_length}")
    print(f"✓ Frequency: {freq_str}")

    return Y_df, horizon, season_length, freq_str


def split_data(Y_df: pd.DataFrame, horizon: int):
    """Split data into train and test sets."""
    print(f"\n{'='*60}")
    print(f"Splitting data...")
    print(f"{'='*60}")

    Y_test_df = Y_df.groupby("unique_id").tail(horizon)
    Y_train_df = Y_df.drop(Y_test_df.index)

    print(f"✓ Training set: {len(Y_train_df)} observations")
    print(f"✓ Test set: {len(Y_test_df)} observations")

    return Y_train_df, Y_test_df


def initialize_models(model_names: list, season_length: int):
    """Initialize forecast models."""
    print(f"\n{'='*60}")
    print(f"Initializing models...")
    print(f"{'='*60}")

    models = []

    if "AutoARIMA" in model_names:
        models.append(AutoARIMA(season_length=season_length))
        print("✓ AutoARIMA (automatic ARIMA with seasonality)")

    if "AutoETS" in model_names:
        models.append(AutoETS(season_length=season_length))
        print("✓ AutoETS (automatic exponential smoothing)")

    if "Theta" in model_names:
        models.append(Theta(season_length=season_length))
        print("✓ Theta (Theta method)")

    if "SeasonalNaive" in model_names:
        models.append(SeasonalNaive(season_length=season_length))
        print("✓ SeasonalNaive (basic seasonal baseline)")

    return models


def generate_forecasts(
    Y_train_df: pd.DataFrame,
    models: list,
    horizon: int,
    freq: str,
):
    """Generate forecasts using all models."""
    print(f"\n{'='*60}")
    print(f"Generating forecasts...")
    print(f"{'='*60}")

    # Check if original data has integer indices (M4 style)
    # M4 data uses object dtype with integer values
    has_integer_ds = (
        Y_train_df["ds"].dtype in [np.int64, np.int32] or
        (Y_train_df["ds"].dtype == object and isinstance(Y_train_df["ds"].iloc[0], (int, np.integer)))
    )

    print(f"Forecasting {horizon} periods ahead...")
    sf = StatsForecast(models=models, freq=freq, n_jobs=-1)
    Y_hat_df = sf.forecast(df=Y_train_df, h=horizon)

    # Reset index to get unique_id and ds as columns
    if "unique_id" not in Y_hat_df.columns:
        Y_hat_df = Y_hat_df.reset_index()

    # For M4 data with integer indices, reconstruct the integer sequence
    if has_integer_ds:
        # Get max ds value per series from training data
        max_ds_per_series = Y_train_df.groupby("unique_id")["ds"].max()

        # Create proper integer indices for each series
        new_ds = []
        for uid in Y_hat_df["unique_id"].unique():
            series_max = max_ds_per_series[uid]
            # Forecast indices start from max+1 and go to max+horizon
            forecast_indices = range(series_max + 1, series_max + 1 + horizon)
            new_ds.extend(forecast_indices)

        Y_hat_df["ds"] = new_ds

    print(f"✓ Generated forecasts for {len(Y_hat_df['unique_id'].unique())} series")

    return Y_hat_df


def evaluate_forecasts(
    Y_test_df: pd.DataFrame,
    Y_hat_df: pd.DataFrame,
    model_names: list,
):
    """Evaluate forecast accuracy."""
    print(f"\n{'='*60}")
    print(f"Evaluating forecasts...")
    print(f"{'='*60}")

    # Merge test data with forecasts
    Y_test_df = Y_test_df.rename(columns={"y": "y_test"})
    evaluation_df = Y_test_df.merge(Y_hat_df, on=["unique_id", "ds"], how="left")

    # Calculate metrics for each model
    results = {}

    for model_name in model_names:
        if model_name in evaluation_df.columns:
            y_true = evaluation_df["y_test"].values
            y_pred = evaluation_df[model_name].values

            # Remove NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true = y_true[mask]
            y_pred = y_pred[mask]

            if len(y_true) > 0:
                # RMSE
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

                # MAE
                mae = np.mean(np.abs(y_true - y_pred))

                # sMAPE (symmetric MAPE)
                smape = 100 * np.mean(
                    2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
                )

                # MASE (Mean Absolute Scaled Error) - using naive forecast as baseline
                # For simplicity, using MAE / mean(|y_true|) as approximation
                naive_mae = np.mean(np.abs(y_true))
                mase = mae / (naive_mae + 1e-8)

                results[model_name] = {
                    "RMSE": rmse,
                    "MAE": mae,
                    "sMAPE": smape,
                    "MASE": mase,
                }

    # Convert to DataFrame
    if not results:
        print("⚠ No results to display")
        return pd.DataFrame()

    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values("RMSE")

    print("\n" + "="*80)
    print("FORECAST ACCURACY (sorted by RMSE)")
    print("="*80)
    print(results_df.to_string())
    print("="*80)

    return results_df


def run_benchmark(args):
    """Run complete benchmark."""
    all_results = {}

    print("\n" + "="*80)
    print("STATISTICAL FORECASTING BASELINES BENCHMARK")
    print("="*80)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Number of series per dataset: {args.n_series}")
    print("="*80)

    for dataset_name in args.datasets:
        print(f"\n{'#'*80}")
        print(f"# DATASET: {dataset_name.upper()}")
        print(f"{'#'*80}")

        # Load dataset
        Y_df, horizon, season_length, freq = load_dataset(
            dataset_name, args.data_dir, args.n_series
        )

        # Override defaults if specified
        if args.horizon is not None:
            horizon = args.horizon
        if args.season_length is not None:
            season_length = args.season_length

        # Split data
        Y_train_df, Y_test_df = split_data(Y_df, horizon)

        # Initialize models
        models = initialize_models(args.models, season_length)

        # Generate forecasts
        Y_hat_df = generate_forecasts(Y_train_df, models, horizon, freq)

        # Evaluate
        results_df = evaluate_forecasts(Y_test_df, Y_hat_df, args.models)

        # Store results
        all_results[dataset_name] = results_df

    return all_results


def save_results(all_results: dict, args, output_dir: str):
    """Save benchmark results."""
    print(f"\n{'='*60}")
    print(f"Saving results to {output_dir}...")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # Save per-dataset results
    for dataset_name, results_df in all_results.items():
        output_path = Path(output_dir) / f"{dataset_name}_results.csv"
        results_df.to_csv(output_path)
        print(f"✓ Saved {dataset_name}: {output_path}")

    # Create comparative summary
    summary_data = []
    for dataset_name, results_df in all_results.items():
        for model_name in results_df.index:
            row = {"Dataset": dataset_name, "Model": model_name}
            row.update(results_df.loc[model_name].to_dict())
            summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_path = Path(output_dir) / "benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved summary: {summary_path}")

    # Print comparative summary
    print(f"\n{'='*80}")
    print("COMPARATIVE SUMMARY ACROSS ALL DATASETS")
    print(f"{'='*80}")

    # Best model per dataset
    print("\nBest Model per Dataset (by RMSE):")
    print("-" * 80)
    for dataset_name, results_df in all_results.items():
        best_model = results_df.index[0]
        best_rmse = results_df.iloc[0]["RMSE"]
        print(f"{dataset_name:20s} → {best_model:15s} (RMSE: {best_rmse:.4f})")

    # Average rankings
    print(f"\n{'='*80}")
    print("Average Performance Across Datasets:")
    print(f"{'='*80}")
    avg_metrics = summary_df.groupby("Model")[["RMSE", "MAE", "sMAPE", "MASE"]].mean()
    avg_metrics = avg_metrics.sort_values("RMSE")
    print(avg_metrics.to_string())
    print(f"{'='*80}")

    # Save configuration
    config_path = Path(output_dir) / "config.json"
    config = {
        "datasets": args.datasets,
        "models": args.models,
        "n_series": args.n_series,
        "horizon": args.horizon,
        "season_length": args.season_length,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved config: {config_path}")

    print(f"\n{'='*60}")
    print("✓ Benchmark complete!")
    print(f"{'='*60}\n")


def main():
    args = parse_args()

    # Run benchmark
    all_results = run_benchmark(args)

    # Save results
    save_results(all_results, args, args.output_dir)


if __name__ == "__main__":
    main()
