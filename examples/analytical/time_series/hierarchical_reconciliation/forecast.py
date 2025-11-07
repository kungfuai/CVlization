"""
Hierarchical Time Series Forecasting with Reconciliation

This example demonstrates hierarchical forecasting using the Australian Tourism dataset.
It generates base forecasts using classical methods (AutoETS, AutoARIMA) and then
applies reconciliation methods (BottomUp, MinTrace) to ensure coherence across
the hierarchy.

Dataset: Australian Tourism (monthly, 366 series)
Hierarchy: Geographic regions × Travel purposes
Methods: BottomUp, MinTrace(OLS), MinTrace(shrinkage)
Metrics: RMSE, MASE
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import pandas as pd
from datasetsforecast.hierarchical import HierarchicalData
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.evaluation import HierarchicalEvaluation
from hierarchicalforecast.methods import BottomUp, MinTrace
from statsforecast.core import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hierarchical forecasting with reconciliation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="TourismSmall",
        choices=["TourismSmall", "TourismLarge"],
        help="Dataset to use (TourismSmall=366 series, TourismLarge=555 series)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Forecast horizon (number of periods)",
    )
    parser.add_argument(
        "--season-length",
        type=int,
        default=12,
        help="Seasonal period (12 for monthly data)",
    )
    parser.add_argument(
        "--base-models",
        type=str,
        nargs="+",
        default=["AutoETS", "AutoARIMA"],
        choices=["AutoETS", "AutoARIMA"],
        help="Base forecasting models to use",
    )
    parser.add_argument(
        "--reconcilers",
        type=str,
        nargs="+",
        default=["BottomUp", "MinTrace_ols", "MinTrace_shrink"],
        help="Reconciliation methods (BottomUp, MinTrace_ols, MinTrace_shrink)",
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
        default="/root/.cache/cvlization/hierarchical_data",
        help="Directory to cache downloaded data",
    )
    return parser.parse_args()


def load_data(data_dir: str, dataset: str):
    """Load hierarchical tourism dataset."""
    print(f"\n{'='*60}")
    print(f"Loading {dataset} dataset...")
    print(f"{'='*60}")

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Load dataset
    Y_df, S_df, tags = HierarchicalData.load(data_dir, dataset)

    # Ensure datetime format
    Y_df["ds"] = pd.to_datetime(Y_df["ds"])

    # Reset index for S_df
    if "unique_id" not in S_df.columns:
        S_df = S_df.reset_index(names="unique_id")

    print(f"✓ Loaded {len(Y_df['unique_id'].unique())} time series")
    print(f"✓ Date range: {Y_df['ds'].min()} to {Y_df['ds'].max()}")
    print(f"✓ Total observations: {len(Y_df)}")
    print(f"\nHierarchy structure:")
    for tag, series in tags.items():
        print(f"  {tag}: {len(series)} series")

    return Y_df, S_df, tags


def split_data(Y_df: pd.DataFrame, horizon: int):
    """Split data into train and test sets."""
    print(f"\n{'='*60}")
    print(f"Splitting data (forecast horizon = {horizon})...")
    print(f"{'='*60}")

    # Take last 'horizon' periods as test set
    Y_test_df = Y_df.groupby("unique_id").tail(horizon)
    Y_train_df = Y_df.drop(Y_test_df.index)

    print(f"✓ Training set: {len(Y_train_df)} observations")
    print(f"✓ Test set: {len(Y_test_df)} observations")

    return Y_train_df, Y_test_df


def generate_base_forecasts(
    Y_train_df: pd.DataFrame,
    base_models: list,
    horizon: int,
    season_length: int,
):
    """Generate base forecasts and fitted values using classical methods."""
    print(f"\n{'='*60}")
    print(f"Generating base forecasts...")
    print(f"{'='*60}")

    # Initialize models
    models = []
    if "AutoETS" in base_models:
        models.append(AutoETS(season_length=season_length))
        print("✓ AutoETS (automatic exponential smoothing)")
    if "AutoARIMA" in base_models:
        models.append(AutoARIMA(season_length=season_length))
        print("✓ AutoARIMA (automatic ARIMA)")

    # Generate forecasts and fitted values
    print(f"\nForecasting {horizon} periods ahead...")
    sf = StatsForecast(models=models, freq="ME", n_jobs=-1)

    # Get forecasts
    Y_hat_df = sf.forecast(df=Y_train_df, h=horizon, level=[90], fitted=True)

    # Get fitted values (in-sample predictions) from the fitted results
    Y_fitted_df = sf.forecast_fitted_values()

    print(f"✓ Generated forecasts for {len(Y_hat_df['unique_id'].unique())} series")
    print(f"✓ Generated fitted values for reconciliation")

    return Y_hat_df, Y_fitted_df


def reconcile_forecasts(
    Y_hat_df: pd.DataFrame,
    Y_fitted_df: pd.DataFrame,
    Y_train_df: pd.DataFrame,
    S_df: pd.DataFrame,
    tags: dict,
    reconcilers: list,
):
    """Apply reconciliation methods to ensure hierarchy coherence."""
    print(f"\n{'='*60}")
    print(f"Applying reconciliation methods...")
    print(f"{'='*60}")

    # Parse reconciler specifications
    reconciler_objs = []
    for rec_name in reconcilers:
        if rec_name == "BottomUp":
            reconciler_objs.append(BottomUp())
            print("✓ BottomUp (aggregate from bottom level)")
        elif rec_name == "MinTrace_ols":
            reconciler_objs.append(MinTrace(method="ols"))
            print("✓ MinTrace with OLS (ordinary least squares)")
        elif rec_name == "MinTrace_shrink":
            reconciler_objs.append(MinTrace(method="mint_shrink"))
            print("✓ MinTrace with shrinkage (robust covariance)")
        elif rec_name == "MinTrace_wls":
            reconciler_objs.append(MinTrace(method="wls_var"))
            print("✓ MinTrace with WLS (weighted least squares)")
        else:
            print(f"⚠ Unknown reconciler: {rec_name}, skipping")

    # Apply reconciliation
    print(f"\nReconciling forecasts...")
    hrec = HierarchicalReconciliation(reconcilers=reconciler_objs)
    Y_rec_df = hrec.reconcile(
        Y_hat_df=Y_hat_df,
        Y_df=Y_fitted_df,  # Use fitted values for covariance estimation
        S_df=S_df,
        tags=tags,
    )

    print(f"✓ Reconciliation complete")

    return Y_rec_df


def evaluate_forecasts(
    Y_test_df: pd.DataFrame,
    Y_rec_df: pd.DataFrame,
    tags: dict,
):
    """Evaluate forecast accuracy using RMSE and MASE."""
    print(f"\n{'='*60}")
    print(f"Evaluating forecast accuracy...")
    print(f"{'='*60}")

    # Prepare test data
    Y_test_df = Y_test_df.rename(columns={"y": "y_test"})

    # Merge forecasts with test data
    evaluation_df = Y_test_df.merge(Y_rec_df, on=["unique_id", "ds"], how="left")

    # Evaluate using HierarchicalEvaluation
    evaluator = HierarchicalEvaluation(evaluators=[])

    # Get forecast columns (exclude base forecasts, keep reconciled only)
    forecast_cols = [col for col in Y_rec_df.columns
                    if col not in ["unique_id", "ds"]
                    and "/" in col]  # Reconciled forecasts have "/" in name

    # Also include base forecasts for comparison
    base_cols = [col for col in Y_rec_df.columns
                if col not in ["unique_id", "ds"]
                and "/" not in col]

    all_cols = base_cols + forecast_cols

    # Calculate metrics manually for better control
    results = {}

    for col in all_cols:
        if col in evaluation_df.columns:
            # Calculate RMSE
            rmse = ((evaluation_df["y_test"] - evaluation_df[col]) ** 2).mean() ** 0.5

            # Calculate MAE
            mae = (evaluation_df["y_test"] - evaluation_df[col]).abs().mean()

            # Calculate MAPE (avoid division by zero)
            mape = (
                ((evaluation_df["y_test"] - evaluation_df[col]).abs()
                 / evaluation_df["y_test"].replace(0, 1))
                .mean() * 100
            )

            results[col] = {
                "RMSE": rmse,
                "MAE": mae,
                "MAPE": mape,
            }

    # Convert to DataFrame for nice display
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values("RMSE")

    print("\n" + "="*80)
    print("OVERALL FORECAST ACCURACY (sorted by RMSE)")
    print("="*80)
    print(results_df.to_string())
    print("="*80)

    # Calculate accuracy by hierarchy level
    print("\n" + "="*80)
    print("ACCURACY BY HIERARCHY LEVEL")
    print("="*80)

    level_results = {}
    for tag, series_list in tags.items():
        level_df = evaluation_df[evaluation_df["unique_id"].isin(series_list)]
        level_metrics = {}

        for col in all_cols:
            if col in level_df.columns:
                rmse = ((level_df["y_test"] - level_df[col]) ** 2).mean() ** 0.5
                mae = (level_df["y_test"] - level_df[col]).abs().mean()
                level_metrics[col] = {"RMSE": rmse, "MAE": mae}

        level_results[tag] = level_metrics

    # Display level results
    for tag, metrics in level_results.items():
        if metrics:
            print(f"\n{tag} ({len(tags[tag])} series):")
            level_df = pd.DataFrame(metrics).T
            level_df = level_df.sort_values("RMSE")
            print(level_df.to_string())

    print("="*80)

    return results_df, level_results


def save_results(
    Y_rec_df: pd.DataFrame,
    results_df: pd.DataFrame,
    level_results: dict,
    args,
    output_dir: str,
):
    """Save forecasts and evaluation results."""
    print(f"\n{'='*60}")
    print(f"Saving results to {output_dir}...")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # Save reconciled forecasts
    forecast_path = Path(output_dir) / "reconciled_forecasts.csv"
    Y_rec_df.to_csv(forecast_path, index=False)
    print(f"✓ Saved forecasts: {forecast_path}")

    # Save overall metrics
    metrics_path = Path(output_dir) / "metrics.csv"
    results_df.to_csv(metrics_path)
    print(f"✓ Saved metrics: {metrics_path}")

    # Save level-wise metrics as JSON
    level_metrics_path = Path(output_dir) / "metrics_by_level.json"
    with open(level_metrics_path, "w") as f:
        # Convert nested dict to serializable format
        serializable_results = {}
        for level, methods in level_results.items():
            serializable_results[level] = {
                method: {k: float(v) for k, v in metrics.items()}
                for method, metrics in methods.items()
            }
        json.dump(serializable_results, f, indent=2)
    print(f"✓ Saved level metrics: {level_metrics_path}")

    # Save configuration
    config_path = Path(output_dir) / "config.json"
    config = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "season_length": args.season_length,
        "base_models": args.base_models,
        "reconcilers": args.reconcilers,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved config: {config_path}")

    print(f"\n{'='*60}")
    print("✓ Hierarchical forecasting complete!")
    print(f"{'='*60}\n")


def main():
    args = parse_args()

    print("\n" + "="*80)
    print("HIERARCHICAL TIME SERIES FORECASTING WITH RECONCILIATION")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Horizon: {args.horizon} periods")
    print(f"Season: {args.season_length} periods")
    print(f"Base models: {', '.join(args.base_models)}")
    print(f"Reconcilers: {', '.join(args.reconcilers)}")
    print("="*80)

    # 1. Load data
    Y_df, S_df, tags = load_data(args.data_dir, args.dataset)

    # 2. Split data
    Y_train_df, Y_test_df = split_data(Y_df, args.horizon)

    # 3. Generate base forecasts
    Y_hat_df, Y_fitted_df = generate_base_forecasts(
        Y_train_df, args.base_models, args.horizon, args.season_length
    )

    # 4. Reconcile forecasts
    Y_rec_df = reconcile_forecasts(
        Y_hat_df, Y_fitted_df, Y_train_df, S_df, tags, args.reconcilers
    )

    # 5. Evaluate
    results_df, level_results = evaluate_forecasts(Y_test_df, Y_rec_df, tags)

    # 6. Save results
    save_results(Y_rec_df, results_df, level_results, args, args.output_dir)


if __name__ == "__main__":
    main()
