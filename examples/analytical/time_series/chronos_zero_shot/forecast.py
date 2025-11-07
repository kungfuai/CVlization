"""
Zero-shot time series forecasting with Amazon Chronos models.
Supports Chronos-Bolt (fast), Chronos-2 (latest), and Chronos-T5 (original).
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from chronos import BaseChronosPipeline
from datasetsforecast.m4 import M4


def parse_args():
    parser = argparse.ArgumentParser(
        description='Zero-shot forecasting with Amazon Chronos'
    )

    # Model
    parser.add_argument('--model', type=str, default='amazon/chronos-bolt-small',
                       help='Chronos model variant (chronos-bolt-small/base, '
                            'chronos-2, chronos-t5-small/base/large)')

    # Dataset
    parser.add_argument('--dataset', type=str, default='m4_hourly',
                       choices=['m4_hourly', 'm4_daily', 'm4_weekly', 'm4_monthly'],
                       help='Dataset name')
    parser.add_argument('--data-dir', type=str,
                       default='/root/.cache/cvlization/chronos_data',
                       help='Data directory')

    # Forecasting
    parser.add_argument('--prediction-length', type=int, default=None,
                       help='Forecast horizon (uses dataset default if not specified)')
    parser.add_argument('--context-length', type=int, default=None,
                       help='Historical context length (auto if not specified)')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of sample paths for quantile estimation')

    # Evaluation
    parser.add_argument('--max-series', type=int, default=10,
                       help='Maximum number of series to evaluate (for faster testing)')

    # Output
    parser.add_argument('--output-dir', type=str, default='./artifacts',
                       help='Output directory for results')

    # Device
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda or cpu)')

    return parser.parse_args()


def load_dataset(dataset_name, data_dir, max_series=None):
    """
    Load M4 dataset.

    Returns:
        Y_df: Full dataset
        horizon: Forecast horizon
        frequency: Frequency string
        season_length: Seasonal period
    """
    print(f"\n{'='*80}")
    print(f"Loading {dataset_name} dataset...")
    print(f"{'='*80}")

    os.makedirs(data_dir, exist_ok=True)

    # M4 dataset
    freq_map = {
        'm4_hourly': 'Hourly',
        'm4_daily': 'Daily',
        'm4_weekly': 'Weekly',
        'm4_monthly': 'Monthly'
    }
    freq = freq_map[dataset_name]

    # Load data
    Y_df, *_ = M4.load(directory=data_dir, group=freq)

    # M4 default horizons and seasons
    horizon_map = {'Hourly': 48, 'Daily': 14, 'Weekly': 13, 'Monthly': 18}
    season_map = {'Hourly': 24, 'Daily': 7, 'Weekly': 1, 'Monthly': 12}

    horizon = horizon_map[freq]
    season_length = season_map[freq]
    frequency = {'Hourly': 'H', 'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M'}[freq]

    # Sample series if specified
    if max_series is not None:
        unique_ids = Y_df['unique_id'].unique()
        if len(unique_ids) > max_series:
            sample_ids = np.random.choice(unique_ids, size=max_series, replace=False)
            Y_df = Y_df[Y_df['unique_id'].isin(sample_ids)]

    print(f"✓ Loaded {len(Y_df['unique_id'].unique())} series")
    print(f"✓ Total observations: {len(Y_df)}")
    print(f"✓ Horizon: {horizon} periods")
    print(f"✓ Season length: {season_length}")
    print(f"✓ Frequency: {frequency}")

    return Y_df, horizon, frequency, season_length


def prepare_forecasting_data(Y_df, horizon):
    """
    Split data into train and test sets.

    Returns:
        train_df: Training data (historical context)
        test_df: Test data (ground truth for evaluation)
    """
    print(f"\n{'='*80}")
    print(f"Preparing data for forecasting...")
    print(f"{'='*80}")

    # Split by taking last horizon points as test
    test_df = Y_df.groupby('unique_id').tail(horizon)
    train_df = Y_df.drop(test_df.index)

    print(f"✓ Training observations: {len(train_df)}")
    print(f"✓ Test observations: {len(test_df)}")

    return train_df, test_df


def generate_forecasts(pipeline, train_df, horizon, context_length, num_samples, device):
    """
    Generate forecasts using Chronos pipeline.

    Returns:
        forecasts_dict: Dictionary mapping unique_id to forecast array
    """
    print(f"\n{'='*80}")
    print(f"Generating forecasts...")
    print(f"{'='*80}")

    forecasts_dict = {}
    unique_ids = train_df['unique_id'].unique()

    with tqdm(unique_ids, desc='Forecasting') as pbar:
        for uid in pbar:
            # Get historical data for this series
            series_data = train_df[train_df['unique_id'] == uid]['y'].values

            # Limit context if specified
            if context_length is not None and len(series_data) > context_length:
                series_data = series_data[-context_length:]

            # Convert to tensor
            context = torch.tensor(series_data, dtype=torch.float32).to(device)

            # Generate forecast
            # Output shape: [num_samples, prediction_length]
            # Note: Chronos-Bolt doesn't support num_samples parameter
            try:
                forecast = pipeline.predict(
                    context=context,
                    prediction_length=horizon,
                    num_samples=num_samples
                )
            except TypeError:
                # Chronos-Bolt interface - returns quantiles directly
                forecast = pipeline.predict(
                    context=context,
                    prediction_length=horizon
                )

            # Convert to numpy and store
            forecasts_dict[uid] = forecast.cpu().numpy()

    print(f"✓ Generated forecasts for {len(forecasts_dict)} series")

    return forecasts_dict


def evaluate_forecasts(forecasts_dict, test_df, horizon):
    """
    Evaluate forecast accuracy using standard metrics.

    Returns:
        metrics: Dictionary of aggregated metrics
        series_metrics: DataFrame of per-series metrics
    """
    print(f"\n{'='*80}")
    print(f"Evaluating forecasts...")
    print(f"{'='*80}")

    series_results = []

    for uid in forecasts_dict.keys():
        # Get ground truth
        y_true = test_df[test_df['unique_id'] == uid]['y'].values

        # Get forecast (median of samples or quantiles)
        forecast_samples = forecasts_dict[uid]  # (num_samples, horizon) or (num_quantiles, horizon)

        # Check if this is quantile forecast (Chronos-Bolt) or sample forecast
        if forecast_samples.ndim == 2:
            if forecast_samples.shape[0] < 20:  # Likely quantiles (typically 9)
                # Use median quantile (middle one)
                median_idx = forecast_samples.shape[0] // 2
                y_pred = forecast_samples[median_idx, :]  # (horizon,)
            else:  # Sample forecast
                y_pred = np.median(forecast_samples, axis=0)  # (horizon,)
        else:
            # Already 1D
            y_pred = forecast_samples

        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        if min_len == 0:
            continue

        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        # Calculate metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # sMAPE
        smape = 100 * np.mean(
            2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
        )

        # MASE (using naive baseline)
        naive_mae = np.mean(np.abs(y_true))
        mase = mae / (naive_mae + 1e-8)

        series_results.append({
            'unique_id': uid,
            'MAE': mae,
            'RMSE': rmse,
            'sMAPE': smape,
            'MASE': mase
        })

    # Create DataFrame
    series_metrics = pd.DataFrame(series_results)

    # Aggregate metrics
    metrics = {
        'MAE': series_metrics['MAE'].mean(),
        'RMSE': series_metrics['RMSE'].mean(),
        'sMAPE': series_metrics['sMAPE'].mean(),
        'MASE': series_metrics['MASE'].mean(),
        'num_series': len(series_metrics)
    }

    print(f"\n{'='*80}")
    print(f"FORECAST ACCURACY")
    print(f"{'='*80}")
    print(f"MAE:    {metrics['MAE']:.4f}")
    print(f"RMSE:   {metrics['RMSE']:.4f}")
    print(f"sMAPE:  {metrics['sMAPE']:.2f}%")
    print(f"MASE:   {metrics['MASE']:.4f}")
    print(f"{'='*80}")

    return metrics, series_metrics


def main():
    args = parse_args()

    print(f"\n{'='*80}")
    print(f"CHRONOS ZERO-SHOT FORECASTING")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Max series: {args.max_series}")
    print(f"{'='*80}")

    # Load dataset
    Y_df, horizon, frequency, season_length = load_dataset(
        args.dataset, args.data_dir, args.max_series
    )

    # Override horizon if specified
    if args.prediction_length is not None:
        horizon = args.prediction_length
        print(f"Using custom horizon: {horizon}")

    # Prepare data
    train_df, test_df = prepare_forecasting_data(Y_df, horizon)

    # Load Chronos pipeline
    print(f"\n{'='*80}")
    print(f"Loading Chronos model: {args.model}")
    print(f"{'='*80}")

    pipeline = BaseChronosPipeline.from_pretrained(
        args.model,
        device_map=args.device,
        torch_dtype=torch.bfloat16 if args.device == 'cuda' else torch.float32,
    )

    print(f"✓ Model loaded successfully")

    # Generate forecasts
    forecasts_dict = generate_forecasts(
        pipeline, train_df, horizon,
        args.context_length, args.num_samples, args.device
    )

    # Evaluate
    metrics, series_metrics = evaluate_forecasts(forecasts_dict, test_df, horizon)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Save aggregate metrics
    metrics_path = os.path.join(args.output_dir, f'{args.dataset}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Saved metrics: {metrics_path}")

    # Save per-series metrics
    series_path = os.path.join(args.output_dir, f'{args.dataset}_series_metrics.csv')
    series_metrics.to_csv(series_path, index=False)
    print(f"✓ Saved series metrics: {series_path}")

    # Save config
    config_path = os.path.join(args.output_dir, 'config.json')
    config = {
        'model': args.model,
        'dataset': args.dataset,
        'horizon': horizon,
        'context_length': args.context_length,
        'num_samples': args.num_samples,
        'max_series': args.max_series,
        'device': args.device
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved config: {config_path}")

    print(f"\n{'='*80}")
    print(f"✓ Forecasting complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
