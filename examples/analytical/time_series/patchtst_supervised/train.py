"""
Supervised time series forecasting with PatchTST.

PatchTST is a state-of-the-art Transformer-based model that achieves 20%+ improvement
over other Transformer models by using patching and channel-independence.
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    PatchTSTConfig,
    PatchTSTForPrediction,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Supervised forecasting with PatchTST'
    )

    # Dataset
    parser.add_argument('--dataset', type=str, default='ETTh1',
                       choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2',
                               'weather', 'electricity', 'traffic'],
                       help='Dataset name')
    parser.add_argument('--data-dir', type=str,
                       default='/root/.cache/cvlization/ltsf_data',
                       help='Data directory')

    # Forecasting
    parser.add_argument('--context-length', type=int, default=512,
                       help='Historical context length')
    parser.add_argument('--prediction-length', type=int, default=96,
                       help='Forecast horizon')
    parser.add_argument('--patch-length', type=int, default=16,
                       help='Patch length (must divide context-length evenly)')

    # Model architecture
    parser.add_argument('--d-model', type=int, default=128,
                       help='Model dimension')
    parser.add_argument('--num-layers', type=int, default=3,
                       help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=16,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')

    # Training
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help='Early stopping patience')

    # Output
    parser.add_argument('--output-dir', type=str, default='./artifacts',
                       help='Output directory for results')

    # Device
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda or cpu)')

    return parser.parse_args()


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting with PatchTST."""

    def __init__(self, data, context_length, prediction_length, scaler=None, fit_scaler=False):
        """
        Args:
            data: DataFrame with time series columns
            context_length: Historical window size
            prediction_length: Forecast horizon
            scaler: StandardScaler instance
            fit_scaler: Whether to fit the scaler
        """
        self.context_length = context_length
        self.prediction_length = prediction_length

        # Extract values (exclude timestamp column if present)
        if 'date' in data.columns:
            self.values = data.drop('date', axis=1).values
        else:
            self.values = data.values

        # Scale data
        if fit_scaler:
            self.scaler = StandardScaler()
            self.values = self.scaler.fit_transform(self.values)
        elif scaler is not None:
            self.scaler = scaler
            self.values = self.scaler.transform(self.values)
        else:
            self.scaler = None

        # Create samples
        self.samples = []
        total_length = context_length + prediction_length

        for i in range(len(self.values) - total_length + 1):
            past = self.values[i:i + context_length]
            future = self.values[i + context_length:i + total_length]
            self.samples.append((past, future))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        past_values, future_values = self.samples[idx]
        return {
            'past_values': torch.FloatTensor(past_values),
            'future_values': torch.FloatTensor(future_values)
        }


def load_dataset(dataset_name, data_dir):
    """
    Load LTSF benchmark dataset.

    Returns:
        train_df, val_df, test_df: DataFrames with time series data
        num_features: Number of features/channels
        frequency: Frequency string
    """
    print(f"\n{'='*80}")
    print(f"Loading {dataset_name} dataset...")
    print(f"{'='*80}")

    os.makedirs(data_dir, exist_ok=True)

    # Dataset URLs and info
    dataset_info = {
        'ETTh1': {
            'url': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv',
            'freq': 'H',
            'features': 7
        },
        'ETTh2': {
            'url': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv',
            'freq': 'H',
            'features': 7
        },
        'ETTm1': {
            'url': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv',
            'freq': '15T',
            'features': 7
        },
        'ETTm2': {
            'url': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv',
            'freq': '15T',
            'features': 7
        },
        'weather': {
            'url': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/WTH/WTH.csv',
            'freq': '10T',
            'features': 21
        },
        'electricity': {
            'url': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ECL/ECL.csv',
            'freq': 'H',
            'features': 321
        },
        'traffic': {
            'url': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/traffic/traffic.csv',
            'freq': 'H',
            'features': 862
        }
    }

    info = dataset_info[dataset_name]
    csv_path = os.path.join(data_dir, f'{dataset_name}.csv')

    # Download if not exists
    if not os.path.exists(csv_path):
        print(f"Downloading {dataset_name} dataset...")
        import urllib.request
        urllib.request.urlretrieve(info['url'], csv_path)
        print(f"✓ Downloaded to {csv_path}")

    # Load data
    df = pd.read_csv(csv_path)

    # Standard split ratios for LTSF benchmarks
    n = len(df)
    train_end = int(0.7 * n)
    val_end = int(0.9 * n)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"✓ Loaded {dataset_name}")
    print(f"✓ Total observations: {len(df)}")
    print(f"✓ Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"✓ Features: {info['features']}")
    print(f"✓ Frequency: {info['freq']}")

    return train_df, val_df, test_df, info['features'], info['freq']


def create_model(num_features, args):
    """Create PatchTST model with configuration."""
    print(f"\n{'='*80}")
    print(f"Creating PatchTST model...")
    print(f"{'='*80}")

    # Validate patch_length divides context_length
    if args.context_length % args.patch_length != 0:
        raise ValueError(
            f"patch_length ({args.patch_length}) must divide "
            f"context_length ({args.context_length}) evenly"
        )

    config = PatchTSTConfig(
        # Data dimensions
        num_input_channels=num_features,
        context_length=args.context_length,
        prediction_length=args.prediction_length,

        # Patching
        patch_length=args.patch_length,
        patch_stride=args.patch_length,  # Non-overlapping patches

        # Architecture
        d_model=args.d_model,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        ffn_dim=args.d_model * 4,  # Standard Transformer ratio

        # Regularization
        dropout=args.dropout,
        head_dropout=args.dropout,
        attention_dropout=args.dropout,

        # Normalization
        norm_type='batchnorm',
        pre_norm=True,

        # Scaling
        scaling='std',
        loss='mse'
    )

    model = PatchTSTForPrediction(config)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ Model created")
    print(f"✓ Parameters: {num_params:,}")
    print(f"✓ Context length: {args.context_length}")
    print(f"✓ Prediction length: {args.prediction_length}")
    print(f"✓ Patch length: {args.patch_length}")
    print(f"✓ Num patches: {args.context_length // args.patch_length}")

    return model, config


def compute_metrics(eval_pred):
    """Compute MSE and MAE metrics."""
    predictions, labels = eval_pred
    # predictions shape: (batch, prediction_length, features)
    mse = np.mean((predictions - labels) ** 2)
    mae = np.mean(np.abs(predictions - labels))
    return {'mse': mse, 'mae': mae}


def train_model(model, train_dataset, val_dataset, args):
    """Train PatchTST model with HuggingFace Trainer."""
    print(f"\n{'='*80}")
    print(f"Training model...")
    print(f"{'='*80}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,

        # Evaluation
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='mse',
        greater_is_better=False,

        # Logging
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_strategy='epoch',
        report_to='none',  # Disable wandb/tensorboard

        # Performance
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        remove_unused_columns=False,

        # Save
        save_total_limit=2,
    )

    # Custom trainer to properly pass labels
    class PatchTSTTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            past_values = inputs['past_values']
            future_values = inputs['future_values']

            outputs = model(
                past_values=past_values,
                future_values=future_values
            )
            loss = outputs.loss

            return (loss, outputs) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            """Custom prediction step to return predictions for compute_metrics."""
            past_values = inputs['past_values']
            future_values = inputs['future_values']

            with torch.no_grad():
                outputs = model(
                    past_values=past_values,
                    future_values=future_values
                )
                loss = outputs.loss
                predictions = outputs.prediction_outputs

            return (loss, predictions, future_values)

    trainer = PatchTSTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        ]
    )

    # Train
    trainer.train()

    print(f"\n✓ Training complete!")

    return trainer


def evaluate_model(model, test_dataset, args):
    """Evaluate model on test set."""
    print(f"\n{'='*80}")
    print(f"Evaluating on test set...")
    print(f"{'='*80}")

    model.eval()
    device = torch.device(args.device)
    model.to(device)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Inference'):
            past_values = batch['past_values'].to(device)
            future_values = batch['future_values']

            # Forward pass
            outputs = model(past_values=past_values)
            pred = outputs.prediction_outputs

            predictions.append(pred.cpu().numpy())
            targets.append(future_values.numpy())

    # Concatenate all batches
    predictions = np.concatenate(predictions, axis=0)  # (N, pred_len, features)
    targets = np.concatenate(targets, axis=0)

    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)

    # Per-feature metrics
    mse_per_feature = np.mean((predictions - targets) ** 2, axis=(0, 1))
    mae_per_feature = np.mean(np.abs(predictions - targets), axis=(0, 1))

    metrics = {
        'MSE': float(mse),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MSE_per_feature': mse_per_feature.tolist(),
        'MAE_per_feature': mae_per_feature.tolist()
    }

    print(f"\n{'='*80}")
    print(f"TEST SET RESULTS")
    print(f"{'='*80}")
    print(f"MSE:  {mse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"{'='*80}")

    return metrics, predictions, targets


def main():
    args = parse_args()

    print(f"\n{'='*80}")
    print(f"PATCHTST SUPERVISED FORECASTING")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Context length: {args.context_length}")
    print(f"Prediction length: {args.prediction_length}")
    print(f"Patch length: {args.patch_length}")
    print(f"Device: {args.device}")
    print(f"{'='*80}")

    # Load data
    train_df, val_df, test_df, num_features, frequency = load_dataset(
        args.dataset, args.data_dir
    )

    # Create datasets
    print(f"\n{'='*80}")
    print(f"Creating datasets...")
    print(f"{'='*80}")

    train_dataset = TimeSeriesDataset(
        train_df, args.context_length, args.prediction_length,
        scaler=None, fit_scaler=True
    )
    scaler = train_dataset.scaler

    val_dataset = TimeSeriesDataset(
        val_df, args.context_length, args.prediction_length,
        scaler=scaler, fit_scaler=False
    )

    test_dataset = TimeSeriesDataset(
        test_df, args.context_length, args.prediction_length,
        scaler=scaler, fit_scaler=False
    )

    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")

    # Create model
    model, config = create_model(num_features, args)

    # Train
    trainer = train_model(model, train_dataset, val_dataset, args)

    # Evaluate
    metrics, predictions, targets = evaluate_model(model, test_dataset, args)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Saved metrics: {metrics_path}")

    # Save config
    config_path = os.path.join(args.output_dir, 'config.json')
    config_dict = {
        'dataset': args.dataset,
        'context_length': args.context_length,
        'prediction_length': args.prediction_length,
        'patch_length': args.patch_length,
        'd_model': args.d_model,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'num_features': num_features,
        'frequency': frequency
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"✓ Saved config: {config_path}")

    # Save sample predictions
    sample_path = os.path.join(args.output_dir, 'sample_predictions.npz')
    np.savez(
        sample_path,
        predictions=predictions[:100],  # Save first 100 samples
        targets=targets[:100]
    )
    print(f"✓ Saved sample predictions: {sample_path}")

    print(f"\n{'='*80}")
    print(f"✓ Training and evaluation complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
