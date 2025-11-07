"""
Evaluation script for Anomaly Transformer.
Computes anomaly scores and evaluates with Precision, Recall, F1 metrics.
"""

import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from model import AnomalyTransformer
from data_factory.data_loader import load_dataset, get_loader_segment, download_data


def parse_args():
    parser = argparse.ArgumentParser(description='Anomaly Transformer Evaluation')

    # Dataset
    parser.add_argument('--dataset', type=str, default='SMAP',
                       choices=['SMAP', 'MSL', 'SMD', 'PSM'],
                       help='Dataset name')
    parser.add_argument('--data-dir', type=str,
                       default='/root/.cache/cvlization/anomaly_data',
                       help='Data directory')

    # Model checkpoint
    parser.add_argument('--checkpoint', type=str,
                       default='./artifacts/SMAP_checkpoint.pth',
                       help='Path to model checkpoint')

    # Evaluation
    parser.add_argument('--win-size', type=int, default=100,
                       help='Window size')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--threshold-percentile', type=float, default=99,
                       help='Percentile for anomaly threshold (e.g., 99 = top 1%%)')

    # Output
    parser.add_argument('--output-dir', type=str, default='./artifacts',
                       help='Output directory for results')

    # GPU
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device')

    return parser.parse_args()


def compute_anomaly_score(model, test_loader, device):
    """
    Compute anomaly scores for test data.

    Uses reconstruction error + association discrepancy as anomaly score.

    Args:
        model: Trained AnomalyTransformer
        test_loader: Test data loader
        device: Device

    Returns:
        anomaly_scores: (N,) numpy array of anomaly scores per time point
        test_labels: (N,) numpy array of ground truth labels
    """
    model.eval()

    reconstruction_errors = []
    discrepancy_scores = []
    all_labels = []

    with torch.no_grad():
        with tqdm(test_loader, desc='Computing anomaly scores') as pbar:
            for batch_x, batch_y in pbar:
                batch_x = batch_x.float().to(device)

                # Forward pass
                reconstruction, series_list, prior_list = model(batch_x)

                # Reconstruction error (per point)
                rec_error = torch.mean((reconstruction - batch_x) ** 2, dim=-1)  # (B, L)

                # Association discrepancy
                disc_score = torch.zeros_like(rec_error)
                for series, prior in zip(series_list, prior_list):
                    if series is not None and prior is not None:
                        # Average over heads
                        series = series.mean(dim=1)  # (B, L, L)

                        # Compute per-point discrepancy
                        series = torch.clamp(series, min=1e-8, max=1.0)
                        prior = torch.clamp(prior, min=1e-8, max=1.0)

                        # KL divergence per point
                        kl = (series * (torch.log(series) - torch.log(prior))).sum(dim=-1)  # (B, L)
                        disc_score += kl

                disc_score = disc_score / len(series_list)

                reconstruction_errors.append(rec_error.cpu().numpy())
                discrepancy_scores.append(disc_score.cpu().numpy())
                all_labels.append(batch_y.numpy())

    # Concatenate and flatten
    reconstruction_errors = np.concatenate(reconstruction_errors, axis=0)  # (N_windows, win_size)
    discrepancy_scores = np.concatenate(discrepancy_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Flatten to get per-point scores
    reconstruction_errors = reconstruction_errors.reshape(-1)
    discrepancy_scores = discrepancy_scores.reshape(-1)
    test_labels = all_labels.reshape(-1)

    # Combined anomaly score (reconstruction + discrepancy)
    anomaly_scores = reconstruction_errors + discrepancy_scores

    return anomaly_scores, test_labels


def point_adjustment(labels, predictions):
    """
    Apply point adjustment (PA) protocol.

    If any point in an anomaly segment is detected, count the entire segment as detected.
    This is the standard evaluation protocol for SMAP/MSL datasets.

    Args:
        labels: (N,) binary ground truth labels
        predictions: (N,) binary predictions

    Returns:
        adjusted_predictions: (N,) adjusted binary predictions
    """
    # Find anomaly segments
    segments = []
    start = None
    for i in range(len(labels)):
        if labels[i] == 1 and start is None:
            start = i
        elif labels[i] == 0 and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(labels) - 1))

    # Adjust predictions: if any point in segment is detected, mark entire segment
    adjusted_predictions = predictions.copy()
    for start, end in segments:
        if predictions[start:end+1].sum() > 0:
            adjusted_predictions[start:end+1] = 1

    return adjusted_predictions


def evaluate_metrics(labels, predictions, scores):
    """
    Compute evaluation metrics.

    Args:
        labels: Ground truth binary labels
        predictions: Predicted binary labels
        scores: Anomaly scores (for ROC-AUC)

    Returns:
        metrics: Dictionary of metrics
    """
    metrics = {}

    # Standard metrics (without point adjustment)
    metrics['precision'] = precision_score(labels, predictions, zero_division=0)
    metrics['recall'] = recall_score(labels, predictions, zero_division=0)
    metrics['f1'] = f1_score(labels, predictions, zero_division=0)

    # Point-adjusted metrics
    adjusted_predictions = point_adjustment(labels, predictions)
    metrics['precision_pa'] = precision_score(labels, adjusted_predictions, zero_division=0)
    metrics['recall_pa'] = recall_score(labels, adjusted_predictions, zero_division=0)
    metrics['f1_pa'] = f1_score(labels, adjusted_predictions, zero_division=0)

    # ROC-AUC
    try:
        metrics['roc_auc'] = roc_auc_score(labels, scores)
    except:
        metrics['roc_auc'] = 0.0

    return metrics


def main():
    args = parse_args()
    print(f"\n{'='*80}")
    print(f"ANOMALY TRANSFORMER EVALUATION")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'='*80}\n")

    # Load checkpoint
    print(f"Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    checkpoint_args = argparse.Namespace(**checkpoint['args'])

    # Download and load data
    download_data(args.data_dir)
    _, _, test_data, test_labels = load_dataset(args.dataset, args.data_dir)

    # Normalize using training statistics
    mean = checkpoint['mean']
    std = checkpoint['std']
    test_data = (test_data - mean) / std

    print(f"\nCreating test data loader...")
    # Use step=win_size for test to avoid overlap
    test_loader = get_loader_segment(
        test_data, test_labels,
        batch_size=args.batch_size,
        win_size=checkpoint_args.win_size,
        step=checkpoint_args.win_size,
        mode='test',
        shuffle=False
    )
    print(f"✓ Test batches: {len(test_loader)}")

    # Initialize model
    n_features = test_data.shape[1]
    print(f"\nInitializing model...")
    model = AnomalyTransformer(
        win_size=checkpoint_args.win_size,
        enc_in=n_features,
        c_out=n_features,
        d_model=checkpoint_args.d_model,
        n_heads=checkpoint_args.n_heads,
        e_layers=checkpoint_args.e_layers,
        d_ff=checkpoint_args.d_ff,
        dropout=checkpoint_args.dropout,
        activation=checkpoint_args.activation,
        output_attention=True
    ).to(args.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")

    # Compute anomaly scores
    print(f"\n{'='*80}")
    print(f"COMPUTING ANOMALY SCORES")
    print(f"{'='*80}\n")

    anomaly_scores, labels = compute_anomaly_score(model, test_loader, args.device)

    # Threshold at percentile
    threshold = np.percentile(anomaly_scores, args.threshold_percentile)
    predictions = (anomaly_scores > threshold).astype(int)

    print(f"✓ Anomaly scores computed")
    print(f"  Threshold ({args.threshold_percentile}th percentile): {threshold:.6f}")
    print(f"  Predicted anomalies: {predictions.sum()}/{len(predictions)} "
          f"({100*predictions.mean():.2f}%)")
    print(f"  True anomalies: {labels.sum()}/{len(labels)} "
          f"({100*labels.mean():.2f}%)")

    # Evaluate metrics
    print(f"\n{'='*80}")
    print(f"EVALUATION METRICS")
    print(f"{'='*80}\n")

    metrics = evaluate_metrics(labels[:len(predictions)], predictions, anomaly_scores)

    print(f"Standard Metrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    print(f"\nPoint-Adjusted Metrics (PA Protocol):")
    print(f"  Precision: {metrics['precision_pa']:.4f}")
    print(f"  Recall:    {metrics['recall_pa']:.4f}")
    print(f"  F1 Score:  {metrics['f1_pa']:.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f'{args.dataset}_evaluation.npz')
    np.savez(results_path,
             anomaly_scores=anomaly_scores,
             predictions=predictions,
             labels=labels[:len(predictions)],
             metrics=metrics)
    print(f"\n✓ Results saved: {results_path}")

    print(f"\n{'='*80}")
    print(f"✓ Evaluation complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
