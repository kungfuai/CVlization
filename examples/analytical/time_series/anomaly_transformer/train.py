"""
Training script for Anomaly Transformer.
Implements minimax training strategy with association discrepancy.
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from model import AnomalyTransformer
from data_factory.data_loader import load_dataset, get_loader_segment, download_data


def parse_args():
    parser = argparse.ArgumentParser(description='Anomaly Transformer Training')

    # Dataset
    parser.add_argument('--dataset', type=str, default='SMAP',
                       choices=['SMAP', 'MSL', 'SMD', 'PSM'],
                       help='Dataset name')
    parser.add_argument('--data-dir', type=str,
                       default='/root/.cache/cvlization/anomaly_data',
                       help='Data directory')

    # Model
    parser.add_argument('--d-model', type=int, default=256,
                       help='Model dimension')
    parser.add_argument('--n-heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--e-layers', type=int, default=3,
                       help='Number of encoder layers')
    parser.add_argument('--d-ff', type=int, default=512,
                       help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout rate')
    parser.add_argument('--activation', type=str, default='gelu',
                       help='Activation function')

    # Training
    parser.add_argument('--win-size', type=int, default=100,
                       help='Window size')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of data loading workers')

    # Output
    parser.add_argument('--output-dir', type=str, default='./artifacts',
                       help='Output directory for checkpoints')

    # GPU
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device')

    return parser.parse_args()


def association_discrepancy(series, prior):
    """
    Compute association discrepancy between series-association and prior-association.

    Args:
        series: (B, H, L, L) series-association (learned attention)
        prior: (B, L, L) prior-association (Gaussian kernel)

    Returns:
        discrepancy: Scalar loss value
    """
    # Average over heads
    series = series.mean(dim=1)  # (B, L, L)

    # Compute KL divergence
    # Add small epsilon for numerical stability
    series = torch.clamp(series, min=1e-8, max=1.0)
    prior = torch.clamp(prior, min=1e-8, max=1.0)

    # KL(series || prior)
    kl = (series * (torch.log(series) - torch.log(prior))).sum(dim=-1).mean()

    return kl


def train_epoch(model, train_loader, optimizer, criterion, device, k=3):
    """
    Train for one epoch with minimax strategy.

    Args:
        model: AnomalyTransformer model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss criterion (MSE for reconstruction)
        device: Device
        k: Number of minimax iterations

    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    count = 0

    with tqdm(train_loader, desc='Training') as pbar:
        for batch_x, _ in pbar:
            batch_x = batch_x.float().to(device)

            # Minimax strategy
            for i in range(k):
                # Forward pass
                reconstruction, series_list, prior_list = model(batch_x)

                # Reconstruction loss
                rec_loss = criterion(reconstruction, batch_x)

                # Association discrepancy (across all layers)
                disc_loss = 0
                for series, prior in zip(series_list, prior_list):
                    if series is not None and prior is not None:
                        disc_loss += association_discrepancy(series, prior)

                disc_loss = disc_loss / len(series_list)

                # Minimax: Phase 1 - Maximize discrepancy (update model)
                if i < k - 1:
                    loss = -disc_loss
                # Phase 2 - Minimize reconstruction + discrepancy (final update)
                else:
                    loss = rec_loss - disc_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += rec_loss.item()
            count += 1

            pbar.set_postfix({'loss': rec_loss.item()})

    return total_loss / count


def main():
    args = parse_args()
    print(f"\n{'='*80}")
    print(f"ANOMALY TRANSFORMER TRAINING")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Window size: {args.win_size}")
    print(f"Model dimension: {args.d_model}")
    print(f"Encoder layers: {args.e_layers}")
    print(f"Epochs: {args.epochs}")
    print(f"{'='*80}\n")

    # Download and load data
    download_data(args.data_dir)
    train_data, train_labels, _, _ = load_dataset(args.dataset, args.data_dir)

    # Normalize data
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    std = np.where(std == 0, 1, std)  # Avoid division by zero
    train_data = (train_data - mean) / std

    print(f"\nCreating training data loader...")
    train_loader = get_loader_segment(
        train_data, train_labels,
        batch_size=args.batch_size,
        win_size=args.win_size,
        step=1,
        mode='train'
    )
    print(f"✓ Training batches: {len(train_loader)}")

    # Initialize model
    n_features = train_data.shape[1]
    print(f"\nInitializing Anomaly Transformer...")
    print(f"  Input features: {n_features}")

    model = AnomalyTransformer(
        win_size=args.win_size,
        enc_in=n_features,
        c_out=n_features,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        activation=args.activation,
        output_attention=True
    ).to(args.device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training loop
    print(f"\n{'='*80}")
    print(f"TRAINING")
    print(f"{'='*80}\n")

    best_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
        print(f"  Average Loss: {loss:.6f}")

        # Save best model
        if loss < best_loss:
            best_loss = loss
            os.makedirs(args.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.output_dir,
                                          f'{args.dataset}_checkpoint.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'mean': mean,
                'std': std,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint: {checkpoint_path}")

    print(f"\n{'='*80}")
    print(f"✓ Training complete!")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
