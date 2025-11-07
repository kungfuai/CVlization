"""
Data loader for SMAP, MSL, SMD, and PSM anomaly detection datasets.
Adapted from thuml/Anomaly-Transformer.
"""

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import subprocess


def download_data(data_dir):
    """Download SMAP/MSL/SMD/PSM datasets if not present."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    # Check if data already exists
    required_files = ['SMAP', 'MSL', 'SMD', 'PSM']
    if all(os.path.exists(os.path.join(data_dir, f)) for f in required_files):
        print("✓ Datasets already downloaded")
        return

    print("Downloading datasets from telemanom...")
    # Download and extract
    zip_path = os.path.join(data_dir, 'data.zip')
    subprocess.run([
        'wget', '-O', zip_path,
        'https://s3-us-west-2.amazonaws.com/telemanom/data.zip'
    ], check=True)

    subprocess.run(['unzip', '-o', zip_path, '-d', data_dir], check=True)
    subprocess.run(['rm', zip_path], check=True)
    print("✓ Datasets downloaded successfully")


class SegmentDataset(Dataset):
    """Sliding window dataset for anomaly detection."""

    def __init__(self, data, labels, win_size, step=1):
        """
        Args:
            data: (N, D) numpy array of time series
            labels: (N,) numpy array of binary labels
            win_size: Window size for sliding window
            step: Step size for sliding window
        """
        self.data = data
        self.labels = labels
        self.win_size = win_size
        self.step = step

        # Create sliding windows
        self.segments = []
        self.seg_labels = []

        for i in range(0, len(data) - win_size + 1, step):
            segment = data[i:i + win_size]
            label = labels[i:i + win_size]
            self.segments.append(segment)
            self.seg_labels.append(label)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        """
        Returns:
            segment: (win_size, n_features) tensor
            label: (win_size,) tensor of binary labels
        """
        return self.segments[idx], self.seg_labels[idx]


def load_dataset(dataset_name, data_dir):
    """
    Load SMAP, MSL, SMD, or PSM dataset.

    Args:
        dataset_name: One of ['SMAP', 'MSL', 'SMD', 'PSM']
        data_dir: Root directory containing datasets

    Returns:
        train_data: (N_train, D) numpy array
        train_labels: (N_train,) numpy array (all zeros for training)
        test_data: (N_test, D) numpy array
        test_labels: (N_test,) numpy array (binary anomaly labels)
    """
    dataset_path = os.path.join(data_dir, dataset_name)

    if dataset_name in ['SMAP', 'MSL']:
        # Load train data
        train_files = [f for f in os.listdir(os.path.join(dataset_path, 'train'))
                      if f.endswith('.npy')]
        train_data_list = []
        for f in sorted(train_files):
            data = np.load(os.path.join(dataset_path, 'train', f))
            train_data_list.append(data)
        train_data = np.concatenate(train_data_list, axis=0)
        train_labels = np.zeros(len(train_data))

        # Load test data
        test_files = [f for f in os.listdir(os.path.join(dataset_path, 'test'))
                     if f.endswith('.npy')]
        test_data_list = []
        for f in sorted(test_files):
            data = np.load(os.path.join(dataset_path, 'test', f))
            test_data_list.append(data)
        test_data = np.concatenate(test_data_list, axis=0)

        # Load test labels
        labels_path = os.path.join(dataset_path, 'labeled_anomalies.csv')
        labels_df = pd.read_csv(labels_path)

        # Create binary label array
        test_labels = np.zeros(len(test_data))
        for _, row in labels_df.iterrows():
            # Mark anomaly segments
            start_idx = int(row['start'])
            end_idx = int(row['end'])
            test_labels[start_idx:end_idx+1] = 1

    elif dataset_name == 'SMD':
        # Load SMD dataset (server machine dataset)
        train_path = os.path.join(dataset_path, 'train.npy')
        test_path = os.path.join(dataset_path, 'test.npy')
        labels_path = os.path.join(dataset_path, 'test_label.npy')

        train_data = np.load(train_path)
        test_data = np.load(test_path)
        test_labels = np.load(labels_path)
        train_labels = np.zeros(len(train_data))

    elif dataset_name == 'PSM':
        # Load PSM dataset
        train_path = os.path.join(dataset_path, 'train.npy')
        test_path = os.path.join(dataset_path, 'test.npy')
        labels_path = os.path.join(dataset_path, 'test_label.npy')

        train_data = np.load(train_path)
        test_data = np.load(test_path)
        test_labels = np.load(labels_path)
        train_labels = np.zeros(len(train_data))

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"✓ Loaded {dataset_name} dataset")
    print(f"  Train: {train_data.shape}, Test: {test_data.shape}")
    print(f"  Features: {train_data.shape[1]}")
    print(f"  Anomalies in test: {test_labels.sum()}/{len(test_labels)} "
          f"({100*test_labels.mean():.2f}%)")

    return train_data, train_labels, test_data, test_labels


def get_loader_segment(data, labels, batch_size, win_size=100, step=1,
                       mode='train', shuffle=True):
    """
    Create DataLoader for sliding window segments.

    Args:
        data: (N, D) numpy array
        labels: (N,) numpy array
        batch_size: Batch size
        win_size: Window size for sliding window
        step: Step size (use 1 for train, win_size for test to avoid overlap)
        mode: 'train' or 'test'
        shuffle: Whether to shuffle

    Returns:
        DataLoader
    """
    dataset = SegmentDataset(data, labels, win_size, step)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 for simplicity
        drop_last=False
    )

    return loader
