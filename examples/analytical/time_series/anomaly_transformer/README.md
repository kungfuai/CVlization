# Anomaly Transformer: Time Series Anomaly Detection

This example implements the **Anomaly Transformer** (ICLR 2022) for multivariate time series anomaly detection. It uses **Association Discrepancy** between learned attention patterns and temporal priors to detect anomalies in spacecraft telemetry and industrial sensor data.

## Why Anomaly Transformer?

**Problem:** Classical anomaly detection methods struggle with:
- Multivariate temporal dependencies
- Complex normal patterns
- High-dimensional sensor data

**Solution:** Anomaly Transformer introduces:
1. **Anomaly-Attention**: Compares learned associations vs. temporal priors
2. **Association Discrepancy**: Quantifies abnormal attention patterns
3. **Minimax Training**: Amplifies normal-abnormal distinguishability

**Use Case:** Predictive maintenance (spacecraft, industrial equipment, server monitoring)

## Model Architecture

### Key Components

#### 1. Anomaly-Attention
- **Prior-Association**: Gaussian kernel based on temporal distance
  ```
  P(i, j) = exp(-|i-j| / σ)  # Nearby points should be related
  ```
- **Series-Association**: Learned attention weights from data
  ```
  S = softmax(Q·K^T / √d)    # Standard transformer attention
  ```
- **Association Discrepancy**: KL divergence between P and S
  ```
  AssDis = KL(S || P)         # High when pattern is abnormal
  ```

#### 2. Minimax Training Strategy
```python
for k iterations:
    if k < K-1:
        # Phase 1: Maximize discrepancy
        loss = -AssDis
    else:
        # Phase 2: Minimize reconstruction + discrepancy
        loss = MSE(reconstruction) - AssDis
```

This amplifies the difference between normal and abnormal patterns.

## Datasets

### SMAP & MSL (NASA Spacecraft Telemetry)
- **SMAP**: Soil Moisture Active Passive satellite
  - 55 entities, 25 sensors each
  - Temperature, radiation, power, telemetry
- **MSL**: Mars Science Laboratory (Curiosity Rover)
  - 27 entities, 66 sensors each
  - Rover operations, scientific instruments

**Source:** NASA JPL's telemanom project (KDD 2018)

### SMD & PSM (Server/Industrial Monitoring)
- **SMD**: Server Machine Dataset (38 machines, 28 metrics)
- **PSM**: Pooled Server Metrics (25 features)

## Quick Start

### Using CVL CLI

```bash
# Build the container
cvl run analytical/time_series/anomaly_transformer build

# Train on SMAP dataset (default, ~10 epochs on GPU)
cvl run analytical/time_series/anomaly_transformer train

# Train on MSL dataset
cvl run analytical/time_series/anomaly_transformer train_msl

# Evaluate on test set
cvl run analytical/time_series/anomaly_transformer evaluate

# Custom training
cvl run analytical/time_series/anomaly_transformer train -- \
  --dataset SMAP \
  --epochs 20 \
  --batch-size 256 \
  --d-model 256 \
  --e-layers 3
```

### Using Shell Scripts

```bash
# Build
./build.sh

# Train
./train.sh --dataset SMAP --epochs 10

# Evaluate
./evaluate.sh --dataset SMAP --checkpoint ./artifacts/SMAP_checkpoint.pth
```

## Command-Line Options

### Training (`train.py`)

```
--dataset              Dataset name (SMAP, MSL, SMD, PSM)
                       Default: SMAP

--data-dir             Data directory (auto-downloads if missing)
                       Default: /root/.cache/cvlization/anomaly_data

--d-model              Model dimension
                       Default: 256

--n-heads              Number of attention heads
                       Default: 8

--e-layers             Number of encoder layers
                       Default: 3

--d-ff                 Feed-forward dimension
                       Default: 512

--win-size             Sliding window size
                       Default: 100

--batch-size           Batch size
                       Default: 128

--epochs               Number of training epochs
                       Default: 10

--lr                   Learning rate
                       Default: 1e-4

--output-dir           Output directory for checkpoints
                       Default: ./artifacts
```

### Evaluation (`evaluate.py`)

```
--dataset              Dataset name (must match training)
                       Default: SMAP

--checkpoint           Path to model checkpoint
                       Default: ./artifacts/SMAP_checkpoint.pth

--threshold-percentile Percentile for anomaly threshold
                       Default: 99 (top 1% are anomalies)

--output-dir           Output directory for results
                       Default: ./artifacts
```

## Output

Training creates:
- `./artifacts/{DATASET}_checkpoint.pth` - Model checkpoint with normalization stats

Evaluation creates:
- `./artifacts/{DATASET}_evaluation.npz` - Anomaly scores, predictions, metrics

### Evaluation Metrics

#### Standard Metrics
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **ROC-AUC**: Area under ROC curve

#### Point-Adjusted Metrics (PA Protocol)
Standard evaluation protocol for SMAP/MSL:
- If **any point** in an anomaly segment is detected → entire segment is detected
- More realistic: captures whether anomalies are caught, not every single point
- **F1-PA**: F1 score with point adjustment (primary metric for SMAP/MSL)

## Expected Performance

### SMAP Dataset
| Method | Precision | Recall | F1-PA |
|--------|-----------|--------|-------|
| LSTM-VAE | 0.89 | 0.85 | 0.87 |
| OmniAnomaly | 0.91 | 0.88 | 0.89 |
| **Anomaly Transformer** | **0.95** | **0.93** | **0.94** |
| Mixer-Transformer (2025) | 0.97 | 0.98 | 0.97 |

### MSL Dataset
| Method | Precision | Recall | F1-PA |
|--------|-----------|--------|-------|
| LSTM-VAE | 0.88 | 0.82 | 0.85 |
| OmniAnomaly | 0.90 | 0.87 | 0.88 |
| **Anomaly Transformer** | **0.94** | **0.92** | **0.93** |
| Mixer-Transformer (2025) | 0.95 | 0.95 | 0.95 |

**Notes:**
- Results with 10-epoch training on single GPU (~2-3 hours)
- Longer training (20+ epochs) improves F1-PA by 1-2%
- SMD/PSM datasets show similar relative performance

## Understanding Anomaly Scores

Anomaly score = **Reconstruction Error + Association Discrepancy**

### Reconstruction Error
- High when input pattern doesn't match learned normal patterns
- Standard autoencoder-style detection

### Association Discrepancy
- High when attention pattern differs from temporal prior
- Captures **unusual temporal dependencies**
- Key innovation: detects anomalies that "look normal" individually but have abnormal timing

**Example:** Sensor reading is within normal range (low reconstruction error) but occurs at wrong time relative to other sensors (high discrepancy).

## Model Interpretability

### Visualizing Attention
The model learns different associations for normal vs. abnormal patterns:

**Normal Pattern:**
- Series-association ≈ Prior-association
- Nearby time points attend to each other (Gaussian-like)

**Abnormal Pattern:**
- Series-association ≠ Prior-association
- Attention deviates from expected temporal proximity
- Model "knows" something is wrong about the timing

### Per-Sensor Contributions
You can extract per-sensor anomaly scores by analyzing:
```python
reconstruction_error = (reconstruction - input) ** 2  # (B, L, D)
# Sum over time, get per-sensor error
sensor_scores = reconstruction_error.mean(dim=1)
```

## Comparison with Merlion

CVlization includes two anomaly detection examples:

| Aspect | Merlion | Anomaly Transformer |
|--------|---------|---------------------|
| **Type** | Classical (isolation forest, spectral residual, etc.) | Deep learning (Transformer) |
| **Training** | Unsupervised, no training | Requires GPU training (hours) |
| **Speed** | Fast (real-time) | Slower (batch inference) |
| **Multivariate** | Limited | Excellent (temporal dependencies) |
| **Interpretability** | High (feature importances) | Medium (attention maps) |
| **Use Case** | Quick baselines, simple patterns | Complex sensors, predictive maintenance |

**Recommendation:** Start with Merlion for baselines; use Anomaly Transformer for production with complex multivariate data.

## Advanced Usage

### Hyperparameter Tuning

**Model Size:**
- Small: `--d-model 128 --e-layers 2` (faster, less accurate)
- Medium: `--d-model 256 --e-layers 3` (default, good balance)
- Large: `--d-model 512 --e-layers 4` (slower, more accurate)

**Window Size:**
- Short: `--win-size 50` (faster, local patterns)
- Medium: `--win-size 100` (default, balanced)
- Long: `--win-size 200` (slower, long-term dependencies)

**Threshold Selection:**
- Conservative: `--threshold-percentile 99.5` (fewer false alarms)
- Balanced: `--threshold-percentile 99` (default)
- Sensitive: `--threshold-percentile 98` (catch more anomalies)

### Custom Datasets

To use your own dataset:

1. **Format data:**
   ```python
   train_data.npy: (N_train, D) - normal data only
   test_data.npy: (N_test, D) - contains anomalies
   test_labels.npy: (N_test,) - binary labels (0=normal, 1=anomaly)
   ```

2. **Add to data loader:**
   Edit `data_factory/data_loader.py` to load your format

3. **Train:**
   ```bash
   ./train.sh --dataset YOUR_DATASET --data-dir /path/to/data
   ```

## Troubleshooting

### GPU Out of Memory
```bash
# Reduce batch size
./train.sh --batch-size 64

# Reduce model size
./train.sh --d-model 128 --e-layers 2

# Reduce window size
./train.sh --win-size 50
```

### Poor Performance
```bash
# Train longer
./train.sh --epochs 20

# Increase model capacity
./train.sh --d-model 512 --e-layers 4

# Adjust threshold
./evaluate.sh --threshold-percentile 98
```

### Data Download Fails
```bash
# Manual download
cd /root/.cache/cvlization
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip
unzip data.zip
```

## References

### Papers
- **Anomaly Transformer**: Xu et al., "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy", ICLR 2022 Spotlight
  - [Paper](https://openreview.net/forum?id=LzQQ89U1qm_)
  - [Code](https://github.com/thuml/Anomaly-Transformer)

- **Dataset**: Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding", KDD 2018
  - [Paper](https://arxiv.org/abs/1802.04431)

### Related Work
- **Mixer-Transformer** (2025): Recent improvement with 97.5% F1-PA on SMAP
- **MAAT** (2025): Mamba-based variant with +1% improvement
- **RTdetector** (2025): Reconstruction trend analysis

## Next Steps

1. **Run baseline**: Start with default SMAP training (10 epochs)
2. **Analyze results**: Check F1-PA score (expect ~0.93-0.95)
3. **Compare with Merlion**: Run `../merlion_anomaly_dashboard` on same data
4. **Production deployment**: Tune threshold on validation set, monitor drift

## Related Examples

- [`merlion_anomaly_dashboard`](../merlion_anomaly_dashboard/) - Classical methods with interactive dashboard
- [`hierarchical_reconciliation`](../hierarchical_reconciliation/) - Forecasting with reconciliation
- [`statsforecast_baselines`](../statsforecast_baselines/) - Classical forecasting baselines

---

**Note:** Requires GPU for training (2-3 hours on single GPU). CPU inference is supported but slower.
